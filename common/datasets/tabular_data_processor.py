import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from urllib import request

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from ucimlrepo import fetch_ucirepo

from common.datasets.data_partitioner import TabularDataPartitioner
from common.datasets.image_data_processor import save_data_statistics
from common.gradient_market_configs import AppConfig  # Import your main config

logger = logging.getLogger(__name__)


def _load_and_prepare_tabular_df(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """A unified helper to load and prepare a tabular DataFrame from various sources."""
    dataset_name = config['name']
    logger.info(f"📦 Loading and preparing the '{dataset_name}' tabular dataset...")

    # --- Fetch data based on its source type ---
    if config['source_type'] == 'uci':
        dataset_obj = fetch_ucirepo(id=config['uci_id'])
        df = pd.concat([dataset_obj.data.features, dataset_obj.data.targets], axis=1)
        df.columns = ["".join(c if c.isalnum() else '_' for c in str(x)) for x in df.columns]
    elif config['source_type'] in ['url', 'local_csv']:
        path = config['url'] if config['source_type'] == 'url' else config['path']
        df = pd.read_csv(path, header='infer' if config.get('has_header', True) else None)
    elif config['source_type'] == 'numpy':
        local_filename = f"{dataset_name.lower()}.npz"
        if not os.path.exists(local_filename):
            logger.info(f"  - Downloading {dataset_name} dataset from {config['url']}...")
            request.urlretrieve(config['url'], local_filename)
        data = np.load(local_filename)
        features, labels = data[config['data_key']], data[config['labels_key']]
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        df[config['target_column']] = labels
    else:
        raise ValueError(f"Unsupported source_type: {config['source_type']}")

    # --- Apply generic preprocessing steps from config ---
    if 'query' in config: df = df.query(config['query']).dropna()
    if 'missing_value_placeholder' in config:
        df.replace(config['missing_value_placeholder'], np.nan, inplace=True)
        df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if config['target_column'] in categorical_cols:
        categorical_cols.remove(config['target_column'])

    logger.info(f"  - ✅ Dataset '{dataset_name}' loaded successfully. Shape: {df.shape}")
    return df, categorical_cols


def get_tabular_dataset(cfg: AppConfig):
    logger.info(f"--- Starting Federated Tabular Dataset Setup for '{cfg.experiment.dataset_name}' ---")

    # 1. Load dataset-specific configuration and raw data
    with open(cfg.data.tabular.dataset_config_path, 'r') as f:
        all_tabular_configs = yaml.safe_load(f)
    d_cfg = all_tabular_configs[cfg.experiment.dataset_name]
    df, categorical_cols = _load_and_prepare_tabular_df(config=d_cfg)
    target_col = d_cfg['target_column']

    # 2. Preprocess Data (Dummify, Scale, Split)
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )

    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=np.number).columns

    # --- ADD THIS DEBUG LOG ---
    if numerical_cols.empty:
        logger.error("CRITICAL BUG: No numerical columns found for scaling!")
    else:
        logger.info(f"Applying StandardScaler to {len(numerical_cols)} columns.")
    # --- END DEBUG LOG ---

    if not numerical_cols.empty:
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    feature_to_idx = {col: i for i, col in enumerate(X_train.columns)}

    # 3. Convert to PyTorch Datasets
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    # --- ADD THIS FIX ---
    # Check for and clean NaN/Inf values created by the scaler
    if torch.isnan(X_train_tensor).any() or torch.isinf(X_train_tensor).any():
        logger.warning("NaN/Inf detected in training data after scaling. Cleaning with nan_to_num(0.0)...")
        # Replaces all NaN, +Inf, and -Inf with 0.0
        X_train_tensor = torch.nan_to_num(X_train_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # --- AND ADD THIS FIX ---
    if torch.isnan(X_test_tensor).any() or torch.isinf(X_test_tensor).any():
        logger.warning("NaN/Inf detected in test data after scaling. Cleaning with nan_to_num(0.0)...")
        X_test_tensor = torch.nan_to_num(X_test_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    input_dim = X_train_tensor.shape[1]
    num_classes = len(torch.unique(y_test_tensor))

    # 4. Partition the training data using the consistent partitioner
    partitioner = TabularDataPartitioner(
        dataset=train_dataset,
        features=X_train,
        targets=y_train,
        num_clients=cfg.experiment.n_sellers,
        seed=cfg.seed
    )
    partitioner.partition(
        client_partition_strategy=cfg.data.tabular.strategy,
        partition_params=cfg.data.tabular.property_skew,
        buyer_fraction=cfg.data.tabular.buyer_ratio
    )
    buyer_indices, seller_splits, _ = partitioner.get_splits()
    # --- ADD THIS SECTION to generate a dedicated path ---
    logger.info("Generating and saving tabular data split statistics...")

    # Create a unique key for the tabular split
    tabular_split_params = {
        "dataset": cfg.experiment.dataset_name,
        "n_sellers": cfg.experiment.n_sellers,
        "seed": cfg.seed,
        "strategy": cfg.data.tabular.strategy,
        "buyer_ratio": cfg.data.tabular.buyer_ratio,
        "partition_params": cfg.data.tabular.property_skew
    }
    config_string = json.dumps(tabular_split_params, sort_keys=True)
    config_hash = hashlib.md5(config_string.encode('utf-8')).hexdigest()

    stats_dir = Path(cfg.data_root) / "data_statistics"
    stats_save_path = stats_dir / f"{config_hash}_stats.json"
    # --- END ADDITION ---

    # Pass the new path to the function
    save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        client_properties={},
        targets=y_train.values,
        save_filepath=stats_save_path  # Use the new path
    )

    # 5. Create final DataLoaders
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers

    buyer_loader = DataLoader(
        Subset(train_dataset, buyer_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    seller_loaders = {
        f"{cid}": DataLoader(
            Subset(train_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        for cid, indices in seller_splits.items() if len(indices) > 0
    }
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle is False for testing
        num_workers=num_workers
    )

    logger.info(f"✅ Federated tabular dataset setup complete.")
    return buyer_loader, seller_loaders, test_loader, num_classes, input_dim, feature_to_idx
