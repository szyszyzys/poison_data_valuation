# FILE: common/datasets/tabular_data_processor.py

import os
import pandas as pd
import numpy as np
import logging
from urllib import request
from typing import Dict, Any, Tuple, List

import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from ucimlrepo import fetch_ucirepo

from common.datasets.data_partitioner import TabularDataPartitioner
from common.datasets.image_data_processor import save_data_statistics


def get_dataset_tabular(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    A unified function to load and prepare a tabular dataset from various sources
    based on a configuration dictionary.
    """
    dataset_name = config['name']
    logging.info(f"📦 Loading and preparing the '{dataset_name}' tabular dataset...")

    # --- 1. Fetch data based on its source type ---
    if config['source_type'] == 'uci':
        dataset_obj = fetch_ucirepo(id=config['uci_id'])
        df = pd.concat([dataset_obj.data.features, dataset_obj.data.targets], axis=1)
        df.columns = ["".join(c if c.isalnum() else '_' for c in str(x)) for x in df.columns]

    elif config['source_type'] == 'url':
        df = pd.read_csv(config['url'], header='infer' if config.get('has_header', True) else None)
        if not config.get('has_header', True):
            num_features = len(df.columns) - 1
            df.columns = [f'feature_{i}' for i in range(num_features)] + [config['target_column']]

    elif config['source_type'] == 'numpy':
        local_filename = f"{dataset_name.lower()}.npz"
        if not os.path.exists(local_filename):
            logging.info(f"  - Downloading {dataset_name} dataset...")
            request.urlretrieve(config['url'], local_filename)
            logging.info(f"  - Downloaded to {local_filename}")
        data = np.load(local_filename)
        features, labels = data[config['data_key']], data[config['labels_key']]
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)
        feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feature_columns)
        df[config['target_column']] = labels

    elif config['source_type'] == 'local_csv':
        df = pd.read_csv(config['path'], header='infer' if config.get('has_header', True) else None)
        if not config.get('has_header', True):
            num_features = len(df.columns) - 1
            df.columns = [f'feature_{i}' for i in range(num_features)] + [config['target_column']]

    else:
        raise ValueError(f"Unsupported source_type: {config['source_type']}")

    if 'feature_columns' in config:
        df = df[config['feature_columns']]

    if 'query' in config:
        df = df.query(config['query']).dropna()

    if 'missing_value_placeholder' in config:
        df.replace(config['missing_value_placeholder'], np.nan, inplace=True)
        df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    if 'binarize' in config:
        for column, details in config['binarize'].items():
            positive_value = details['positive_value']
            df[column] = df[column].apply(lambda x: 1 if str(x).strip() == str(positive_value) else 0)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if config['target_column'] in categorical_cols:
        categorical_cols.remove(config['target_column'])
    if config.get('sensitive_column') in categorical_cols:
        categorical_cols.remove(config.get('sensitive_column'))

    logging.info(f"  - ✅ Dataset '{dataset_name}' loaded successfully. Shape: {df.shape}")
    return df, categorical_cols


def get_tabular_dataset_federated(cfg) -> Tuple[DataLoader, Dict[int, DataLoader], DataLoader, Dict, int]:
    """
    A unified function to load, partition, and prepare federated TABULAR datasets.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Federated Tabular Dataset Setup for '{cfg.experiment.dataset_name}' ---")

    # === UPDATE 1: Use dynamic config path ===
    tabular_cfg_path = cfg.data.tabular_config_path
    with open(tabular_cfg_path, 'r') as f:
        all_tabular_configs = yaml.safe_load(f)
    d_cfg = all_tabular_configs[cfg.experiment.dataset_name]

    df, categorical_cols = get_dataset_tabular(config=d_cfg)

    # Preprocess Data
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df.drop(columns=[d_cfg['target_column']])
    y = df[d_cfg['target_column']]

    # Split before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y # Note: stratify assumes classification
    )

    # Scale numerical features
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) # Note: .long() assumes classification
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    num_classes = len(torch.unique(y_train_tensor))

    # Initialize and Run the Partitioner
    partitioner = TabularDataPartitioner(
        features=X_train,
        targets=y_train,
        num_clients=cfg.experiment.n_sellers,
        seed=cfg.seed
    )

    partitioner.partition(
        strategy=cfg.data.tabular.strategy,
        partition_params=cfg.data.tabular.property_skew # Pass the config object/dict
    )
    buyer_indices, seller_splits = partitioner.get_splits() # Unpack only two values

    # Generate Statistics
    stats = save_data_statistics(
        buyer_indices=buyer_indices,
        seller_splits=seller_splits,
        targets=y_train.values, # Use the numpy/pandas values for stats
        output_dir=cfg.experiment.save_path
    )

    # Create Datasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers

    buyer_loader = DataLoader(Subset(train_dataset, buyer_indices), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers) if buyer_indices.size > 0 else None

    seller_loaders = {
        cid: DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)
        for cid, indices in seller_splits.items() if indices
    }

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"✅ Federated tabular dataset setup complete. Using {num_classes} classes.")

    return buyer_loader, seller_loaders, test_loader, stats, num_classes






