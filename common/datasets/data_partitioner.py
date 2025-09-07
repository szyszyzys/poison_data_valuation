import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from common.datasets.image_data_processor import Camelyon16Custom
from common.datasets.image_data_processor import CelebACustom
from common.datasets.text_data_processor import get_text_property_indices
from common.gradient_market_configs import PropertySkewParams, TextPropertySkewParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def _extract_targets(dataset: Dataset) -> np.ndarray:
    """
    Extracts a single target label from each sample in a dataset,
    handling Subset wrappers and special cases like CelebA.
    """
    # Case 1: The dataset is a Subset wrapper
    if isinstance(dataset, Subset):
        underlying_dataset = dataset.dataset
        indices = dataset.indices

        # If the underlying dataset is our custom CelebA...
        if isinstance(underlying_dataset, CelebACustom):
            # ...use the specific property attribute as the "target" for statistics.
            prop_idx = underlying_dataset.property_idx
            return underlying_dataset.attr[indices, prop_idx].numpy()

        # For other datasets, try to get targets from the underlying dataset
        if hasattr(underlying_dataset, "targets"):
            return np.array(underlying_dataset.targets)[indices]
        # Fallback for subsets of datasets without a .targets attribute
        return np.array([underlying_dataset[i][1] for i in indices])

    # Case 2: The dataset is the full object itself
    if isinstance(dataset, CelebACustom):
        prop_idx = dataset.property_idx
        return dataset.attr[:, prop_idx].numpy()

    # Fallback for other full datasets
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)

    # Original (and slowest) fallback if no .targets attribute is found
    return np.array([dataset[i][1] for i in range(len(dataset))])


class FederatedDataPartitioner:
    """Handles the partitioning of a dataset for a federated learning scenario."""

    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        self.targets = _extract_targets(dataset)
        self.buyer_indices: np.ndarray = np.array([], dtype=int)
        self.client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
        self.client_properties: Dict[int, str] = {}  # Ground truth for PIA
        random.seed(seed)
        np.random.seed(seed)

    def partition(self, strategy: str, buyer_config: Dict, partition_params: Dict):
        """Main partitioning dispatcher with a new 'overall_percentage' buyer strategy."""
        is_subset = isinstance(self.dataset, Subset)
        actual_dataset = self.dataset.dataset if is_subset else self.dataset

        available_indices = np.array(self.dataset.indices if is_subset else np.arange(len(self.dataset)))
        np.random.shuffle(available_indices)  # Shuffle all available indices once at the start

        self.test_indices = np.array([], dtype=int)

        # --- NEW Stage 1: Define Buyer and Seller Pools based on overall percentage ---
        # This new logic will take precedence if 'buyer_overall_fraction' is provided
        buyer_overall_fraction = buyer_config.get("buyer_overall_fraction")  # e.g., 0.1 for 10% of overall data

        if buyer_overall_fraction is not None and 0 < buyer_overall_fraction < 1:
            logger.info(f"Partitioning buyer pool from {buyer_overall_fraction * 100:.2f}% of overall data.")

            num_overall_buyer_samples = int(len(available_indices) * buyer_overall_fraction)

            # Buyer pool consists of the first 'num_overall_buyer_samples' shuffled indices
            buyer_pool_all_data = available_indices[:num_overall_buyer_samples]
            seller_pool_indices = available_indices[num_overall_buyer_samples:]

            # Now, split the buyer's *own* data into root and test
            buyer_root_fraction_of_pool = buyer_config.get("root_set_fraction", 0.2)  # e.g., 20% of the buyer's pool
            split_idx = int(len(buyer_pool_all_data) * buyer_root_fraction_of_pool)

            self.buyer_indices = buyer_pool_all_data[:split_idx]
            self.test_indices = buyer_pool_all_data[split_idx:]

            logger.info(
                f"Overall Buyer Pool: {len(buyer_pool_all_data)} samples. "
                f"Root Set: {len(self.buyer_indices)} samples. "
                f"Test Set: {len(self.test_indices)} samples."
            )

        # --- Original Stage 1: Metadata-Based Buyer/Seller Pool (Fallback if not using overall_fraction) ---
        else:  # Fallback to original metadata-based partitioning if buyer_overall_fraction is not set
            seller_pool_indices = np.copy(available_indices)  # Initialize for metadata split
            if isinstance(actual_dataset, CelebACustom):
                logger.info("Partitioning CelebA based on 'identity' metadata.")

                identities = actual_dataset.identity[available_indices].squeeze().numpy()
                buyer_ids = set(range(1, 101))  # Example: first 100 identities are buyers

                is_buyer_mask = np.isin(identities, list(buyer_ids))
                buyer_pool_all_data = available_indices[is_buyer_mask]
                seller_pool_indices = available_indices[~is_buyer_mask]

                np.random.shuffle(buyer_pool_all_data)  # Shuffle the buyer pool

                # Split buyer_pool into root and test sets
                buyer_root_fraction_of_pool = buyer_config.get("root_set_fraction",
                                                               0.2)  # e.g., 80% for root, 20% for test
                split_idx = int(len(buyer_pool_all_data) * buyer_root_fraction_of_pool)

                self.buyer_indices = buyer_pool_all_data[:split_idx]
                self.test_indices = buyer_pool_all_data[split_idx:]

                logger.info(
                    f"CelebA buyer pool split: {len(self.buyer_indices)} for root, {len(self.test_indices)} for test.")

            elif isinstance(actual_dataset, Camelyon16Custom):
                logger.info("Partitioning Camelyon16 based on 'center' metadata.")

                meta_subset = actual_dataset.metadata.iloc[available_indices]
                buyer_mask = (meta_subset['center'] == 'Utrecht')
                buyer_pool_all_data = available_indices[buyer_mask]
                seller_pool_indices = available_indices[~buyer_mask]

                np.random.shuffle(buyer_pool_all_data)
                split_idx = int(len(buyer_pool_all_data) * buyer_config.get("root_set_fraction", 0.2))
                self.buyer_indices = buyer_pool_all_data[:split_idx]
                self.test_indices = buyer_pool_all_data[split_idx:]

                logger.info(
                    f"Camelyon16 buyer pool split: {len(self.buyer_indices)} for root, {len(self.test_indices)} for test.")
            else:
                raise ValueError(
                    "No metadata-based partitioning defined for this dataset type, and 'buyer_overall_fraction' not provided.")

        logger.info(
            f"Partitioning {len(seller_pool_indices)} seller samples among {self.num_clients} clients using '{strategy}' strategy.")

        if strategy == 'property-skew':
            self._partition_property_skew(seller_pool_indices, PropertySkewParams(**partition_params))
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

        return self

    def _partition_property_skew(self, seller_pool_indices: np.ndarray, config: PropertySkewParams):
        """Partitions sellers based on the prevalence of a specific data property."""
        actual_dataset = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
        num_low_prevalence = self.num_clients - config.num_high_prevalence_clients - config.num_security_attackers

        # --- Define client groups ---
        client_ids = list(range(self.num_clients))
        np.random.shuffle(client_ids)

        high_clients = client_ids[:config.num_high_prevalence_clients]
        low_clients = client_ids[
                      config.num_high_prevalence_clients: config.num_high_prevalence_clients + num_low_prevalence]
        security_clients = client_ids[config.num_high_prevalence_clients + num_low_prevalence:]

        for cid in high_clients: self.client_properties[cid] = f"High-Prevalence ({config.property_key})"
        for cid in low_clients: self.client_properties[cid] = f"Low-Prevalence ({config.property_key})"
        for cid in security_clients: self.client_properties[cid] = "Security-Attacker (Standard-Prevalence)"

        # --- Separate seller data by property ---
        prop_true_indices, prop_false_indices = [], []
        for idx in seller_pool_indices:
            # Note: The has_property method in the dataset now needs to know the property_key
            if isinstance(actual_dataset, CelebACustom):
                has_prop = actual_dataset.has_property(idx)
            else:
                has_prop = actual_dataset.has_property(idx, property_key=config.property_key)

            if has_prop:
                prop_true_indices.append(idx)
            else:
                prop_false_indices.append(idx)

        np.random.shuffle(prop_true_indices)
        np.random.shuffle(prop_false_indices)

        # --- Distribute data using pointers for efficiency ---
        samples_per_client = len(seller_pool_indices) // self.num_clients
        true_ptr, false_ptr = 0, 0

        def assign_data(client_list, prevalence):
            nonlocal true_ptr, false_ptr
            for client_id in client_list:
                num_prop_true = int(samples_per_client * prevalence)
                num_prop_false = samples_per_client - num_prop_true

                # Assign samples with the property
                end_true = true_ptr + num_prop_true
                self.client_indices[client_id].extend(prop_true_indices[true_ptr:end_true])
                true_ptr = end_true

                # Assign samples without the property
                end_false = false_ptr + num_prop_false
                self.client_indices[client_id].extend(prop_false_indices[false_ptr:end_false])
                false_ptr = end_false

        assign_data(high_clients, config.high_prevalence_ratio)
        assign_data(low_clients, config.low_prevalence_ratio)
        assign_data(security_clients, config.standard_prevalence_ratio)

    def get_splits(self) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray]:
        """Returns the final buyer, client, and test index splits."""
        return self.buyer_indices, self.client_indices, getattr(self, 'test_indices', np.array([]))

    def _partition_text_property_skew(self, seller_pool_indices: np.ndarray, config: TextPropertySkewParams,
                                      raw_seller_dataset: Any, text_field: str):
        """Partitions sellers based on the prevalence of a specific keyword in the text."""
        # Note: This method operates on the raw HF dataset to find properties before subsetting.

        num_low_prevalence = self.num_clients - config.num_high_prevalence_clients - config.num_security_attackers

        # --- Define client groups (same as for images) ---
        client_ids = list(range(self.num_clients))
        np.random.shuffle(client_ids)

        high_clients = client_ids[:config.num_high_prevalence_clients]
        low_clients = client_ids[
                      config.num_high_prevalence_clients: config.num_high_prevalence_clients + num_low_prevalence]
        security_clients = client_ids[config.num_high_prevalence_clients + num_low_prevalence:]

        for cid in high_clients: self.client_properties[cid] = f"High-Prevalence ({config.property_key})"
        for cid in low_clients: self.client_properties[cid] = f"Low-Prevalence ({config.property_key})"
        for cid in security_clients: self.client_properties[cid] = "Security-Attacker (Standard-Prevalence)"

        # --- Get property indices using the new helper ---
        # We scan the raw dataset corresponding to the seller pool to get our lists
        prop_true_indices, prop_false_indices = get_text_property_indices(
            raw_seller_dataset, config.property_key, text_field
        )

        np.random.shuffle(prop_true_indices)
        np.random.shuffle(prop_false_indices)

        # --- Distribute data using pointers (this logic is reusable and identical to your image version!) ---
        samples_per_client = len(seller_pool_indices) // self.num_clients
        true_ptr, false_ptr = 0, 0

        def assign_data(client_list, prevalence):
            nonlocal true_ptr, false_ptr
            for client_id in client_list:
                num_prop_true = int(samples_per_client * prevalence)
                num_prop_false = samples_per_client - num_prop_true

                end_true = true_ptr + num_prop_true
                self.client_indices[client_id].extend(prop_true_indices[true_ptr:end_true])
                true_ptr = end_true

                end_false = false_ptr + num_prop_false
                self.client_indices[client_id].extend(prop_false_indices[false_ptr:end_false])
                false_ptr = end_false

        assign_data(high_clients, config.high_prevalence_ratio)
        assign_data(low_clients, config.low_prevalence_ratio)
        assign_data(security_clients, config.standard_prevalence_ratio)
