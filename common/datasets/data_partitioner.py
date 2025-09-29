import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from common.datasets.data_split import BuyerSplitStrategy
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

    def _split_buyer_pool(self, buyer_pool_all_data: np.ndarray, buyer_config: Dict):
        """Helper to split a pool of indices into root and test sets."""
        np.random.shuffle(buyer_pool_all_data)
        fraction = buyer_config.get("root_set_fraction", 0.2)
        split_idx = int(len(buyer_pool_all_data) * fraction)
        self.buyer_indices = buyer_pool_all_data[:split_idx]
        self.test_indices = buyer_pool_all_data[split_idx:]
        logger.info(
            f"Buyer pool split: {len(self.buyer_indices)} root, {len(self.test_indices)} test."
        )

    def partition(self, buyer_split_strategy: BuyerSplitStrategy, client_partition_strategy: str, buyer_config: Dict,
                  partition_params: Dict):
        """Main partitioning dispatcher using strategy objects."""
        actual_dataset = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
        available_indices = np.array(
            self.dataset.indices if isinstance(self.dataset, Subset) else np.arange(len(self.dataset)))
        np.random.shuffle(available_indices)

        # Stage 1: Use the strategy object to get buyer/seller pools
        buyer_pool_indices, seller_pool_indices = buyer_split_strategy.split(
            available_indices, actual_dataset, buyer_config
        )

        # Stage 2: Split the buyer's pool into root and test sets
        self._split_buyer_pool(buyer_pool_indices, buyer_config)

        # Stage 3: Partition seller data among clients
        if client_partition_strategy == 'property-skew':
            self._partition_property_skew(seller_pool_indices, PropertySkewParams(**partition_params))
        elif client_partition_strategy == 'dirichlet':
            # --- ADD THIS NEW CASE ---
            alpha = partition_params.get('dirichlet_alpha')
            if alpha is None:
                raise ValueError("Dirichlet strategy requires 'dirichlet_alpha' in partition_params.")
            self.seller_splits = self._partition_by_dirichlet(seller_pool_indices, alpha)

        else:
            raise ValueError(f"Unknown client partitioning strategy: {client_partition_strategy}")

        return self

    def _partition_by_dirichlet(self, seller_pool_indices: np.ndarray, alpha: float) -> Dict[int, List[int]]:
        """
        Partitions data among clients using a Dirichlet distribution to simulate
        Non-IID data heterogeneity.
        """
        logger.info(
            f"Partitioning {len(seller_pool_indices)} samples for {self.num_clients} clients using Dirichlet (alpha={alpha})...")

        # Filter targets to only include those in the seller pool
        pool_targets = self.targets[seller_pool_indices]
        n_classes = len(np.unique(self.targets))

        # Generate the distribution of classes for each client
        # label_distribution shape: (num_classes, num_clients)
        label_distribution = np.random.dirichlet([alpha] * self.num_clients, n_classes)

        # Map class labels to the indices of samples having that label (within the seller pool)
        class_to_indices = {
            label: seller_pool_indices[np.where(pool_targets == label)[0]]
            for label in range(n_classes)
        }

        # This will hold the final index assignments for each client
        client_id_to_indices = {i: [] for i in range(self.num_clients)}

        # Distribute the indices for each class across all clients
        for class_id, indices in class_to_indices.items():
            np.random.shuffle(indices)

            # Get the proportions for this specific class across all clients
            proportions = label_distribution[class_id]

            # Calculate the number of samples for each client, ensuring it sums correctly
            samples_per_client = (proportions * len(indices)).astype(int)
            samples_per_client[-1] = len(indices) - np.sum(samples_per_client[:-1])  # Adjust last client to match total

            start_idx = 0
            for client_id in range(self.num_clients):
                num_samples = samples_per_client[client_id]
                end_idx = start_idx + num_samples
                client_id_to_indices[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        # Final conversion to a dictionary of lists
        final_splits = {cid: list(indices) for cid, indices in client_id_to_indices.items()}
        return final_splits

    def _partition_property_skew(self, seller_pool_indices: np.ndarray, config: PropertySkewParams):
        """Partitions sellers based on the prevalence of a specific data property."""
        wrapped_dataset = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
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
        prop_true_indices, prop_false_indices = [], []
        for idx in seller_pool_indices:
            # No more if/elif! Just call the method directly.
            if wrapped_dataset.has_property(idx, property_key=config.property_key):
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
