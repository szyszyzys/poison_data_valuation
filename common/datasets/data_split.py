from abc import abstractmethod, ABC

import numpy as np


class BuyerSplitStrategy(ABC):
    @abstractmethod
    def split(self, available_indices, dataset, buyer_config):
        """Returns (buyer_pool_indices, seller_pool_indices)"""
        pass


class OverallFractionSplit(BuyerSplitStrategy):
    def split(self, available_indices, dataset, buyer_config):
        fraction = buyer_config["buyer_overall_fraction"]
        num_buyer_samples = int(len(available_indices) * fraction)
        return available_indices[:num_buyer_samples], available_indices[num_buyer_samples:]


class CelebAIdentitySplit(BuyerSplitStrategy):
    def split(self, available_indices, dataset, buyer_config):
        # Assumes dataset is the actual CelebACustom instance
        identities = dataset.identity[available_indices].squeeze().numpy()
        buyer_ids = set(range(1, 101))  # Example IDs
        is_buyer_mask = np.isin(identities, list(buyer_ids))
        return available_indices[is_buyer_mask], available_indices[~is_buyer_mask]
