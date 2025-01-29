from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

from daved.src.utils import get_data


class DatasetManager:
    """Manages dataset loading, preprocessing, and distribution among sellers"""

    def __init__(self, 
                 dataset_type: str = "gaussian",
                 data_dir: str = "./data",
                 random_state: int = 0,
                 num_seller: int = 10000,
                 num_buyer: int = 100,
                 num_val: int = 100,
                 dim: int = 100,
                 noise_level: float = 1.0,
                 cost_range: Optional[List[float]] = None,
                 cost_func: str = "linear"):
        
        self.dataset_type = dataset_type
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.seller_data = {}
        
        # Load data using get_data function
        self.data = get_data(
            dataset=dataset_type,
            data_dir=data_dir,
            random_state=random_state,
            num_seller=num_seller,
            num_buyer=num_buyer,
            num_val=num_val,
            dim=dim,
            noise_level=noise_level,
            cost_range=cost_range,
            cost_func=cost_func
        )
        
        # Extract components
        self.X_sell = self.data["X_sell"]
        self.y_sell = self.data["y_sell"]
        self.costs_sell = self.data.get("costs_sell")
        self.index_sell = self.data.get("index_sell")
        self.img_paths = self.data.get("img_paths")
        
        # Buyer data
        self.X_buy = self.data["X_buy"]
        self.y_buy = self.data["y_buy"]
        self.index_buy = self.data.get("index_buy")
        
        # Validation data
        self.X_val = self.data.get("X_val")
        self.y_val = self.data.get("y_val")
        self.index_val = self.data.get("index_val")

    def allocate_data_to_sellers(self, 
                               seller_configs: List[Dict],
                               adversary_ratio: float = 0.1,
                               overlap_allowed: bool = False) -> Dict[str, Dict]:
        """
        Allocate data to sellers based on configurations
        
        Parameters:
        - seller_configs: List of seller configurations with 'id' and 'type' ('normal' or 'adversary')
        - adversary_ratio: Ratio of data to allocate to adversarial sellers
        - overlap_allowed: Whether sellers can have overlapping data
        
        Returns:
        - Dictionary mapping seller IDs to their allocated data
        """
        np.random.seed(self.random_state)
        n_samples = len(self.X_sell)
        
        # Separate adversarial and normal sellers
        adversary_sellers = [s for s in seller_configs if s['type'] == 'adversary']
        normal_sellers = [s for s in seller_configs if s['type'] == 'normal']
        
        # Calculate sizes
        adv_size = int(n_samples * adversary_ratio)
        normal_size = n_samples - adv_size
        
        # Create index pools
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)
        
        adv_indices = all_indices[:adv_size]
        normal_indices = all_indices[adv_size:]
        
        allocations = {}
        
        # Allocate to adversaries
        if adversary_sellers:
            indices_per_adv = adv_size // len(adversary_sellers)
            for i, seller in enumerate(adversary_sellers):
                start_idx = i * indices_per_adv
                end_idx = start_idx + indices_per_adv
                seller_indices = adv_indices[start_idx:end_idx]
                
                allocations[seller['id']] = {
                    'X': self.X_sell[seller_indices],
                    'y': self.y_sell[seller_indices],
                    'costs': self.costs_sell[seller_indices] if self.costs_sell is not None else None,
                    'indices': self.index_sell[seller_indices] if self.index_sell is not None else None,
                    'img_paths': [self.img_paths[i] for i in seller_indices] if self.img_paths is not None else None
                }
        
        # Allocate to normal sellers
        if normal_sellers:
            indices_per_normal = normal_size // len(normal_sellers)
            for i, seller in enumerate(normal_sellers):
                if overlap_allowed:
                    # Random selection with possible overlap
                    seller_indices = np.random.choice(normal_indices, size=indices_per_normal)
                else:
                    # Sequential allocation without overlap
                    start_idx = i * indices_per_normal
                    end_idx = start_idx + indices_per_normal
                    seller_indices = normal_indices[start_idx:end_idx]
                
                allocations[seller['id']] = {
                    'X': self.X_sell[seller_indices],
                    'y': self.y_sell[seller_indices],
                    'costs': self.costs_sell[seller_indices] if self.costs_sell is not None else None,
                    'indices': self.index_sell[seller_indices] if self.index_sell is not None else None,
                    'img_paths': [self.img_paths[i] for i in seller_indices] if self.img_paths is not None else None
                }
        
        self.seller_data = allocations
        return allocations

    def get_buyer_data(self) -> Dict:
        """Get buyer data"""
        return {
            'X': self.X_buy,
            'y': self.y_buy,
            'indices': self.index_buy
        }

    def get_validation_data(self) -> Dict:
        """Get validation data"""
        return {
            'X': self.X_val,
            'y': self.y_val,
            'indices': self.index_val
        }

    def get_seller_data(self, seller_id: str) -> Optional[Dict]:
        """Get data for specific seller"""
        return self.seller_data.get(seller_id)

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_sellers': len(self.seller_data),
            'num_buyer_samples': len(self.X_buy),
            'num_val_samples': len(self.X_val) if self.X_val is not None else 0,
            'feature_dim': self.X_sell.shape[1],
            'has_costs': self.costs_sell is not None,
            'has_images': self.img_paths is not None
        }
