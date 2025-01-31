import json
from abc import ABC
from typing import List, Dict

import numpy as np
import pandas as pd


class SellerStats:
    """Statistics for a seller's dataset and market performance"""
    total_points: int
    points_selected: int = 0
    selection_rate: float = 0.0
    market_share: float = 0.0
    revenue: float = 0.0
    avg_price: float = 0.0


class BaseSeller(ABC):
    """Enhanced base seller class with statistics tracking"""

    def __init__(self,
                 seller_id: str,
                 dataset: np.ndarray,
                 price_strategy: str = 'uniform',
                 base_price: float = 1.0,
                 price_variation: float = 0.2):
        self.seller_id = seller_id
        self.dataset = dataset
        self.price_strategy = price_strategy
        self.base_price = base_price
        self.price_variation = price_variation

        # Initialize statistics
        self.stats = SellerStats()
        self.prices = self._generate_prices()
        self.selection_history: List[Dict] = []
        self.cur_data = self.dataset
        self.cur_price = self.prices

    @property
    def get_data(self):
        return {
            "X": self.cur_data,
            "cost": self.cur_price,
        }

    def _generate_prices(self) -> np.ndarray:
        """Generate prices based on strategy"""
        if self.price_strategy == 'uniform':
            return np.random.uniform(
                self.base_price * (1 - self.price_variation),
                self.base_price * (1 + self.price_variation),
                size=len(self.dataset)
            )
        elif self.price_strategy == 'gaussian':
            return np.abs(np.random.normal(
                self.base_price,
                self.price_variation * self.base_price,
                size=len(self.dataset)
            ))
        else:
            raise ValueError(f"Unknown price strategy: {self.price_strategy}")

    def record_selection(self, indices: List[int], buyer_id: str):
        """Record data points selected by a buyer"""
        selection_record = {
            'buyer_id': buyer_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_points': len(indices),
            'total_cost': sum(self.prices[indices]),
            'indices': indices
        }
        self.selection_history.append(selection_record)

        # Update statistics
        self.stats.points_selected += len(indices)
        self.stats.selection_rate = self.stats.points_selected / self.stats.total_points
        self.stats.revenue += selection_record['total_cost']
        self.stats.avg_price = self.stats.revenue / self.stats.points_selected

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'seller_id': self.seller_id,
            'dataset_size': self.stats.total_points,
            'points_selected': self.stats.points_selected,
            'selection_rate': self.stats.selection_rate,
            'revenue': self.stats.revenue,
            'avg_price': self.stats.avg_price,
            'market_share': self.stats.market_share
        }

    def save_statistics(self, output_path: str):
        """Save statistics and selection history to file"""
        stats_data = {
            'statistics': self.get_statistics(),
            'selection_history': self.selection_history
        }

        with open(output_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
