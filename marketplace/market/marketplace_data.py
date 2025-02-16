from typing import Dict, Union, Tuple, List

import numpy as np

from attack.privacy_attack.malicious_seller import MaliciousDataSeller
from marketplace.seller.seller import BaseSeller
from marketplace.market.data_market import DataMarketplace
from marketplace.data_selector import DataSelector, SelectionStrategy


class DataMarketplaceData(DataMarketplace):
    def __init__(self, selection_method: str = "frank_wolfe"):
        self.selection_method = selection_method
        self.sellers: Dict[str, Union[BaseSeller, MaliciousDataSeller]] = {}
        self.selector = DataSelector()

    def get_current_market_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Get latest data from all sellers"""
        x_s, y_s, costs, seller_ids = [], [], [], []
        for seller_id, seller in self.sellers.items():
            seller_data = seller.get_data
            if seller_data['X'] is not None:
                x_s.append(seller_data['X'])
                y_s.append(seller_data.get('y', np.zeros(len(seller_data['X']))))
                costs.append(seller_data.get('costs', np.ones(len(seller_data['X']))))
                seller_ids.extend([seller_id] * len(seller_data['X']))

        return (np.vstack(x_s) if x_s else np.array([]),
                np.concatenate(y_s) if y_s else np.array([]),
                np.concatenate(costs) if costs else np.array([]),
                np.array(seller_ids))

    def update_selection(self, s_method: str):
        """Change the selection method."""
        self.selection_method = s_method

    def select_data(self,
                    x_buy: np.ndarray,
                    y_buy: np.ndarray,
                    select_method: SelectionStrategy,
                    num_select: int = 10,
                    **kwargs) -> Dict:
        """Select data using the latest seller data"""
        # Get current market data
        x_s, y_s, costs, seller_ids = self.get_current_market_data()
        if len(x_s) == 0:
            raise ValueError("No data available in marketplace")

        self.selector.set_sell(x_s, y_s, costs)
        weights = self.selector.select_data(x_buy, y_buy, select_method)
        indices = self.selector.get_top_k(weights, num_select, return_indices=True)

        # Record selections for each seller
        selected_seller_ids = [seller_ids[i] for i in indices]
        for seller_id in self.sellers:
            selected_mask = [sid == seller_id for sid in selected_seller_ids]
            if any(selected_mask):
                seller_selections = [indices[i] for i in range(len(selected_mask)) if selected_mask[i]]
                self.sellers[seller_id].record_selection(seller_selections, "buyer")

        return {
            'seller_ids': selected_seller_ids,
            'weights': weights,
            'total_cost': np.sum(costs[indices]) if len(costs) > 0 else 0,
            'indices': indices
        }

    def get_select_info(self,
                        x_buy: np.ndarray,
                        y_buy: np.ndarray,
                        select_method: SelectionStrategy,
                        **kwargs):
        """Select data using the latest seller data, but only return the weights and seller_ids."""
        x_s, y_s, costs, seller_ids = self.get_current_market_data()
        if len(x_s) == 0:
            raise ValueError("No data available in marketplace")

        self.selector.set_sell(x_s, y_s, costs)
        weights = self.selector.select_data(x_buy, y_buy, select_method)
        return weights, seller_ids

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """Register a seller in the marketplace."""
        self.sellers[seller_id] = seller

    def get_market_status(self) -> Dict:
        """Get current market status."""
        return {
            'num_sellers': len(self.sellers),
            'total_datapoints': sum(len(s.get_synthetic_data()['X']) for s in self.sellers.values()),
            'selection_method': self.selection_method,
            'seller_stats': {sid: seller.get_statistics() for sid, seller in self.sellers.items()}
        }
