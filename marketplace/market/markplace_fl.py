import numpy as np
from typing import Dict, Union, List, Tuple

from marketplace.market.data_market import DataMarketplace


class DataMarketplaceFederated(DataMarketplace):
    def __init__(self,
                 aggregator: BaseAggregator,
                 selection_method: str = "fedavg",
                 learning_rate: float = 1.0):
        """
        A marketplace for federated learning where each seller provides gradient updates.

        :param aggregator: An object that holds the global model and
                           implements gradient aggregation (e.g., FedAvg).
        :param selection_method: e.g. "fedavg", "krum", "median", etc.
        :param learning_rate: Step size for updating global model parameters.
        """
        self.aggregator = aggregator
        self.selection_method = selection_method
        self.learning_rate = learning_rate

        # Each seller might be a BaseSeller or an AdversarySeller, etc.
        self.sellers: Dict[str, Union[BaseSeller, AdversarySeller]] = {}

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """
        Register a new seller that can provide gradients.
        """
        self.sellers[seller_id] = seller

    def update_selection(self, new_method: str):
        """
        Update the aggregation/selection method, e.g., from 'fedavg' to 'krum'.
        """
        self.selection_method = new_method

    def get_current_market_gradients(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Collect gradient updates from each seller for the current global model parameters.

        :return:
            gradients: List of gradient vectors (np.ndarray)
            sizes: List of integers indicating the local data size from each seller
            seller_ids: List of seller IDs in the same order as the gradient list
        """
        gradients = []
        sizes = []
        seller_ids = []

        global_params = self.aggregator.get_params()
        for seller_id, seller in self.sellers.items():
            g, size = seller.get_gradient(global_params)
            gradients.append(g)
            sizes.append(size)
            seller_ids.append(seller_id)

        return gradients, sizes, seller_ids

    def select_gradients(self,
                         gradients: List[np.ndarray],
                         sizes: List[int],
                         seller_ids: List[str],
                         num_select: int = None,
                         **kwargs) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Potentially select a subset of gradients for robust or budget-limited reasons.

        By default, if `num_select` is None or equals len(gradients), we use all.
        Otherwise, you can implement a strategy to pick the 'best' subset.

        :return:
            selected_gradients, selected_sizes, selected_seller_ids
        """
        # If no selection is required or num_select is larger than the total:
        if not num_select or num_select >= len(gradients):
            return gradients, sizes, seller_ids

        # Example: pick the first `num_select` sellers (dummy strategy).
        # Replace with advanced strategies (e.g. Krum, sorting, etc.) if needed.
        selected_gradients = gradients[:num_select]
        selected_sizes = sizes[:num_select]
        selected_seller_ids = seller_ids[:num_select]

        return selected_gradients, selected_sizes, selected_seller_ids

    def aggregate_gradients(self,
                            gradients: List[np.ndarray],
                            sizes: List[int]) -> np.ndarray:
        """
        Use the aggregator to compute an aggregated gradient.
        """
        aggregated_grad = self.aggregator.aggregate(gradients, sizes, method=self.selection_method)
        return aggregated_grad

    def update_global_model(self,
                            aggregated_gradient: np.ndarray):
        """
        Apply the aggregated gradient to the aggregator's global model.
        """
        self.aggregator.apply_gradient(aggregated_gradient, learning_rate=self.learning_rate)

    def train_federated_round(self,
                              num_select: int = None,
                              **kwargs) -> Dict:
        """
        Perform one round of federated training:
         1. Collect gradients from all sellers.
         2. Optionally select a subset.
         3. Aggregate the selected gradients.
         4. Update the global model.
         5. Distribute the new global model back to sellers (optional).

        :return:
            A dictionary with info about the round,
            e.g. the final aggregated gradient, seller_ids used, etc.
        """
        # 1. get gradients from sellers
        gradients, sizes, seller_ids = self.get_current_market_gradients()

        # 2. select gradients if needed
        selected_grads, selected_sizes, selected_sellers = self.select_gradients(
            gradients, sizes, seller_ids, num_select=num_select, **kwargs
        )

        # 3. aggregate
        agg_gradient = self.aggregate_gradients(selected_grads, selected_sizes)

        # 4. update global model
        self.update_global_model(agg_gradient)

        # 5. (Optionally) broadcast new global model back to sellers
        self.broadcast_global_model()

        return {
            "aggregated_gradient": agg_gradient,
            "used_sellers": selected_sellers,
            "num_sellers_selected": len(selected_sellers),
            "selection_method": self.selection_method,
        }

    def broadcast_global_model(self):
        """
        Send the updated global model parameters to all sellers
        so they can store/update their local models if needed.
        """
        new_params = self.aggregator.get_params()
        for seller_id, seller in self.sellers.items():
            seller.update_local_model(new_params)

    def get_market_status(self) -> Dict:
        """
        Get status of the marketplace, e.g. number of sellers,
        the current selection/aggregation method, etc.
        """
        return {
            'num_sellers': len(self.sellers),
            'aggregation_method': self.selection_method,
            'learning_rate': self.learning_rate,
            # Potentially more stats or aggregator info
        }
