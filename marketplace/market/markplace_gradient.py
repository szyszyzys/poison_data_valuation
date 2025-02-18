from typing import Dict, Union, List, Tuple, Any

import numpy as np
import torch

from attack.evaluation import evaluate_attack_performance_backdoor_poison
from marketplace.market.data_market import DataMarketplace
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.seller import BaseSeller


# Import your aggregator and seller classes
# from attack.privacy_attack.malicious_seller import MaliciousDataSeller
# from marketplace.market.data_market import DataMarketplace  # base class
# from marketplace.seller.seller import BaseSeller
# from aggregator_file import Aggregator  # example aggregator code above

class DataMarketplaceFederated(DataMarketplace):
    def __init__(self,
                 aggregator: Aggregator,
                 selection_method: str = "fedavg",
                 learning_rate: float = 1.0,
                 broadcast_local=False):
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
        self.broadcast_local = broadcast_local
        # Each seller might be a BaseSeller or an AdversarySeller, etc.
        self.sellers: Dict[str, Union[BaseSeller, Any]] = {}

        # This can store marketplace-level logs or stats, if desired
        self.round_logs: List[Dict[str, Any]] = []

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
            sizes:     List of integers indicating the local data size from each seller
            seller_ids: List of seller IDs in the same order as the gradient list
        """
        gradients = []
        sizes = []
        seller_ids = []

        # Get current global model parameters from aggregator
        current_params = self.aggregator.get_params()  # e.g. dict of state_dict
        # Convert to a form you can send to sellers, or pass directly if they can handle dict
        # e.g. you might pass the aggregator's self.global_model directly

        for seller_id, seller in self.sellers.items():
            # for martfl, local have no access to the global params
            grad_np, local_size = seller.get_gradient(current_params)
            # Expect get_gradient(...) -> (np.ndarray, int) or similar
            gradients.append(grad_np)
            sizes.append(local_size)
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
        Otherwise, pick the first `num_select` sellers. (dummy strategy)
        """
        if not num_select or num_select >= len(gradients):
            return gradients, sizes, seller_ids

        # Example: pick the first `num_select` sellers
        selected_grads = gradients[:num_select]
        selected_sizes = sizes[:num_select]
        selected_seller_ids = seller_ids[:num_select]

        return selected_grads, selected_sizes, selected_seller_ids

    def aggregate_gradients(self, round_number,
                            seller_updates: List[np.ndarray],
                            buyer_updates: List[np.ndarray],
                            sizes: List[int]) -> np.ndarray:
        """
        Use the aggregator to compute an aggregated gradient.
        """
        aggregated_grad = self.aggregator.aggregate(round_number, seller_updates, buyer_updates, sizes,
                                                    method=self.selection_method)
        return aggregated_grad

    def update_global_model(self, aggregated_gradient: np.ndarray):
        """
        Apply the aggregated gradient to the aggregator's global model.
        """
        self.aggregator.apply_gradient(aggregated_gradient, learning_rate=self.learning_rate)

    def broadcast_global_model(self):
        """
        Send the updated global model parameters to all sellers
        so they can store/update their local models if needed.
        """
        new_params = self.aggregator.get_params()
        for seller_id, seller in self.sellers.items():
            seller.update_local_model(new_params)

    def train_federated_round(self,
                              round_number: int,
                              buyer_gradient,
                              num_select: int = None,
                              test_dataloader_buyer_local=None,
                              test_dataloader_global=None,
                              loss_fn=None,
                              clean_loader=None, triggered_loader=None,
                              **kwargs) -> Dict:
        """
        Perform one round of federated training:
         1. Collect gradients from all sellers.
         2. Optionally select a subset.
         3. Aggregate the selected gradients.
         4. Update the global model.
         5. Distribute the new global model back to sellers (optional).
         6. Evaluate final model & log stats (optional).

        :param round_number:  Current round index
        :param num_select:    Number of sellers to select
        :param test_dataloader: Optionally provide a loader for evaluating the global model.
        :param loss_fn:       A torch loss function for evaluation.
        :return:
            A dictionary with info about the round,
            e.g. the final aggregated gradient, seller_ids used, etc.
        """

        # 1. get gradients from sellers
        seller_gradients, sizes, seller_ids = self.get_current_market_gradients()
        # 2. perform aggregation
        aggregated_gradient, selected_ids, outlier_ids, baseline_similarities = self.aggregator.aggregate(round_number,
                                                                                                          seller_gradients,
                                                                                                          buyer_gradient,
                                                                                                          sizes,
                                                                                                          method=self.selection_method)

        # 4. update global model
        self.update_global_model(aggregated_gradient)

        # 5. broadcast updated global model, in martfl the local no broadcast happen
        if self.broadcast_local:
            self.broadcast_global_model()

        # 6. Evaluate the final global model if test_dataloader is provided
        final_perf_local = None
        if test_dataloader_buyer_local is not None and loss_fn is not None:
            # Evaluate aggregator.global_model on test set
            final_perf_local = self.evaluate_global_model(test_dataloader_buyer_local, loss_fn)

        final_perf_global = None
        if test_dataloader_buyer_local is not None and loss_fn is not None:
            # Evaluate aggregator.global_model on test set
            final_perf_global = self.evaluate_global_model(test_dataloader_buyer_local, loss_fn)

        poison_metrics = None
        if clean_loader is not None and triggered_loader is not None:
            poison_metrics = evaluate_attack_performance_backdoor_poison(self.aggregator.global_model, clean_loader,
                                                                         triggered_loader,
                                                                         self.aggregator.device,
                                                                         target_label=None, plot=True)

        # 7. Log round info to aggregator (optional)
        extra_info = {}
        if final_perf_global is not None:
            extra_info["val_loss_global"] = final_perf_global["loss"]
            extra_info["val_acc_global"] = final_perf_global["acc"]

        if final_perf_local is not None:
            extra_info["val_loss_local"] = final_perf_local["loss"]
            extra_info["val_acc_local"] = final_perf_local["acc"]

        if poison_metrics is not None:
            extra_info["poison_metrics"] = poison_metrics
        extra_info["outlier_ids"] = outlier_ids
        self.aggregator.log_round_info(
            round_number=round_number,
            selected_sellers=selected_ids,
            aggregated_gradient=aggregated_gradient,
            extra_info=extra_info
        )

        # 8. Also store a high-level record in the marketplace logs
        round_record = {
            "round_number": round_number,
            "used_sellers": selected_ids,
            "outlier_ids": outlier_ids,
            "num_sellers_selected": len(selected_ids),
            "selection_method": self.selection_method,
            "aggregated_grad_norm": float(np.linalg.norm(aggregated_gradient)) if aggregated_gradient.size > 0 else 0.0,
            "final_perf": final_perf
        }
        self.round_logs.append(round_record)

        # 9. Update each seller about whether they were selected
        for sid in self.sellers:
            # Mark "is_selected" if in selected_sellers
            is_selected = (sid in selected_ids)
            self.sellers[sid].record_federated_round(round_number, is_selected)

        return round_record

    def evaluate_global_model(self, test_dataloader, loss_fn) -> Dict[str, float]:
        """
        Evaluate aggregator.global_model on the given test set, returning a dict with metrics.
        Adjust to your actual code for evaluate_model(...).
        """
        model = self.aggregator.global_model
        if self.aggregator.device:
            model = model.to(self.aggregator.device)

        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                if self.aggregator.device:
                    X = X.to(self.aggregator.device)
                    y = y.to(self.aggregator.device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                total_loss += loss.item() * X.size(0)

                _, preds = torch.max(outputs, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += X.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": acc}

    def get_market_status(self) -> Dict:
        """
        Get status of the marketplace, e.g. number of sellers,
        the current selection/aggregation method, etc.
        """
        return {
            'num_sellers': len(self.sellers),
            'aggregation_method': self.selection_method,
            'learning_rate': self.learning_rate,
        }

    def get_round_logs(self) -> List[Dict[str, Any]]:
        return self.round_logs

    def save_round_logs(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    @property
    def get_all_sellers(self):
        return self.sellers
# class DataMarketplaceFederated(DataMarketplace):
#     def __init__(self,
#                  aggregator: Aggregator,
#                  selection_method: str = "fedavg",
#                  learning_rate: float = 1.0):
#         """
#         A marketplace for federated learning where each seller provides gradient updates.
#
#         :param aggregator: An object that holds the global model and
#                            implements gradient aggregation (e.g., FedAvg).
#         :param selection_method: e.g. "fedavg", "krum", "median", etc.
#         :param learning_rate: Step size for updating global model parameters.
#         """
#         self.aggregator = aggregator
#         self.selection_method = selection_method
#         self.learning_rate = learning_rate
#
#         # Each seller might be a BaseSeller or an AdversarySeller, etc.
#         self.sellers: Dict[str, Union[BaseSeller, MaliciousDataSeller]] = {}
#
#     def register_seller(self, seller_id: str, seller: BaseSeller):
#         """
#         Register a new seller that can provide gradients.
#         """
#         self.sellers[seller_id] = seller
#
#     def update_selection(self, new_method: str):
#         """
#         Update the aggregation/selection method, e.g., from 'fedavg' to 'krum'.
#         """
#         self.selection_method = new_method
#
#     def get_current_market_gradients(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
#         """
#         Collect gradient updates from each seller for the current global model parameters.
#
#         :return:
#             gradients: List of gradient vectors (np.ndarray)
#             sizes: List of integers indicating the local data size from each seller
#             seller_ids: List of seller IDs in the same order as the gradient list
#         """
#         gradients = []
#         sizes = []
#         seller_ids = []
#
#         global_params = self.aggregator.get_params()
#         for seller_id, seller in self.sellers.items():
#             g, size = seller.get_gradient(global_params)
#             gradients.append(g)
#             sizes.append(size)
#             seller_ids.append(seller_id)
#
#         return gradients, sizes, seller_ids
#
#     def select_gradients(self,
#                          gradients: List[np.ndarray],
#                          sizes: List[int],
#                          seller_ids: List[str],
#                          num_select: int = None,
#                          **kwargs) -> Tuple[List[np.ndarray], List[int], List[str]]:
#         """
#         Potentially select a subset of gradients for robust or budget-limited reasons.
#
#         By default, if `num_select` is None or equals len(gradients), we use all.
#         Otherwise, you can implement a strategy to pick the 'best' subset.
#
#         :return:
#             selected_gradients, selected_sizes, selected_seller_ids
#         """
#         # If no selection is required or num_select is larger than the total:
#         if not num_select or num_select >= len(gradients):
#             return gradients, sizes, seller_ids
#
#         # Example: pick the first `num_select` sellers (dummy strategy).
#         # Replace with advanced strategies (e.g. Krum, sorting, etc.) if needed.
#         selected_gradients = gradients[:num_select]
#         selected_sizes = sizes[:num_select]
#         selected_seller_ids = seller_ids[:num_select]
#
#         return selected_gradients, selected_sizes, selected_seller_ids
#
#     def aggregate_gradients(self,
#                             gradients: List[np.ndarray],
#                             sizes: List[int]) -> np.ndarray:
#         """
#         Use the aggregator to compute an aggregated gradient.
#         """
#         aggregated_grad = self.aggregator.aggregate(gradients, sizes, method=self.selection_method)
#         return aggregated_grad
#
#     def update_global_model(self,
#                             aggregated_gradient: np.ndarray):
#         """
#         Apply the aggregated gradient to the aggregator's global model.
#         """
#         self.aggregator.apply_gradient(aggregated_gradient, learning_rate=self.learning_rate)
#
#     def train_federated_round(self,
#                               num_select: int = None,
#                               **kwargs) -> Dict:
#         """
#         Perform one round of federated training:
#          1. Collect gradients from all sellers.
#          2. Optionally select a subset.
#          3. Aggregate the selected gradients.
#          4. Update the global model.
#          5. Distribute the new global model back to sellers (optional).
#
#         :return:
#             A dictionary with info about the round,
#             e.g. the final aggregated gradient, seller_ids used, etc.
#         """
#         # 1. get gradients from sellers
#         gradients, sizes, seller_ids = self.get_current_market_gradients()
#
#         # 2. select gradients if needed
#         selected_grads, selected_sizes, selected_sellers = self.select_gradients(
#             gradients, sizes, seller_ids, num_select=num_select, **kwargs
#         )
#
#         # 3. aggregate
#         agg_gradient = self.aggregate_gradients(selected_grads, selected_sizes)
#
#         # 4. update global model
#         self.update_global_model(agg_gradient)
#
#         # 5. (Optionally) broadcast new global model back to sellers
#         self.broadcast_global_model()
#
#         return {
#             "aggregated_gradient": agg_gradient,
#             "used_sellers": selected_sellers,
#             "num_sellers_selected": len(selected_sellers),
#             "selection_method": self.selection_method,
#         }
#
#     def broadcast_global_model(self):
#         """
#         Send the updated global model parameters to all sellers
#         so they can store/update their local models if needed.
#         """
#         new_params = self.aggregator.get_params()
#         for seller_id, seller in self.sellers.items():
#             seller.update_local_model(new_params)
#
#     def get_market_status(self) -> Dict:
#         """
#         Get status of the marketplace, e.g. number of sellers,
#         the current selection/aggregation method, etc.
#         """
#         return {
#             'num_sellers': len(self.sellers),
#             'aggregation_method': self.selection_method,
#             'learning_rate': self.learning_rate,
#             # Potentially more stats or aggregator info
#         }
