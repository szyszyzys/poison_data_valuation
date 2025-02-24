import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, Union, List, Tuple, Any

from attack.evaluation.evaluation_backdoor import evaluate_attack_performance_backdoor_poison
from marketplace.market.data_market import DataMarketplace
from marketplace.market_mechanism.martfl import Aggregator, flatten
from marketplace.seller.gradient_seller import update_local_model_from_global
from marketplace.seller.seller import BaseSeller
from model.utils import apply_gradient_update


class DataMarketplaceFederated(DataMarketplace):
    def __init__(self,
                 aggregator: Aggregator,
                 selection_method: str = "fedavg",
                 learning_rate: float = 1.0,
                 broadcast_local=False, save_path=''):
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
        self.sellers: OrderedDict[str, Union[BaseSeller, Any]] = OrderedDict()
        self.save_path = save_path
        # This can store marketplace-level logs or stats, if desired
        self.round_logs: List[Dict[str, Any]] = []
        self.malicious_selection_rate_list = []

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

    def get_current_market_gradients(self):
        """
        Collect gradient updates from each seller for the current global model parameters.

        :return:
            gradients: List of gradient vectors (np.ndarray)
            seller_ids: List of seller IDs in the same order as the gradient list
        """
        gradients = OrderedDict()
        seller_ids = []

        # Get current global model parameters from aggregator
        # current_params = self.aggregator.get_params()  # e.g. dict of state_dict
        # Convert to a form you can send to sellers, or pass directly if they can handle dict
        # e.g. you might pass the aggregator's self.global_model directly
        print(f"current sellers: {self.sellers.keys()}")
        for seller_id, seller in self.sellers.items():
            # for martfl, local have no access to the global params
            grad_np = seller.get_gradient_for_upload()
            norm = np.linalg.norm(flatten(grad_np))
            print(f"The {seller_id} gradient norm is: {norm}")
            gradients[seller_id] = grad_np
            seller_ids.append(seller_id)

        return gradients, seller_ids

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

    def update_global_model(self, aggregated_gradient):
        """
        Apply the aggregated gradient to the aggregator's global model.
        """
        # self.aggregator.apply_gradient(aggregated_gradient, learning_rate=self.learning_rate)
        self.aggregator.global_model = apply_gradient_update(self.aggregator.global_model, aggregated_gradient)

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
                              buyer,
                              n_adv=0,
                              num_select: int = None,
                              test_dataloader_buyer_local=None,
                              test_dataloader_global=None,
                              loss_fn=None,
                              clean_loader=None, triggered_loader=None, triggered_clean_label_loader=None, device="cpu",
                              backdoor_target_label=0,
                              dataset_name="",
                              **kwargs):
        """
        Perform one round of federated training:
         1. Collect gradients from all sellers.
         2. Optionally select a subset.
         3. Aggregate the selected gradients.
         4. Update the global model.
         5. Distribute the new global model back to sellers (optional).
         6. Evaluate final model & log stats (optional).
        """
        baseline_gradient = buyer.get_gradient_for_upload()

        # 1. get gradients from sellers
        seller_gradients, seller_ids = self.get_current_market_gradients()
        # 2. perform aggregation
        aggregated_gradient, selected_ids, outlier_ids = self.aggregator.aggregate(round_number,
                                                                                   seller_gradients,
                                                                                   baseline_gradient)
        print(f"round {round_number} aggregated gradient norm: {np.linalg.norm(flatten(aggregated_gradient))}")
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
        if test_dataloader_global is not None and loss_fn is not None:
            # Evaluate aggregator.global_model on test set
            final_perf_global = self.evaluate_global_model(test_dataloader_global, loss_fn)

        poison_metrics = None
        if clean_loader is not None and triggered_loader is not None:
            poison_metrics = evaluate_attack_performance_backdoor_poison(self.aggregator.global_model, clean_loader,
                                                                         triggered_clean_label_loader,
                                                                         self.aggregator.device,
                                                                         target_label=backdoor_target_label, plot=False,
                                                                         save_path=f"{self.save_path}/attack_performance.png")
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
        # 8. Also store a high-level record in the marketplace logs
        round_record = {
            "round_number": round_number,
            "used_sellers": selected_ids,
            "outlier_ids": outlier_ids,
            "num_sellers_selected": len(selected_ids),
            "selection_method": self.selection_method,
            # "aggregated_grad_norm": float(np.linalg.norm(aggregated_gradient)) if len(aggregated_gradient) > 0 else 0.0,
            "final_perf_local": final_perf_local,
            "final_perf_global": final_perf_global,
            "extra_info": extra_info
        }

        self.round_logs.append(round_record)

        # 9. Update each seller about whether they were selected
        update_local_model_from_global(buyer, dataset_name, aggregated_gradient)
        for idx, (sid, seller) in enumerate(self.sellers.items()):
            # Mark "is_selected" if in selected_sellers
            is_selected = (idx in selected_ids)
            update_local_model_from_global(seller, dataset_name, aggregated_gradient)
            # reset the local gradient
            seller.round_end_process(round_number, is_selected)
        print(
            f"round {round_number}, global accuracy: {extra_info['val_acc_global']}, local accuracy: {extra_info['val_acc_local']}, selected: {selected_ids}")
        print(f"Test set eval result: {final_perf_global}")
        print(f"Buyer local eval result: {final_perf_local}")
        if n_adv> 0:
            malicious_ids_set = set(range(n_adv))  # n malicious sellers labeled 0 to n-1
            selected_ids_set = set(selected_ids)  # the set of selected IDs from a round

            malicious_selection_rate = len(malicious_ids_set.intersection(selected_ids_set)) / len(malicious_ids_set)
            self.malicious_selection_rate_list.append(malicious_selection_rate)

            average_selection_rate = sum(self.malicious_selection_rate_list) / len(self.malicious_selection_rate_list)
            print(f"Current malicious selection result: {malicious_selection_rate}")
            print(f"Average malicious selection result: {average_selection_rate}")

        return round_record, aggregated_gradient

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
