import logging
import time

import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, Union, List, Tuple, Any

from attack.evaluation.evaluation_backdoor import evaluate_attack_performance_backdoor_poison
from marketplace.market.data_market import DataMarketplace
from marketplace.market_mechanism.martfl import Aggregator, flatten
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
        self.benign_selection_rate_list = []

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

    # def get_current_market_gradients(self, base_model):
    #     """
    #     Collect gradient updates from each seller for the current global model parameters.
    #
    #     :return:
    #         gradients: List of gradient vectors (np.ndarray)
    #         seller_ids: List of seller IDs in the same order as the gradient list
    #     """
    #     gradients = OrderedDict()
    #     seller_ids = []
    #
    #     # Get current global model parameters from aggregator
    #     # current_params = self.aggregator.get_params()  # e.g. dict of state_dict
    #     # Convert to a form you can send to sellers, or pass directly if they can handle dict
    #     # e.g. you might pass the aggregator's self.global_model directly
    #     print(f"current sellers: {self.sellers.keys()}")
    #     for seller_id, seller in self.sellers.items():
    #         # for martfl, local have no access to the global params
    #         grad_np = seller.get_gradient_for_upload(base_model)
    #         norm = torch.norm(flatten(grad_np))
    #         print(f"The {seller_id} gradient norm is: {norm}")
    #         gradients[seller_id] = grad_np
    #         seller_ids.append(seller_id)
    #
    #     return gradients, seller_ids

    def get_current_market_gradients(self, base_model):
        """
        Collect gradient updates AND performance statistics from each seller
        for the current global model parameters.

        Requires Seller classes' get_gradient_for_upload methods to return:
            (gradient, stats_dict)
        where stats_dict = {'train_loss': float, 'compute_time_ms': float, 'upload_bytes': int}

        Returns:
            gradients_list: List of gradient vectors/tensors/structures.
            seller_ids: List of seller IDs corresponding to the gradients.
            seller_stats_list: List of dictionaries, each containing stats
                                for the corresponding seller.
        """
        gradients_list = OrderedDict()
        seller_ids = []
        seller_stats_list = []  # New list to store stats

        logging.info(f"Collecting gradients and stats from sellers: {list(self.sellers.keys())}")
        for seller_id, seller in self.sellers.items():
            try:
                # --- Call the MODIFIED seller method ---
                # It now returns two values: gradient and stats dict
                # This assumes ALL seller types implement this modified return signature
                grad_data, stats = seller.get_gradient_for_upload(base_model)
                # -----------------------------------------

                # Basic validation
                if grad_data is None:
                    logging.warning(f"Seller {seller_id} returned None gradient. Skipping.")
                    continue
                if not isinstance(stats, dict):
                    logging.warning(f"Seller {seller_id} did not return a stats dictionary. Using default Nones.")
                    stats = {'train_loss': None, 'compute_time_ms': None, 'upload_bytes': None}
                else:
                    # Ensure expected keys exist, adding None if missing
                    stats.setdefault('train_loss', None)
                    stats.setdefault('compute_time_ms', None)
                    stats.setdefault('upload_bytes', None)

                # Log gradient norm (adapt norm calculation based on grad_data type)
                try:
                    # Try flattening assuming list of tensors/arrays or single tensor/array
                    flat_grad = flatten(grad_data)
                    norm = torch.norm(flat_grad).item() if flat_grad.numel() > 0 else 0.0
                except Exception:  # Fallback for other types or errors
                    try:
                        norm = np.linalg.norm(grad_data)  # If it's a simple numpy array
                    except Exception:
                        norm = float('nan')  # Cannot compute norm
                logging.info(f"  Seller {seller_id}: Grad norm = {norm:.4f}, Stats = {stats}")

                # Append results to lists, maintaining order
                gradients_list[seller_id] = grad_data
                seller_ids.append(seller_id)
                seller_stats_list.append(stats)

            except Exception as e:
                logging.error(f"Error getting gradient/stats from seller {seller_id}: {e}", exc_info=True)
                # Decide how to handle errors: skip seller, append Nones, raise?
                # Skipping seller for robustness:
                continue

        logging.info(f"Collected gradients and stats from {len(gradients_list)} out of {len(self.sellers)} sellers.")
        # Return three lists, ensuring alignment
        return gradients_list, seller_ids, seller_stats_list
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

    # def compute_selection_rates(self, round_selected, malicious_ids, benign_ids):
    #     """
    #     Compute the selection rates for malicious and benign sellers over multiple rounds.
    #
    #     Parameters:
    #       all_rounds_selected_ids : list of lists or sets
    #           Each element is a collection of selected seller IDs for one round.
    #       malicious_ids : set
    #           Set of malicious seller IDs.
    #       benign_ids : set
    #           Set of benign seller IDs.
    #
    #     Returns:
    #       A dictionary with:
    #          - 'malicious_rates': list of malicious selection rates per round.
    #          - 'benign_rates': list of benign selection rates per round.
    #          - 'avg_malicious_rate': average malicious selection rate over rounds.
    #          - 'avg_benign_rate': average benign selection rate over rounds.
    #     """
    #     round_selected_set = set(round_selected)
    #
    #     # Calculate selection rate for malicious sellers in this round.
    #     malicious_rate = 0
    #     if len(malicious_ids):
    #         malicious_rate = len(malicious_ids.intersection(round_selected_set)) / len(malicious_ids)
    #     self.malicious_selection_rate_list.append(malicious_rate)
    #
    #     # Calculate selection rate for benign sellers in this round.
    #     benign_rate = len(benign_ids.intersection(round_selected_set)) / len(benign_ids)
    #     self.benign_selection_rate_list.append(benign_rate)
    #
    #     avg_malicious_rate = sum(self.malicious_selection_rate_list) / len(
    #         self.malicious_selection_rate_list) if self.malicious_selection_rate_list else 0
    #     avg_benign_rate = sum(self.benign_selection_rate_list) / len(
    #         self.benign_selection_rate_list) if self.benign_selection_rate_list else 0
    #
    #     return {
    #         'malicious_rate': malicious_rate,
    #         'benign_rate': benign_rate,
    #         'avg_malicious_rate': avg_malicious_rate,
    #         'avg_benign_rate': avg_benign_rate,
    #     }

    def train_federated_round(self,
                              round_number: int,
                              buyer, # Assumes buyer object has get_gradient_for_upload method
                              n_adv=0, # Ground truth number of adversaries
                              test_dataloader_buyer_local=None,
                              test_dataloader_global=None,
                              loss_fn=None,
                              backdoor_target_label=0,
                              backdoor_generator=None, # Assumes it's used by evaluate_attack_performance
                              clip = False,
                              remove_baseline = False
                              ):
        """
        Perform one round of federated training with enhanced logging.
        """
        round_start_time = time.time()
        logging.info(f"--- Round {round_number} Started ---")

        # --- Timing and Metrics Initialization ---
        client_train_losses = []
        client_compute_times_ms = []
        client_upload_bytes = []
        server_aggregation_time_ms = None

        # --- Get Buyer Gradient (Baseline) ---
        # Consider timing this if it's significant
        baseline_gradient, _ = buyer.get_gradient_for_upload(self.aggregator.global_model)

        # --- 1. Get Gradients & Stats from Sellers ---
        seller_gradients_dict, seller_ids, seller_stats = self.get_current_market_gradients(self.aggregator.global_model)
        # Process returned stats
        for stats in seller_stats:
            if stats: # Check if stats were actually returned
                client_train_losses.append(stats.get('train_loss'))
                client_compute_times_ms.append(stats.get('compute_time_ms'))
                client_upload_bytes.append(stats.get('upload_bytes'))

        # --- 2. Perform Aggregation ---
        agg_start_time = time.time()
        aggregated_gradient, selected_indices, outlier_indices = self.aggregator.aggregate(
            round_number,
            seller_gradients_dict, # Assuming aggregate works with list of gradients
            baseline_gradient,
            clip=clip,
            remove_baseline=remove_baseline,
            server_data=test_dataloader_buyer_local
        )
        agg_end_time = time.time()
        server_aggregation_time_ms = (agg_end_time - agg_start_time) * 1000

        # Map selected/outlier indices back to actual seller IDs
        # Assumes seller_ids list corresponds to seller_gradients list order
        selected_ids = [seller_ids[i] for i in selected_indices]
        outlier_ids = [seller_ids[i] for i in outlier_indices]

        print(f"Selected Sellers ({len(selected_ids)}): {selected_ids}")
        print(f"Outlier Sellers ({len(outlier_ids)}): {outlier_ids}")
        print(f"Aggregated gradient norm: {np.linalg.norm(flatten(aggregated_gradient))}") # Assuming flatten exists

        # --- 3. Update Global Model ---
        # Consider timing this if significant
        update_start_time = time.time()
        self.update_global_model(aggregated_gradient)
        update_end_time = time.time()
        server_update_time_ms = (update_end_time - update_start_time) * 1000
        # Combine server times
        server_total_time_ms = server_aggregation_time_ms + server_update_time_ms

        # --- 4. Broadcast Updated Global Model (Optional) ---
        if self.broadcast_local:
            # Consider timing broadcast if needed
            self.broadcast_global_model()

        # --- 5. Evaluation ---
        perf_global = None
        perf_local = None
        global_asr = None
        # Add local ASR evaluation if possible/needed
        # local_asr = None

        if test_dataloader_global is not None and loss_fn is not None:
            eval_global_res = self.evaluate_global_model(test_dataloader_global, loss_fn)
            perf_global = {
                "accuracy": eval_global_res.get("acc"),
                "loss": eval_global_res.get("loss"),
                # Add other metrics from eval_global_res if available
            }
            # Evaluate backdoor attack performance
            if backdoor_generator is not None:
                 # !! ASSUMPTION !! evaluate_attack_performance returns dict with at least ASR
                 poison_metrics = evaluate_attack_performance_backdoor_poison(
                     self.aggregator.global_model,
                     test_dataloader_global,
                     self.aggregator.device,
                     backdoor_generator,
                     target_label=backdoor_target_label, plot=False,
                     # Save path could be made round-specific if needed
                 )
                 global_asr = poison_metrics.get("attack_success_rate")
                 perf_global["attack_success_rate"] = global_asr # Add ASR to perf dict
                 # Add other poison metrics if needed: perf_global['other_poison_metric'] = poison_metrics.get(...)

        if test_dataloader_buyer_local is not None and loss_fn is not None:
            eval_local_res = self.evaluate_global_model(test_dataloader_buyer_local, loss_fn)
            perf_local = {
                "accuracy": eval_local_res.get("acc"),
                "loss": eval_local_res.get("loss"),
                # Add other metrics if available
                # "attack_success_rate": local_asr, # Log local ASR if evaluated
            }

        # --- 6. Calculate Overhead Metrics (Averages/Max) ---
        # Filter out None values before calculating stats
        valid_losses = [l for l in client_train_losses if l is not None]
        valid_times = [t for t in client_compute_times_ms if t is not None]
        valid_bytes = [b for b in client_upload_bytes if b is not None]

        avg_client_train_loss = np.mean(valid_losses) if valid_losses else None
        client_time_avg_ms = np.mean(valid_times) if valid_times else None
        client_time_max_ms = np.max(valid_times) if valid_times else None
        comm_up_avg_bytes = np.mean(valid_bytes) if valid_bytes else None
        comm_up_max_bytes = np.max(valid_bytes) if valid_bytes else None
        # Estimate downlink bytes (model size) - requires model saving/sizing
        # comm_down_bytes = estimate_model_size_bytes(self.aggregator.global_model) # Placeholder

        # --- 7. Compute Selection Rates ---
        # Needs ground truth sets of adversary/benign IDs corresponding to seller_ids
        # Construct these sets based on n_adv and the structure of seller_ids
        # Example: assumes first n_adv IDs in the original full list were adversaries
        # This mapping needs to be robust if seller_ids order changes.
        # It's better if the aggregator or marketplace tracks ground truth status.
        all_seller_ids = list(self.sellers.keys()) # Get IDs in a consistent order if possible
        # Assume IDs like 'adv_0', 'adv_1', ..., 'seller_k', ...
        adv_ids_set = {sid for sid in all_seller_ids if sid.startswith('adv_')}
        benign_ids_set = {sid for sid in all_seller_ids if not sid.startswith('adv_')}
        # Make sure n_adv matches len(adv_ids_set)
        if len(adv_ids_set) != n_adv:
             logging.warning(f"Mismatch between n_adv ({n_adv}) and actual adv IDs found ({len(adv_ids_set)}). Check ID format.")

        selection_rate_info = self.compute_selection_rates(selected_ids, outlier_ids, adv_ids_set, benign_ids_set) # Modified signature needed

        # --- 8. Get Defense/Algorithm Specific Metrics ---
        defense_metrics = {}
        if hasattr(self.aggregator, 'get_specific_metrics'):
             # !! ASSUMPTION !! Aggregator provides specific metrics
             defense_metrics = self.aggregator.get_specific_metrics(round_number)
        elif self.aggregator.aggregation_method == 'MartFL': # Explicit check if no generic method
              defense_metrics['baseline_id'] = self.aggregator.baseline_id if hasattr(self.aggregator, 'baseline_id') else None
        # Add elif blocks for other specific aggregators (Skymask, FLTrust) if needed

        # --- 9. Construct Final Round Record ---
        round_end_time = time.time()
        final_round_record = {
            # Core Info
            "round_number": round_number,
            "timestamp": round_end_time,
            "round_duration_sec": round_end_time - round_start_time,
            "aggregation_method": self.aggregator.aggregation_method,

            # Participation & Selection
            "selected_sellers": selected_ids,
            "num_sellers_selected": len(selected_ids),
            "outlier_sellers": outlier_ids,
            "selection_rate_info": selection_rate_info, # Contains TP/FP rates etc.

            # Performance
            "perf_global": perf_global, # Dict: {acc, loss, attack_success_rate, ...}
            "perf_local": perf_local,   # Dict: {acc, loss, attack_success_rate*, ...} (*if added)
            "avg_client_train_loss": avg_client_train_loss, # Needs modification elsewhere

            # Overhead
            "comm_up_avg_bytes": comm_up_avg_bytes, # Needs modification elsewhere
            "comm_up_max_bytes": comm_up_max_bytes, # Needs modification elsewhere
            "comm_down_bytes": None, # Placeholder - estimate model size
            "client_time_avg_ms": client_time_avg_ms, # Needs modification elsewhere
            "client_time_max_ms": client_time_max_ms, # Needs modification elsewhere
            "server_time_ms": server_total_time_ms,

            # Defense/Algorithm Specifics (optional)
            "defense_metrics": defense_metrics if defense_metrics else None, # Only include if non-empty
        }

        # --- 10. Update Sellers & Log ---
        # Update sellers *after* constructing the log, in case their state affects metrics
        for sid, seller in self.sellers.items():
            is_selected = (sid in selected_ids)
            seller.round_end_process(round_number, is_selected) # Assumes seller has this method

        self.round_logs.append(final_round_record)
        logging.info(f"--- Round {round_number} Ended (Duration: {final_round_record['round_duration_sec']:.2f}s) ---")
        logging.info(f"  Global Acc: {perf_global.get('accuracy', 'N/A'):.4f}, Global ASR: {perf_global.get('attack_success_rate', 'N/A')}")
        logging.info(f"  Local Acc: {perf_local.get('accuracy', 'N/A'):.4f}")
        logging.info(f"  Server Time: {server_total_time_ms:.2f}ms")

        # Return the detailed record and the gradient (as before)
        return final_round_record, aggregated_gradient

    # --- Modified compute_selection_rates signature ---
    def compute_selection_rates(self, selected_ids_list, outlier_ids_list, adv_ids_set, benign_ids_set):
        """Computes selection/rejection rates based on ground truth."""
        selected_set = set(selected_ids_list)
        outlier_set = set(outlier_ids_list) # Assuming aggregate returns disjoint sets

        # Ensure consistency
        if not selected_set.isdisjoint(outlier_set):
            logging.warning("Selected IDs and Outlier IDs overlap! Check aggregator logic.")

        # Malicious Sellers (Adversaries)
        malicious_selected = len(selected_set.intersection(adv_ids_set)) # Adv selected (Missed by defense if rejection is goal) -> FN
        malicious_rejected = len(outlier_set.intersection(adv_ids_set))  # Adv rejected (Caught by defense) -> TP
        total_malicious = len(adv_ids_set)

        # Benign Sellers
        benign_selected = len(selected_set.intersection(benign_ids_set)) # Benign selected (Correctly kept) -> TN
        benign_rejected = len(outlier_set.intersection(benign_ids_set)) # Benign rejected (Incorrectly flagged) -> FP
        total_benign = len(benign_ids_set)

        # Calculate rates (handle division by zero)
        tpr = malicious_rejected / total_malicious if total_malicious > 0 else 0 # True Positive Rate (detection rate)
        fpr = benign_rejected / total_benign if total_benign > 0 else 0       # False Positive Rate
        fnr = malicious_selected / total_malicious if total_malicious > 0 else 0 # False Negative Rate
        tnr = benign_selected / total_benign if total_benign > 0 else 0       # True Negative Rate

        return {
            "malicious_selected (FN)": malicious_selected,
            "malicious_rejected (TP)": malicious_rejected,
            "benign_selected (TN)": benign_selected,
            "benign_rejected (FP)": benign_rejected,
            "total_malicious_in_round": total_malicious, # Might change if participation varies
            "total_benign_in_round": total_benign,
            "detection_rate (TPR)": tpr,
            "false_positive_rate (FPR)": fpr,
        }

    # def train_federated_round(self,
    #                           round_number: int,
    #                           buyer,
    #                           n_adv=0,
    #                           test_dataloader_buyer_local=None,
    #                           test_dataloader_global=None,
    #                           loss_fn=None,
    #                           backdoor_target_label=0,
    #                           backdoor_generator=None,
    #                           clip = False,
    #                           remove_baseline = False
    #                           ):
    #     """
    #     Perform one round of federated training:
    #      1. Collect gradients from all sellers.
    #      2. Optionally select a subset.
    #      3. Aggregate the selected gradients.
    #      4. Update the global model.
    #      5. Distribute the new global model back to sellers (optional).
    #      6. Evaluate final model & log stats (optional).
    #     """
    #     baseline_gradient = buyer.get_gradient_for_upload(self.aggregator.global_model)
    #
    #     # 1. get gradients from sellers
    #     seller_gradients, seller_ids = self.get_current_market_gradients(self.aggregator.global_model)
    #     # 2. perform aggregation
    #     aggregated_gradient, selected_ids, outlier_ids = self.aggregator.aggregate(round_number,
    #                                                                                seller_gradients,
    #                                                                                baseline_gradient, clip=clip, remove_baseline =remove_baseline)
    #     print(f"round {round_number} aggregated gradient norm: {np.linalg.norm(flatten(aggregated_gradient))}")
    #     # 4. update global model
    #     self.update_global_model(aggregated_gradient)
    #
    #     # 5. broadcast updated global model, in martfl the local no broadcast happen
    #     if self.broadcast_local:
    #         self.broadcast_global_model()
    #
    #     # 6. Evaluate the final global model if test_dataloader is provided
    #     final_perf_local = None
    #     if test_dataloader_buyer_local is not None and loss_fn is not None:
    #         # Evaluate aggregator.global_model on test set
    #         final_perf_local = self.evaluate_global_model(test_dataloader_buyer_local, loss_fn)
    #     final_perf_global = None
    #     poison_metrics = None
    #     if test_dataloader_global is not None and loss_fn is not None:
    #         # Evaluate aggregator.global_model on test set
    #         final_perf_global = self.evaluate_global_model(test_dataloader_global, loss_fn)
    #
    #         poison_metrics = evaluate_attack_performance_backdoor_poison(self.aggregator.global_model,
    #                                                                      test_dataloader_global,
    #                                                                      self.aggregator.device,
    #                                                                      backdoor_generator,
    #                                                                      target_label=backdoor_target_label, plot=False,
    #                                                                      save_path=f"{self.save_path}/attack_performance.png")
    #     # 7. Log round info to aggregator (optional)
    #     extra_info = {}
    #     if final_perf_global is not None:
    #         extra_info["val_loss_global"] = final_perf_global["loss"]
    #         extra_info["val_acc_global"] = final_perf_global["acc"]
    #
    #     if final_perf_local is not None:
    #         extra_info["val_loss_local"] = final_perf_local["loss"]
    #         extra_info["val_acc_local"] = final_perf_local["acc"]
    #
    #     if poison_metrics is not None:
    #         extra_info["poison_metrics"] = poison_metrics
    #     extra_info["outlier_ids"] = outlier_ids
    #     # 8. Also store a high-level record in the marketplace logs
    #
    #     # 9. Update each seller about whether they were selected
    #     # update_local_model_from_global(buyer, dataset_name, aggregated_gradient)
    #     for idx, (sid, seller) in enumerate(self.sellers.items()):
    #         # Mark "is_selected" if in selected_sellers
    #         is_selected = (idx in selected_ids)
    #         # update_local_model_from_global(seller, dataset_name, aggregated_gradient)
    #         # reset the local gradient
    #         seller.round_end_process(round_number, is_selected)
    #     print(f"=============round {round_number} end summary=======================")
    #     print(
    #         f"round {round_number}, global accuracy: {extra_info['val_acc_global']}, local accuracy: {extra_info['val_acc_local']}, selected: {selected_ids}")
    #     print(f"Test set eval result: {final_perf_global}")
    #     print(f"Buyer local eval result: {final_perf_local}")
    #     print(f"Backdoor Metrics on Global model: {poison_metrics}")
    #
    #     selection_rate_info = self.compute_selection_rates(selected_ids, set(range(n_adv)),
    #                                                        set(range(n_adv, len(self.sellers))))
    #     print(selection_rate_info)
    #     round_record = {
    #         "round_number": round_number,
    #         "used_sellers": selected_ids,
    #         "outlier_ids": outlier_ids,
    #         "num_sellers_selected": len(selected_ids),
    #         "selection_method": self.selection_method,
    #         "final_perf_local": final_perf_local,
    #         "final_perf_global": final_perf_global,
    #         "extra_info": extra_info,
    #         "selection_rate_info": selection_rate_info if selection_rate_info else None,
    #         "martfl_baseline_id": self.aggregator.baseline_id if self.aggregator.baseline_id else None
    #     }
    #
    #     self.round_logs.append(round_record)
    #
    #     return round_record, aggregated_gradient

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
