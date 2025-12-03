# Add these imports at the top of your valuation.py file
import copy
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from src.mechanism.gradient.aggregator import Aggregator


class KernelSHAPEvaluator:
    """
    Approximates Shapley values using the KernelSHAP method
    by fitting a weighted linear model.

    This is an ONLINE, periodic, and EXPENSIVE method.
    """

    def __init__(self,
                 aggregator_object: Aggregator,
                 buyer_root_loader: DataLoader,
                 device: str,
                 num_samples: int):
        self.aggregator = aggregator_object
        self.buyer_loader = buyer_root_loader
        self.device = device
        self.num_samples = num_samples
        logging.info("KernelSHAPEvaluator initialized.")

    def _get_performance(self, model: torch.nn.Module) -> float:
        """Helper: Evaluates a model's accuracy on the buyer's data."""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the whole batch
            for batch in self.buyer_loader:

                try:
                    if len(batch) == 3:  # Text data
                        labels, data, _ = batch
                    else:  # Image/Tabular
                        data, labels = batch
                except Exception as e:
                    # Log a warning if you have logging imported, otherwise print
                    print(f"[WARN] KernelSHAP skipping eval batch due to unpack error: {e}")
                    continue

                data, labels = data.to(self.device), labels.to(self.device)

                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return (100 * correct / total) if total > 0 else 0.0

    def _get_performance_for_coalition(
            self,
            original_model: torch.nn.Module,
            all_gradients: Dict[str, List[torch.Tensor]],
            coalition: List[str],  # List of seller IDs in the coalition
            round_number: int,
            buyer_gradient: Optional[List[torch.Tensor]]  # ✅ --- 5. ACCEPT IT HERE ---
    ) -> float:
        """
        Simulates an aggregation and update for *only* the sellers
        in the coalition, then returns the resulting model performance.
        """
        if not coalition:
            # Value of the empty set is the performance *before* any update
            return self._get_performance(original_model)

        # 1. Filter gradients to only this coalition
        coalition_gradients = {
            sid: grad for sid, grad in all_gradients.items()
            if sid in coalition
        }

        # 2. Create a deep copy of the model
        temp_model = copy.deepcopy(original_model).to(self.device)

        # 3. Simulate the aggregation
        original_model_state = self.aggregator.strategy.global_model
        self.aggregator.strategy.global_model = temp_model  # Temporarily swap

        agg_grad, _, _, _ = self.aggregator.aggregate(
            global_epoch=round_number,
            seller_updates=coalition_gradients,
            root_gradient=buyer_gradient,
            buyer_data_loader=self.buyer_loader
        )

        # 4. Simulate applying the gradient
        if agg_grad:
            try:
                self.aggregator.apply_gradient(agg_grad)
            except Exception as e:
                logging.error(f"KernelSHAP: Failed to apply temp gradient: {e}")

        # 5. Evaluate the temporary model's performance
        performance = self._get_performance(temp_model)

        # 6. Restore the original model and cleanup
        self.aggregator.strategy.global_model = original_model_state
        del temp_model

        return performance

    def _get_kernelshap_weights(self, z_prime, num_sellers):
        """Calculates the Shapley kernel weight for a coalition size."""
        if z_prime == 0 or z_prime == num_sellers:
            return 1e9  # Effectively infinite weight

        from scipy.special import comb
        return (num_sellers - 1) / (comb(num_sellers, z_prime) * z_prime * (num_sellers - z_prime))

    def evaluate_round(
            self,
            round_number: int,
            current_global_model: torch.nn.Module,
            seller_gradients: Dict[str, List[torch.Tensor]],
            buyer_gradient: Optional[List[torch.Tensor]]  # ✅ --- 1. ACCEPT IT HERE ---
    ) -> Dict[str, Dict[str, Any]]:

        logging.info("Starting KernelSHAP (Linear Model) evaluation...")
        logging.info("Starting KernelSHAP (Linear Model) evaluation...")
        seller_ids = list(seller_gradients.keys())
        num_sellers = len(seller_ids)
        valuations = {sid: {} for sid in seller_ids}

        # 1. Create the dataset for the linear model (X, y, weights)
        X_coalitions = []  # Binary vectors (e.g., [1, 0, 1])
        y_performance = []  # Performance for that coalition
        sample_weights = []  # Shapley kernel weights

        # 2. Add the two required anchor coalitions
        # Coalition 1: Empty set
        X_coalitions.append(np.zeros(num_sellers))
        y_performance.append(self._get_performance_for_coalition(
            current_global_model, seller_gradients, [], round_number,
            buyer_gradient  # ✅ --- 2. PASS IT HERE ---
        ))
        sample_weights.append(self._get_kernelshap_weights(0, num_sellers))

        # Coalition 2: Grand coalition (all sellers)
        X_coalitions.append(np.ones(num_sellers))
        y_performance.append(self._get_performance_for_coalition(
            current_global_model, seller_gradients, seller_ids, round_number,
            buyer_gradient  # ✅ --- 3. PASS IT HERE ---
        ))
        sample_weights.append(self._get_kernelshap_weights(num_sellers, num_sellers))

        # 3. Sample 'M' random coalitions
        num_to_sample = max(0, self.num_samples - 2)  # Already added 2
        for _ in range(num_to_sample):
            # Create a random binary coalition vector
            z_prime_binary = np.random.randint(0, 2, num_sellers)
            z_prime_size = np.sum(z_prime_binary)

            # Convert binary vector to list of seller IDs
            coalition_sids = [
                sid for i, sid in enumerate(seller_ids)
                if z_prime_binary[i] == 1
            ]

            # Get performance and weight
            perf = self._get_performance_for_coalition(
                current_global_model, seller_gradients, coalition_sids, round_number,
                buyer_gradient  # ✅ --- 4. PASS IT HERE ---
            )
            weight = self._get_kernelshap_weights(z_prime_size, num_sellers)

            X_coalitions.append(z_prime_binary)
            y_performance.append(perf)
            sample_weights.append(weight)

        # 4. Fit the weighted linear model
        try:
            X = np.array(X_coalitions)
            y = np.array(y_performance)
            weights = np.array(sample_weights)

            model = LinearRegression()
            model.fit(X, y, sample_weight=weights)

            # The coefficients of the linear model ARE the Shapley values
            shapley_values = model.coef_

            for i, sid in enumerate(seller_ids):
                valuations[sid]['kernelshap_score'] = shapley_values[i]

            logging.info(f"KernelSHAP scores: {shapley_values}")

        except Exception as e:
            logging.error(f"Failed to fit KernelSHAP linear model: {e}")

        return valuations
