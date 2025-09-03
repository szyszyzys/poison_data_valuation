import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# --- Assumed Imports from Your Project ---
# These are placeholders for your actual class implementations.
from attack.evaluation.evaluation_backdoor import evaluate_attack_performance
from common.gradient_market_configs import ServerPrivacyConfig
from entry.gradient_market.privacy_attack import GradientInversionAttacker
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.seller import BaseSeller


# --- Placeholder Base Class ---
class DataMarketplace:
    """Base class placeholder for inheritance."""
    pass


# --- NEW: Typed Configuration for the Marketplace ---
@dataclass
class MarketplaceConfig:
    """Configuration for the DataMarketplaceFederated."""
    save_path: str
    dataset_name: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    privacy_attack_config: ServerPrivacyConfig = field(default_factory=ServerPrivacyConfig)


# --- Helper Class for Evaluation (Unchanged) ---
class FederatedEvaluator:
    """Encapsulates all logic for evaluating a federated model."""

    def __init__(self, loss_fn: nn.Module, device: str):
        self.loss_fn = loss_fn
        self.device = device

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluates model accuracy and loss."""
        model.to(self.device)
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                total_loss += self.loss_fn(outputs, y).item() * X.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == y).sum().item()
                total_samples += X.size(0)
        return {"acc": total_correct / total_samples, "loss": total_loss / total_samples}

    def evaluate_backdoor_asr(self, model: nn.Module, test_loader: DataLoader, backdoor_generator,
                              target_label) -> float:
        """Evaluates the Attack Success Rate for a backdoor attack."""
        poison_metrics = evaluate_attack_performance(
            model, test_loader, self.device, backdoor_generator, target_label
        )
        return poison_metrics.get("attack_success_rate")


# --- Main Marketplace Class (Updated) ---
class DataMarketplaceFederated(DataMarketplace):
    def __init__(self, aggregator: Aggregator, config: MarketplaceConfig):
        self.aggregator = aggregator
        self.config = config
        self.save_path = Path(config.save_path)
        self.sellers: OrderedDict[str, BaseSeller] = OrderedDict()

        # NEW: Setup for incremental logging to a CSV file
        self.log_file_path = self.save_path / "round_logs.csv"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.evaluator = FederatedEvaluator(loss_fn=nn.CrossEntropyLoss(), device=aggregator.device)
        self.attacker = None

        # Conditionally initialize the privacy attacker
        if self.config.privacy_attack_config.perform_gradient_inversion:
            self.attacker = GradientInversionAttacker(
                attack_config=self.config.privacy_attack_config,
                model_template=aggregator.global_model,
                device=aggregator.device,
                save_dir=str(self.save_path),
                # CHANGED: Pass necessary dataset info from the main config
                dataset_name=self.config.dataset_name,
                input_shape=self.config.input_shape,
                num_classes=self.config.num_classes
            )

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """Registers a new seller in the marketplace."""
        self.sellers[seller_id] = seller
        logging.info(f"Marketplace: Registered seller '{seller_id}'. Total sellers: {len(self.sellers)}")

    def train_federated_round(
            self,
            round_number: int,
            test_loader_global: DataLoader,
            # CHANGED: Now requires ground_truth_dict for the privacy attack evaluation
            ground_truth_dict: Dict[str, Dict[str, torch.Tensor]],
            backdoor_generator=None,
            backdoor_target_label=None
    ) -> Tuple[Dict, Any]:
        """Orchestrates a single round of federated learning."""
        round_start_time = time.time()
        logging.info(f"--- Round {round_number} Started ---")

        # 1. Collect gradients from all active sellers
        gradients_dict, seller_ids, _ = self._get_current_market_gradients()

        # 2. Perform privacy attack (optional)
        attack_log = None
        if self.attacker and self.attacker.should_run(round_number):
            # CHANGED: Pass the ground truth data to the attacker for evaluation
            attack_log = self.attacker.execute(round_number, gradients_dict, seller_ids, ground_truth_dict)

        # 3. Aggregate gradients
        agg_grad, selected_ids, outlier_ids = self.aggregator.aggregate(round_number, gradients_dict)

        # 4. Update global model
        if agg_grad:
            self.aggregator.apply_gradient(agg_grad)

        # 5. Evaluate the updated model
        perf_global = self.evaluator.evaluate(self.aggregator.global_model, test_loader_global)
        if backdoor_generator:
            perf_global["asr"] = self.evaluator.evaluate_backdoor_asr(
                self.aggregator.global_model, test_loader_global, backdoor_generator, backdoor_target_label
            )

        # 6. Log results for the round and save incrementally
        round_record = self._log_round_results(
            round_number, time.time() - round_start_time, perf_global,
            selected_ids, outlier_ids, attack_log
        )

        # 7. Notify sellers of round end
        for sid, seller in self.sellers.items():
            seller.round_end_process(round_number, (sid in selected_ids))

        logging.info(f"--- Round {round_number} Ended (Duration: {round_record['duration_sec']:.2f}s) ---")
        return round_record, agg_grad

    def _get_current_market_gradients(self) -> Tuple[Dict, List, List]:
        """Collects gradients and stats from all sellers."""
        gradients_dict = OrderedDict()
        seller_ids, seller_stats_list = [], []
        for sid, seller in self.sellers.items():
            try:
                grad, stats = seller.get_gradient_for_upload(self.aggregator.global_model)
                if grad is not None:
                    gradients_dict[sid] = grad
                    seller_ids.append(sid)
                    seller_stats_list.append(stats)
            except Exception as e:
                logging.error(f"Error getting gradient from seller {sid}: {e}", exc_info=True)
        return gradients_dict, seller_ids, seller_stats_list

    def _log_round_results(
            self, round_num: int, duration: float, perf: Dict,
            selected: List[str], outliers: List[str], attack_log: Optional[Dict]
    ) -> Dict:
        """Compiles the log dictionary for the round and saves it to a file."""
        log_entry = {
            "round": round_num,
            "duration_sec": duration,
            "global_acc": perf.get('acc'),
            "global_loss": perf.get('loss'),
            "global_asr": perf.get('asr'),
            "num_selected": len(selected),
            "num_outliers": len(outliers),
            "attack_performed": bool(attack_log),
            "attack_victim": attack_log.get('victim_id') if attack_log else None,
            "attack_best_psnr": attack_log.get('metrics', {}).get('psnr') if attack_log else None,
            "attack_best_lr": attack_log.get('metrics', {}).get('best_tuned_lr') if attack_log else None,
        }

        # NEW: Append log to CSV file incrementally
        try:
            new_log_df = pd.DataFrame([log_entry])
            if not self.log_file_path.exists():
                new_log_df.to_csv(self.log_file_path, index=False, header=True)
            else:
                new_log_df.to_csv(self.log_file_path, index=False, header=False, mode='a')
        except Exception as e:
            logging.error(f"Failed to write to log file {self.log_file_path}: {e}")

        return log_entry
