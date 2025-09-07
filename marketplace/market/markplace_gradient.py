import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from common.enums import ServerAttackMode
from common.gradient_market_configs import AppConfig, ServerAttackConfig
from entry.gradient_market.privacy_attack import GradientInversionAttacker
from marketplace.market.data_market import DataMarketplace
from marketplace.market_mechanism.aggregator import Aggregator
from marketplace.seller.seller import BaseSeller


@dataclass
class MarketplaceConfig:
    """Configuration for the DataMarketplaceFederated."""
    save_path: str
    dataset_name: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    privacy_attack_config: ServerAttackConfig = field(default_factory=ServerAttackConfig)


class DataMarketplaceFederated(DataMarketplace):
    def __init__(self, cfg: AppConfig, aggregator: Aggregator, sellers: dict,
                 input_shape: tuple, attacker=None):
        """
        Initializes the marketplace with all necessary components and the main config.
        """
        self.cfg = cfg  # Store the main config object
        self.aggregator = aggregator
        self.sellers = sellers
        self.attacker = attacker  # For server-side privacy attacks

        # Conditionally initialize the privacy attacker
        if self.cfg.server_attack_config.attack_name == ServerAttackMode.GRADIENT_INVERSION:
            # 2. Use the correct variable name: 'self.cfg' not 'self.config'
            self.attacker = GradientInversionAttacker(
                attack_config=self.cfg.server_attack_config,
                model_template=aggregator.strategy.global_model,
                device=aggregator.device,
                # 3. Use the correct path for the save directory
                save_dir=self.cfg.experiment.save_path,

                # 4. Use the correct paths for other experiment parameters
                dataset_name=self.cfg.experiment.dataset_name,
                input_shape=input_shape,  # Passed in, as it's determined at runtime
                num_classes=self.cfg.experiment.num_classes
            )
            logging.info("Initialized GradientInversionAttacker.")

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """Registers a new seller in the marketplace."""
        self.sellers[seller_id] = seller
        logging.info(f"Marketplace: Registered seller '{seller_id}'. Total sellers: {len(self.sellers)}")

    def train_federated_round(
            self,
            round_number: int,
            ground_truth_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict, Any]:
        """Orchestrates a single, config-driven round of federated learning."""
        round_start_time = time.time()
        logging.info(f"--- Round {round_number} Started ---")

        # 1. Collect gradients from all active sellers
        gradients_dict, seller_ids, _ = self._get_current_market_gradients()

        # 2. Perform privacy attack (optional)
        attack_log = None
        if self.attacker and self.attacker.should_run(round_number):
            attack_log = self.attacker.execute(round_number, gradients_dict, seller_ids, ground_truth_dict)

        # 3. Aggregate gradients
        agg_grad, selected_ids, outlier_ids = self.aggregator.aggregate(round_number, gradients_dict)

        # 4. Update global model
        if agg_grad:
            self.aggregator.apply_gradient(agg_grad)

        if self.cfg.debug.save_individual_gradients:
            if round_number % self.cfg.debug.gradient_save_frequency == 0:
                grad_save_dir = Path(self.cfg.experiment.save_path) / "individual_gradients" / f"round_{round_number}"
                grad_save_dir.mkdir(parents=True, exist_ok=True)
                for sid, grad in gradients_dict.items():
                    torch.save(grad, grad_save_dir / f"{sid}_grad.pt")
                logging.info(f"Saved {len(gradients_dict)} individual gradients to {grad_save_dir}")
                torch.save(agg_grad, grad_save_dir / f"aggregated_grad.pt")

        # 6. Create a simple record of the round's events.
        #    The _log_round_results helper can be removed or simplified as it no longer handles performance metrics.
        duration = time.time() - round_start_time
        round_record = {
            "round": round_number,
            "duration_sec": duration,
            "num_selected": len(selected_ids),
            "num_outliers": len(outlier_ids),
            "attack_performed": bool(attack_log),
            "attack_victim": attack_log.get('victim_id') if attack_log else None
        }

        # 7. Notify sellers of round end
        for sid, seller in self.sellers.items():
            seller.round_end_process(round_number, (sid in selected_ids))

        logging.info(f"--- Round {round_number} Ended (Duration: {round_record['duration_sec']:.2f}s) ---")

        # Return the record and the raw gradient for potential future use
        return round_record, agg_grad

    def _get_current_market_gradients(self) -> Tuple[Dict, List, List]:
        """Collects gradients and stats from all sellers."""
        gradients_dict = OrderedDict()
        seller_ids, seller_stats_list = [], []
        for sid, seller in self.sellers.items():
            try:
                grad, stats = seller.get_gradient_for_upload(self.aggregator.strategy.global_model)
                if grad is not None:
                    gradients_dict[sid] = grad
                    seller_ids.append(sid)
                    seller_stats_list.append(stats)
            except Exception as e:
                logging.error(f"Error getting gradient from seller {sid}: {e}", exc_info=True)
        return gradients_dict, seller_ids, seller_stats_list
