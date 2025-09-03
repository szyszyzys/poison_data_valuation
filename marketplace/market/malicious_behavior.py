import logging
import random
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch import nn
from enum import Enum, auto

class VictimStrategy(Enum):
    RANDOM = auto()
    FIXED = auto()


from entry.gradient_market.privacy_attack import perform_and_evaluate_inversion_attack

@dataclass
class GradientInversionAttackerConfig:
    """Typed configuration for the Gradient Inversion Attack."""
    perform_gradient_inversion: bool = False
    frequency: int = 10
    victim_strategy: str = 'random'
    fixed_victim_idx: int = 0
    lrs_to_try: List[float] = field(default_factory=lambda: [0.01])
    params: Dict[str, Any] = field(default_factory=dict)

class GradientInversionAttacker:
    """Encapsulates all logic for performing a Gradient Inversion Attack."""

    def __init__(
            self,
            attack_config: GradientInversionAttackerConfig,
            model_template: nn.Module,
            device: str,
            save_dir: str,
            # NEW: Add static dataset info here
            dataset_name: str,
            input_shape: Tuple[int, int, int],
            num_classes: int,
    ):
        self.config = attack_config
        self.model_template = model_template
        self.device = torch.device(device)
        self.save_dir = Path(save_dir) / "gradient_inversion"

        # Store dataset info
        self.dataset_name = dataset_name
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Convert strategy string from config to Enum for safer comparison
        self.victim_strategy = VictimStrategy(self.config.victim_strategy)

    def should_run(self, round_number: int) -> bool:
        """Determines if the attack should be performed this round."""
        return self.config.perform_gradient_inversion and (round_number % self.config.frequency == 0)

    def execute(
            self,
            round_number: int,
            gradients_dict: Dict[str, List[torch.Tensor]],
            all_seller_ids: List[str],
            # NEW: Pass ground truth data for evaluation
            ground_truth_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, Any]]:
        """Selects a victim, tunes the attack LR, runs GIA, and returns the best result log."""
        if not all_seller_ids:
            logging.warning("GIA: No seller IDs provided. Skipping attack.")
            return None

        # 1. Select Victim
        if self.victim_strategy == VictimStrategy.RANDOM:
            victim_id = random.choice(all_seller_ids)
        else:  # FIXED
            idx = self.config.fixed_victim_idx
            if not (0 <= idx < len(all_seller_ids)):
                logging.warning(f"GIA: Fixed victim index {idx} is out of bounds. Skipping attack.")
                return None
            victim_id = all_seller_ids[idx]

        target_gradient = gradients_dict.get(victim_id)
        victim_ground_truth = ground_truth_dict.get(victim_id, {})
        gt_images = victim_ground_truth.get("images")
        gt_labels = victim_ground_truth.get("labels")

        if target_gradient is None:
            logging.warning(f"GIA: Could not retrieve gradient for victim '{victim_id}'. Skipping.")
            return None

        # 2. Tune and Run Attack
        logging.info(f"GIA: Starting LR tuning for attack on victim '{victim_id}' (Round {round_number})...")
        best_log, best_psnr = None, -float('inf')

        for lr in self.config.lrs_to_try:
            logging.info(f"  Trying LR: {lr}")

            # Create a GIAConfig instance for the core attack function
            # This combines base parameters with the specific LR for this run
            core_attack_config = GIAConfig(
                lr=lr,
                **self.config.base_attack_params
            )

            # Call the updated orchestrator function with all required arguments
            current_log = perform_and_evaluate_inversion_attack(
                dataset_name=self.dataset_name,
                target_gradient=target_gradient,
                model_template=self.model_template,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                device=self.device,
                attack_config=core_attack_config,  # Pass the typed config object
                ground_truth_images=gt_images,
                ground_truth_labels=gt_labels,
                save_visuals=True,
                save_dir=self.save_dir,
                round_num=round_number,
                victim_id=victim_id,
            )

            # Find the best result based on PSNR
            current_psnr = current_log.get("metrics", {}).get("psnr", -1)
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_log = current_log
                # Store the best LR found so far
                if best_log.get("metrics"):  # Ensure metrics dict exists
                    best_log["metrics"]["best_tuned_lr"] = lr

        if best_log:
            logging.info(
                f"GIA Complete for '{victim_id}'. Best PSNR: {best_psnr:.2f} with LR: {best_log['metrics']['best_tuned_lr']}")

        return best_log






