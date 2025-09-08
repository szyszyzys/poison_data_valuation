import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from common.utils import ExperimentLoader


def create_mock_gradient(norm=1.0, sparsity=0.1, seed=None):
    """Creates a mock gradient vector for probe attacks."""
    rng = np.random.default_rng(seed)
    grad = rng.random(100)
    grad[rng.random(100) < sparsity] = 0
    return grad * (norm / np.linalg.norm(grad))


class Adversary(ABC):
    """Abstract base class for all adversaries."""

    def __init__(self, adversary_id: str):
        self.adversary_id = adversary_id
        self.knowledge = {}
        logging.info(f"Adversary '{self.adversary_id}' initialized.")

    @abstractmethod
    def execute_attack(self, marketplace_state: Dict[str, Any]):
        pass

    def log_knowledge(self, key: str, value: Any):
        """Adds a piece of learned information to the adversary's knowledge base."""
        self.knowledge[key] = value
        logging.info(f"Adversary '{self.adversary_id}' learned: {key} = {value}")


class SellerAdversary(Adversary):
    """Represents a malicious seller or a Sybil coalition."""

    def __init__(self, adversary_id: str, controls_sybil_identities: List[str], can_craft_gradients: bool = True):
        super().__init__(adversary_id)
        self.sybil_identities = controls_sybil_identities
        self.can_craft_gradients = can_craft_gradients
        self.attack_strategy = None
        self.payment_history = pd.DataFrame()

    def set_attack_strategy(self, strategy_name: str):
        strategies = {"monitor_convergence": self.monitor_convergence}
        if strategy_name in strategies:
            self.attack_strategy = strategies[strategy_name]
            logging.info(f"SellerAdversary '{self.adversary_id}' strategy set to '{strategy_name}'.")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def execute_attack(self, marketplace_state: Dict[str, Any] = None):
        if self.attack_strategy:
            return self.attack_strategy(marketplace_state)
        logging.warning("No attack strategy set for seller adversary.")
        return None

    def update_history(self, round_history: pd.DataFrame):
        """Updates the adversary's observed history from the marketplace logs."""
        sybil_history = round_history[round_history['seller_id'].isin(self.sybil_identities)]
        self.payment_history = pd.concat([self.payment_history, sybil_history])

    def monitor_convergence(self, marketplace_state: Dict[str, Any] = None):
        """ATTACK: Analyzes payment history to infer model convergence."""
        logging.info("Executing 'monitor_convergence' attack...")
        if len(self.payment_history['round'].unique()) > 3:
            avg_payments = self.payment_history.groupby('round')['payment_received'].mean()
            if avg_payments.iloc[-1] < avg_payments.iloc[-2] < avg_payments.iloc[-3]:
                self.log_knowledge("inferred_convergence_state", "Model may be converging (payments decreasing).")


class BuyerAdversary(Adversary):
    """Represents a malicious buyer running the marketplace."""

    def __init__(self, adversary_id: str, can_design_probe_queries: bool = True):
        super().__init__(adversary_id)
        self.can_design_probe_queries = can_design_probe_queries

    def execute_attack(self, marketplace_state: Dict[str, Any]):
        if self.can_design_probe_queries:
            self.fingerprint_sellers(marketplace_state.get("submitted_gradients", {}))

    def fingerprint_sellers(self, submitted_gradients: Dict[str, np.ndarray]):
        """ATTACK: Analyzes gradients to identify sellers with rare data."""
        logging.info("Executing 'fingerprint_sellers' attack...")
        # In a real scenario, the buyer would know the 'signature' of a rare feature's gradient.
        # Here we simulate this by looking for a gradient with an unusually high norm.
        fingerprinted_sellers = []
        for seller_id, grad in submitted_gradients.items():
            if np.linalg.norm(grad) > 1.2:  # Assuming 1.0 is normal
                fingerprinted_sellers.append(seller_id)

        if fingerprinted_sellers:
            self.log_knowledge("fingerprinted_sellers_with_rare_data", fingerprinted_sellers)


# --- PART 3: Analysis Orchestration ---

def create_adversaries_from_run(run_data: Dict[str, Any], adv_prefix="adv") -> Dict[str, Adversary]:
    """Factory to create adversary objects based on logged data."""
    adversaries = {}

    # Create Seller Adversary if Sybils are found
    all_sellers = list(run_data["sellers"].keys())
    sybil_ids = [sid for sid in all_sellers if sid.startswith(adv_prefix)]
    if sybil_ids:
        adversaries["seller_adversary"] = SellerAdversary(
            "Sybil-Master",
            controls_sybil_identities=sybil_ids
        )

    # Always create a buyer adversary for analysis
    adversaries["buyer_adversary"] = BuyerAdversary("Malicious-Buyer")

    return adversaries


class PostHocAnalysisPipeline:
    """Orchestrates a post-hoc analysis of a completed experiment run."""

    def __init__(self, run_data: Dict[str, Any], loader: ExperimentLoader):
        self.run_data = run_data
        self.loader = loader
        self.adversaries = create_adversaries_from_run(run_data)

    def run_analysis(self):
        """Replays the experiment round-by-round to perform adversarial analysis."""
        logging.info(f"\n--- Starting Post-Hoc Analysis for Run: {self.run_data['run_path'].name} ---")
        seller_adv = self.adversaries.get("seller_adversary")
        buyer_adv = self.adversaries.get("buyer_adversary")

        # Determine the total number of rounds from seller logs
        any_seller_id = next(iter(self.run_data["sellers"]))
        total_rounds = self.run_data["sellers"][any_seller_id]['round'].max()

        for round_num in range(total_rounds + 1):
            logging.info(f"--- Analyzing Round {round_num} ---")

            # --- Seller Adversary Actions ---
            if seller_adv:
                # Update seller's knowledge with this round's results
                round_history = pd.concat([df[df['round'] == round_num] for df in self.run_data["sellers"].values()])
                seller_adv.update_history(round_history)
                # Execute analysis based on history
                seller_adv.execute_attack()

            # --- Buyer Adversary Actions ---
            if buyer_adv and round_num in self.run_data["gradient_paths"]:
                # Buyer collects all gradients from this round
                submitted_gradients = {
                    sid: self.loader.load_gradient(g_path)
                    for sid, g_path in self.run_data["gradient_paths"][round_num].items()
                }
                # Filter out any that failed to load
                submitted_gradients = {k: v for k, v in submitted_gradients.items() if v is not None}

                marketplace_state = {"submitted_gradients": submitted_gradients}
                buyer_adv.execute_attack(marketplace_state)

        logging.info("\n--- Analysis Complete. Final Learned Knowledge: ---")
        for adv_name, adv_obj in self.adversaries.items():
            print(f"  - {adv_name} ({adv_obj.adversary_id}): {adv_obj.knowledge}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # --- IMPORTANT: Set this to your experiment results directory ---
    # This directory should contain subfolders like 'run_0_seed_123', 'run_1_seed_456', etc.
    EXPERIMENT_ROOT = "./exp_results/text_agnews_cnn_10seller"

    try:
        loader = ExperimentLoader(EXPERIMENT_ROOT)

        # Analyze the first run found by the loader
        if loader.runs:
            first_run_path = loader.runs[0]
            run_data = loader.load_run_data(first_run_path)

            # Check if run data is usable for analysis
            if not run_data["sellers"] or not run_data["gradient_paths"]:
                raise RuntimeError(
                    f"Run {first_run_path.name} is missing necessary seller history or gradient files for analysis.")

            # Setup seller adversary and set a strategy to test
            # This part is still manual: you decide which attack to run post-hoc
            pipeline = PostHocAnalysisPipeline(run_data, loader)
            if "seller_adversary" in pipeline.adversaries:
                pipeline.adversaries["seller_adversary"].set_attack_strategy("monitor_convergence")

            pipeline.run_analysis()

        else:
            logging.warning("No experiment runs found in the specified directory.")

    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"Analysis failed. {e}")
        logging.error(
            "Please ensure EXPERIMENT_ROOT is set correctly and that the target run folder contains the expected log files.")
