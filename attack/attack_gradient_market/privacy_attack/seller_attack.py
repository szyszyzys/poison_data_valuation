import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
# --- Import your project's utilities and loader ---
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common.utils import ExperimentLoader
from utils import get_buyer_dataset, model_factory


class MarketplaceSellerAttacker:
    """
    Implements a Buyer Intent Inference Attack from a Sybil adversary's perspective.
    This class now calculates inference scores both WITH and WITHOUT marketplace signals
    to provide a direct comparison and quantify the impact of economic leakage.
    """

    def __init__(self, run_data: Dict[str, Any], attacker_ids: List[str], device: str = "cpu"):
        self.run_data = run_data
        self.attacker_ids = attacker_ids
        self.device = device
        self.mf = model_factory(run_data)
        self.probe_dataset = get_buyer_dataset(run_data)
        self.global_model_history = []
        logging.info(f"MarketplaceSellerAttacker initialized for Sybil adversary controlling: {attacker_ids}.")

    def _get_model_loss(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Helper to calculate the average loss of a model on a dataset."""
        model.eval()
        total_loss, total_samples = 0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def reconstruct_global_model_history(self, loader: ExperimentLoader):
        """Reconstructs the sequence of global models by applying aggregated gradients."""
        logging.info("Reconstructing global model history from aggregated gradients...")
        model = self.mf().to(self.device)
        self.global_model_history.append(model.state_dict())
        learning_rate = 0.01
        max_round = max(self.run_data['gradient_paths'].keys()) if self.run_data['gradient_paths'] else -1

        for round_num in tqdm(range(max_round + 1), desc="Reconstructing Models"):
            agg_grad = loader.load_gradient(self.run_data, round_num, seller_id=None)
            if agg_grad:
                with torch.no_grad():
                    for param, grad in zip(model.parameters(), agg_grad):
                        param.data.sub_(grad, alpha=learning_rate)
                self.global_model_history.append(model.state_dict())
        logging.info(f"Reconstructed {len(self.global_model_history)} global model states.")

    def run_inference_comparison_attack(
            self,
            probe_objectives: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """
        Analyzes model history to infer the buyer's objective using two methods:
        1. Passive-Only: Traditional FL attack based on model improvement.
        2. Marketplace-Aware: Augments the passive attack with aggregated Sybil signals.
        """
        if not self.global_model_history:
            raise RuntimeError("Global model history is empty. Call reconstruct first.")

        logging.info("Running comparative inference attack (Passive vs. Marketplace-Aware)...")

        # --- Step 1: Passively observe global model improvement on probe sets (Baseline) ---
        probe_loaders = {}
        all_labels = [label for _, label in self.probe_dataset]
        for name, classes in probe_objectives.items():
            indices = [i for i, label in enumerate(all_labels) if label in classes]
            probe_loaders[name] = DataLoader(Subset(self.probe_dataset, indices), batch_size=128)

        loss_results = []
        model = self.mf().to(self.device)
        for round_num, model_state in enumerate(tqdm(self.global_model_history, desc="Probing Models")):
            model.load_state_dict(model_state)
            round_losses = {"round": round_num}
            for name, loader in probe_loaders.items():
                loss = self._get_model_loss(model, loader)
                round_losses[f"loss_{name}"] = loss
            loss_results.append(round_losses)

        loss_df = pd.DataFrame(loss_results).set_index('round')
        passive_scores_df = -loss_df.diff().fillna(0)  # Score is the improvement in loss
        passive_scores_df = passive_scores_df.rename(columns=lambda c: c.replace('loss_', 'passive_score_'))

        # --- Step 2: Aggregate signals from all Sybil identities ---
        sybil_histories = [self.run_data["sellers"][sid].set_index('round')[['payment_received', 'assigned_weight']] for
                           sid in self.attacker_ids]
        sybil_agg_history = pd.concat(sybil_histories).groupby('round').sum()

        # --- Step 3: Create Marketplace-Aware scores by augmenting the passive scores ---
        market_scores_df = passive_scores_df.copy().rename(
            columns=lambda c: c.replace('passive_score_', 'market_score_'))
        for col in market_scores_df.columns:
            # Inference Score = Model Improvement * SUM(Payments) * SUM(Weights)
            market_scores_df[col] = market_scores_df[col] * sybil_agg_history['payment_received'] * sybil_agg_history[
                'assigned_weight']

        # --- Step 4: Determine inferred objective for both methods ---
        passive_score_cols = list(passive_scores_df.columns)
        market_score_cols = list(market_scores_df.columns)

        passive_scores_df['passive_inferred_objective'] = passive_scores_df[passive_score_cols].idxmax(
            axis=1).str.replace('passive_score_', '')
        market_scores_df['market_inferred_objective'] = market_scores_df[market_score_cols].idxmax(axis=1).str.replace(
            'market_score_', '')

        # --- Step 5: Combine all results into a single DataFrame ---
        final_df = loss_df.join(passive_scores_df).join(market_scores_df).join(sybil_agg_history).reset_index()
        return final_df


def analyze_and_plot_results(df: pd.DataFrame, ground_truth_objective: str, run_name: str, num_sybils: int):
    """Calculates metrics and plots the results for both attack variations."""
    logging.info("\n--- 📊 FINAL ANALYSIS REPORT (Task 3.2 - Comparative Sybil Attack) 📊 ---")

    # --- Metrics for Passive-Only Attack (Baseline) ---
    passive_final_pred = df['passive_inferred_objective'].iloc[-1]
    passive_accuracy = 1.0 if passive_final_pred == ground_truth_objective else 0.0
    passive_correct = (df['passive_inferred_objective'] == ground_truth_objective)
    passive_latency = 0
    if passive_correct.any():
        for i in range(len(passive_correct)):
            if passive_correct.iloc[i:].all(): passive_latency = df['round'].iloc[i]; break

    logging.info(f"\n--- [Baseline: Passive-Only Attack] ---")
    logging.info(f"Final Accuracy: {passive_accuracy:.2f} (Predicted: {passive_final_pred})")
    logging.info(f"Inference Latency: {passive_latency} rounds")

    # --- Metrics for Marketplace-Aware Attack ---
    market_final_pred = df['market_inferred_objective'].iloc[-1]
    market_accuracy = 1.0 if market_final_pred == ground_truth_objective else 0.0
    market_correct = (df['market_inferred_objective'] == ground_truth_objective)
    market_latency = 0
    if market_correct.any():
        for i in range(len(market_correct)):
            if market_correct.iloc[i:].all(): market_latency = df['round'].iloc[i]; break

    logging.info(f"\n--- [Augmented: Marketplace-Aware Attack] ---")
    logging.info(f"Final Accuracy: {market_accuracy:.2f} (Predicted: {market_final_pred})")
    logging.info(f"Inference Latency: {market_latency} rounds")
    logging.info(f"\nGround Truth Objective: {ground_truth_objective}")

    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'Buyer Intent Inference Comparison (Sybil Attack, {num_sybils} sellers)\nRun: {run_name}',
                 fontsize=16)

    # Plot 1: Marketplace-Aware Scores
    market_score_cols = [col for col in df.columns if col.startswith('market_score_')]
    for col in market_score_cols:
        obj_name = col.replace('market_score_', '')
        ax1.plot(df['round'], df[col], label=obj_name,
                 linestyle='-' if obj_name == ground_truth_objective else '--',
                 linewidth=3.0 if obj_name == ground_truth_objective else 1.5)
    ax1.axvline(x=market_latency, color='r', linestyle=':', linewidth=2, label=f'Latency ({market_latency} rounds)')
    ax1.set_ylabel('Marketplace-Aware Score')
    ax1.set_title('Attack Method 1: Marketplace-Aware Inference (Augmented Signal)')
    ax1.legend();
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Passive-Only Scores
    passive_score_cols = [col for col in df.columns if col.startswith('passive_score_')]
    for col in passive_score_cols:
        obj_name = col.replace('passive_score_', '')
        ax2.plot(df['round'], df[col], label=obj_name,
                 linestyle='-' if obj_name == ground_truth_objective else '--',
                 linewidth=3.0 if obj_name == ground_truth_objective else 1.5)
    ax2.axvline(x=passive_latency, color='g', linestyle=':', linewidth=2, label=f'Latency ({passive_latency} rounds)')
    ax2.set_ylabel('Passive Score (Loss Improvement)')
    ax2.set_title('Attack Method 2: Passive-Only Inference (Baseline)')
    ax2.legend();
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Raw Loss Trajectories
    loss_cols = [col for col in df.columns if col.startswith('loss_')]
    for col in loss_cols:
        obj_name = col.replace('loss_', '')
        ax3.plot(df['round'], df[col], label=obj_name,
                 linestyle='-' if obj_name == ground_truth_objective else '--',
                 linewidth=2.0 if obj_name == ground_truth_objective else 1.0)
    ax3.set_xlabel('Training Round');
    ax3.set_ylabel('Loss on Probe Dataset')
    ax3.set_title('Underlying Signal: Loss Trajectory for Each Potential Objective')
    ax3.legend();
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f"task3_2_sybil_intent_inference_comparison_{run_name}.png"
    plt.savefig(save_path)
    logging.info(f"Results plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    EXPERIMENT_ROOT = "./exp_results/text_agnews_cnn_10seller"
    TARGET_RUN_INDEX = 0
    GROUND_TRUTH_OBJECTIVE = "Focus on Sci/Tech (Class 2)"

    try:
        loader = ExperimentLoader(EXPERIMENT_ROOT)
        all_runs = loader.load_all_runs_data()
        target_run_data = all_runs[TARGET_RUN_INDEX]
        run_name = target_run_data['run_path'].name
        logging.info(f"Loaded data for run: {run_name}")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load experiment data: {e}")
        exit()

    try:
        all_seller_ids = list(target_run_data['sellers'].keys())
        adversary_seller_ids = [sid for sid in all_seller_ids if sid.startswith('adv')]

        if not adversary_seller_ids:
            raise ValueError("No adversary sellers found (e.g., named 'adv_...'). Cannot run Sybil attack.")

        attacker = MarketplaceSellerAttacker(target_run_data, attacker_ids=adversary_seller_ids, device="cpu")

        probe_objectives = {
            "Focus on World News (Class 0)": [0], "Focus on Sports (Class 1)": [1],
            "Focus on Sci/Tech (Class 2)": [2], "Focus on Business (Class 3)": [3],
            "General (All Classes)": [0, 1, 2, 3]
        }

        attacker.reconstruct_global_model_history(loader)
        inference_df = attacker.run_inference_comparison_attack(probe_objectives)
        analyze_and_plot_results(inference_df, GROUND_TRUTH_OBJECTIVE, run_name, len(adversary_seller_ids))

    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
