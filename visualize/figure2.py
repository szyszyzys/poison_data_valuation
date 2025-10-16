import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

def load_metrics_for_sweep(exp_paths: list) -> pd.DataFrame:
    """Loads final metrics and config from a list of experiment paths."""
    all_metrics = []
    for path in exp_paths:
        metrics_file = path / "final_metrics.json"
        config_file = path / "config_snapshot.json"

        if not metrics_file.exists() or not config_file.exists():
            print(f"Warning: Missing files in {path}. Skipping.")
            continue

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        with open(config_file, 'r') as f:
            config = json.load(f)

        all_metrics.append({
            'Adversary Rate': config.get('experiment', {}).get('adv_rate', 0.0),
            'Main Accuracy': metrics.get('main_accuracy', 0.0),
            'ASR': metrics.get('ASR', 0.0)
        })
    return pd.DataFrame(all_metrics)

def plot_accuracy_vs_asr(data: pd.DataFrame):
    """Generates the dual-axis plot for Accuracy vs. ASR."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Main Accuracy on the primary y-axis (ax1)
    sns.lineplot(
        data=data, x='Adversary Rate', y='Main Accuracy',
        ax=ax1, color='royalblue', marker='o', markersize=8,
        label='Main Accuracy'
    )
    ax1.set_xlabel('Adversary Rate', fontsize=12)
    ax1.set_ylabel('Main Accuracy', fontsize=12, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(0, 1.0)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    sns.lineplot(
        data=data, x='Adversary Rate', y='ASR',
        ax=ax2, color='orangered', marker='s', markersize=8,
        label='Attack Success Rate (ASR)'
    )
    ax2.set_ylabel('Attack Success Rate (ASR)', fontsize=12, color='orangered')
    ax2.tick_params(axis='y', labelcolor='orangered')
    ax2.set_ylim(0, 1.0)

    ax1.set_title('Attack Stealthiness: ASR Increases While Accuracy Remains High', fontsize=16, weight='bold')

    # Unify legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.get_legend().remove()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig("figure_2_stealth_attack.png", dpi=300)
    print("Plot saved to figure_2_stealth_attack.png")
    plt.close()

if __name__ == '__main__':
    # --- Step 1: Update these paths to your experiment output folders ---
    # These should be from an 'attack_impact' scenario where you swept the adversary rate.
    # For example, for the MartFL aggregator and a cnn model on cifar10.
    experiment_base_path = Path("./results/impact_vary_adv_rate_cifar10_cnn_martfl/")

    adv_rates = [0.1, 0.2, 0.3, 0.4, 0.5] # The rates you swept in your config

    experiment_paths = [
        experiment_base_path / f"adv_rate_{rate}/run_0_seed_42" for rate in adv_rates
    ]

    # Add a "no attack" baseline for a clean starting point if you have it
    # no_attack_path = Path("./results/no_attack_cifar10_cnn_martfl/run_0_seed_42")
    # if no_attack_path.exists():
    #     experiment_paths.insert(0, no_attack_path)

    sweep_data = load_metrics_for_sweep(experiment_paths)

    if sweep_data.empty:
        print("Error: No data was loaded. Please check your paths.")
    else:
        # Sort the data by adversary rate to ensure lines are drawn correctly
        sweep_data.sort_values(by='Adversary Rate', inplace=True)

        print("\n--- Data for Plotting ---")
        print(sweep_data)
        print("-------------------------\n")

        plot_accuracy_vs_asr(sweep_data)