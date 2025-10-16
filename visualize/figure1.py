import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_final_metrics(exp_path: Path, aggregator_name: str) -> dict:
    """Loads the final_metrics.json and adds the aggregator name."""
    metrics_file = exp_path / "final_metrics.json"

    if not metrics_file.exists():
        print(f"Warning: Missing final_metrics.json in {exp_path}. Skipping.")
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return {
        'Aggregator': aggregator_name,
        'Main Accuracy': metrics.get('main_accuracy', 0.0),
        'ASR': metrics.get('ASR', 0.0)
    }

def plot_main_summary(data: pd.DataFrame):
    """Generates the main summary bar chart."""
    df_melted = data.melt(id_vars='Aggregator', var_name='Metric', value_name='Value')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df_melted,
        x='Aggregator',
        y='Value',
        hue='Metric',
        ax=ax,
        palette={'Main Accuracy': 'royalblue', 'ASR': 'orangered'}
    )

    ax.set_title('Defense Performance Against a Coordinated Sybil Attack', fontsize=16, weight='bold')
    ax.set_xlabel('Aggregation Method', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Metric')
    ax.grid(axis='y', linestyle='--', linewidth=0.7)

    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig("figure_1_main_summary.png", dpi=300)
    print("Plot saved to figure_1_main_summary.png")
    plt.close()

if __name__ == '__main__':
    # --- Step 1: Update these paths to your experiment output folders ---
    # These should all use the same strong attack settings (e.g., Sybil 'mimic', adv_rate=0.3)
    # but each with a different aggregator.
    experiment_dirs = {
        "FedAvg": Path("./results/main_summary_cifar10_cnn_fedavg/run_0_seed_42"),
        "FLTrust": Path("./results/main_summary_cifar10_cnn_fltrust/run_0_seed_42"),
        "MartFL": Path("./results/main_summary_cifar10_cnn_martfl/run_0_seed_42"),
        "SkyMask": Path("./results/main_summary_cifar10_cnn_skymask/run_0_seed_42"),
    }

    all_metrics = []
    for aggregator, path in experiment_dirs.items():
        metrics = load_final_metrics(path, aggregator)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        print("Error: No data was loaded. Please check your paths in 'experiment_dirs'.")
    else:
        final_df = pd.DataFrame(all_metrics)
        print("\n--- Data for Plotting ---")
        print(final_df)
        print("-------------------------\n")

        plot_main_summary(final_df)