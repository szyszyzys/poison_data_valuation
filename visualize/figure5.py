import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_metrics_for_buyer_data_sweep(exp_paths: list) -> pd.DataFrame:
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

        # Navigate the nested config to find the buyer_percentage
        buyer_percent = config.get('data', {}).get('image', {}).get('buyer_ratio', 0.1)

        all_metrics.append({
            'Buyer Data Percentage': buyer_percent * 100,  # Convert to percentage for plotting
            'Main Accuracy': metrics.get('main_accuracy', 0.0),
            'ASR': metrics.get('ASR', 0.0)
        })
    return pd.DataFrame(all_metrics)

def plot_buyer_data_impact(data: pd.DataFrame):
    """Generates the plot for the impact of buyer data size."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    df_melted = data.melt(id_vars='Buyer Data Percentage', var_name='Metric', value_name='Value')

    sns.lineplot(
        data=df_melted,
        x='Buyer Data Percentage',
        y='Value',
        hue='Metric',
        style='Metric',
        markers=True,
        markersize=8,
        ax=ax
    )

    ax.set_title('Defense Performance vs. Buyer\'s Local Data Size', fontsize=16, weight='bold')
    ax.set_xlabel('Percentage of Total Data Held by Buyer (%)', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.legend(title='Metric')

    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Format x-axis to show percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))


    plt.tight_layout()
    plt.savefig("figure_5_buyer_data_impact.png", dpi=300)
    print("Plot saved to figure_5_buyer_data_impact.png")
    plt.close()

if __name__ == '__main__':
    # --- Step 1: Update these paths to your experiment output folders ---
    # These should be from a 'buyer_data_impact' scenario where you swept the buyer_percentage.
    experiment_base_path = Path("./results/buyer_data_impact_cifar10_cnn_fltrust/")

    percentages = [0.01, 0.05, 0.10, 0.20] # The percentages you swept

    experiment_paths = [
        experiment_base_path / f"buyer_pct_{pct}/run_0_seed_42" for pct in percentages
    ]

    sweep_data = load_metrics_for_buyer_data_sweep(experiment_paths)

    if sweep_data.empty:
        print("Error: No data was loaded. Please check your paths.")
    else:
        # Sort the data for correct line plotting
        sweep_data.sort_values(by='Buyer Data Percentage', inplace=True)

        print("\n--- Data for Plotting ---")
        print(sweep_data)
        print("-------------------------\n")

        plot_buyer_data_impact(sweep_data)