import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_metrics_for_heterogeneity_sweep(exp_paths: list) -> pd.DataFrame:
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

        # Navigate the nested config to find the dirichlet_alpha
        alpha = config.get('data', {}).get('image', {}).get('property_skew', {}).get('dirichlet_alpha', 1.0)

        all_metrics.append({
            'Dirichlet Alpha': alpha,
            'Main Accuracy': metrics.get('main_accuracy', 0.0),
            'ASR': metrics.get('ASR', 0.0)
        })
    return pd.DataFrame(all_metrics)

def plot_heterogeneity_impact(data: pd.DataFrame):
    """Generates the plot for the impact of data heterogeneity."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Melt the DataFrame for easier plotting with seaborn
    df_melted = data.melt(id_vars='Dirichlet Alpha', var_name='Metric', value_name='Value')

    sns.lineplot(
        data=df_melted,
        x='Dirichlet Alpha',
        y='Value',
        hue='Metric',
        style='Metric',
        markers=True,
        markersize=8,
        ax=ax
    )

    ax.set_title('Defense Performance Degrades with Increased Data Heterogeneity', fontsize=16, weight='bold')
    ax.set_xlabel('Dirichlet Alpha (Higher value = More IID)', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.legend(title='Metric')

    # Use a logarithmic scale for the x-axis as Dirichlet alpha spans orders of magnitude
    ax.set_xscale('log')
    ax.set_xticks(data['Dirichlet Alpha'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("figure_4_heterogeneity_impact.png", dpi=300)
    print("Plot saved to figure_4_heterogeneity_impact.png")
    plt.show()

if __name__ == '__main__':
    # --- Step 1: Update these paths to your experiment output folders ---
    # These should be from a 'heterogeneity_impact' scenario where you swept the Dirichlet alpha.
    experiment_base_path = Path("./results/heterogeneity_impact_cifar10_cnn_fltrust/")

    alphas = [100.0, 1.0, 0.1] # The alpha values you swept in your config

    experiment_paths = [
        experiment_base_path / f"alpha_{alpha}/run_0_seed_42" for alpha in alphas
    ]

    sweep_data = load_metrics_for_heterogeneity_sweep(experiment_paths)

    if sweep_data.empty:
        print("Error: No data was loaded. Please check your paths.")
    else:
        # Sort the data by alpha for correct line plotting
        sweep_data.sort_values(by='Dirichlet Alpha', ascending=False, inplace=True)

        print("\n--- Data for Plotting ---")
        print(sweep_data)
        print("-------------------------\n")

        plot_heterogeneity_impact(sweep_data)