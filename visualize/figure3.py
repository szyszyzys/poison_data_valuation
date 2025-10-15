import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_and_process_selection_data(exp_path: Path, group_name: str) -> pd.DataFrame:
    """Loads seller metrics and config, then computes selection rates."""
    log_file = exp_path / "seller_round_metrics.csv"
    config_file = exp_path / "config_snapshot.json"

    if not log_file.exists() or not config_file.exists():
        print(f"Warning: Missing files in {exp_path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(log_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
    adv_rate = config.get('experiment', {}).get('adv_rate', 0.0)

    df['seller_type'] = df['seller_id'].apply(
        lambda sid: 'Attacker' if 'adv' in sid else 'Benign'
    )
    df['group'] = df['seller_type'].apply(
        lambda st: group_name if st == 'Attacker' else 'Benign Control'
    )

    selection_rates = df.groupby('group')['selected'].mean().reset_index()
    selection_rates.rename(columns={'selected': 'selection_rate'}, inplace=True)
    selection_rates['adversary_rate'] = adv_rate

    return selection_rates

def plot_selection_impact(all_data: pd.DataFrame):
    """Generates the plot showing the impact of Sybil attacks on selection rates."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=all_data,
        x='adversary_rate',
        y='selection_rate',
        hue='group',
        style='group',
        markers=True,
        dashes=True,
        markersize=8,
        ax=ax
    )

    ax.set_title('Sybil Attacks Successfully Increase Seller Selection Rate', fontsize=16, weight='bold')
    ax.set_xlabel('Adversary Rate', fontsize=12)
    ax.set_ylabel('Average Selection Rate', fontsize=12)
    ax.legend(title='Seller Group')
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("figure_3_sybil_selection_impact.png", dpi=300)
    print("Plot saved to figure_3_sybil_selection_impact.png")
    plt.show()

if __name__ == '__main__':
    # --- Step 1: Update paths to your experiment output folders ---
    # You will need runs for both baseline (no sybil) and sybil attacks at various adversary rates.
    experiment_dirs = {
        "Standard Backdoor": [
            Path("./results/sybil_baseline_cifar10_cnn_adv0.2/run_0_seed_42"),
            Path("./results/sybil_baseline_cifar10_cnn_adv0.3/run_0_seed_42"),
            Path("./results/sybil_baseline_cifar10_cnn_adv0.4/run_0_seed_42"),
        ],
        "Sybil Backdoor (Mimic)": [
            Path("./results/sybil_mimic_cifar10_cnn_adv0.2/run_0_seed_42"),
            Path("./results/sybil_mimic_cifar10_cnn_adv0.3/run_0_seed_42"),
            Path("./results/sybil_mimic_cifar10_cnn_adv0.4/run_0_seed_42"),
        ]
    }

    all_results = []

    for group_name, paths in experiment_dirs.items():
        for path in paths:
            processed_df = load_and_process_selection_data(path, group_name)
            if not processed_df.empty:
                all_results.append(processed_df)

    if not all_results:
        print("Error: No data was loaded. Please check your paths in 'experiment_dirs'.")
    else:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.groupby(['group', 'adversary_rate']).mean().reset_index()

        print("\n--- Processed Data ---")
        print(final_df)
        print("----------------------\n")

        plot_selection_impact(final_df)