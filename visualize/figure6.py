import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

def calculate_gini(x):
    """Calculate the Gini coefficient of a numpy array."""
    x = np.asarray(x, dtype=float)
    if np.amin(x) < 0:
        # Values cannot be negative:
        x -= np.amin(x)
    # Values cannot be 0:
    x += 0.0000001
    # Values must be sorted:
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

def load_and_process_economic_data(exp_path: Path, scenario_name: str) -> dict:
    """Loads log data and calculates cost composition and Gini coefficient."""
    log_file = exp_path / "seller_round_metrics.csv"
    if not log_file.exists():
        print(f"Warning: Missing log file in {exp_path}. Skipping.")
        return None

    df = pd.read_csv(log_file)
    df['seller_type'] = df['seller_id'].apply(lambda sid: 'Malicious' if 'adv' in sid else 'Benign')

    # --- 1. Calculate Cost Composition ---
    # Total selections (payments) per group
    cost_per_group = df[df['selected']].groupby('seller_type').size()
    total_rounds = df['round'].max()
    avg_cost_benign = cost_per_group.get('Benign', 0) / total_rounds
    avg_cost_malicious = cost_per_group.get('Malicious', 0) / total_rounds

    # --- 2. Calculate Gini Coefficient for Benign Sellers ---
    benign_df = df[df['seller_type'] == 'Benign']
    # Total payments per benign seller
    benign_payments = benign_df.groupby('seller_id')['selected'].sum().values
    gini = calculate_gini(benign_payments) if len(benign_payments) > 1 else 0

    return {
        'Scenario': scenario_name,
        'Avg Cost (Benign)': avg_cost_benign,
        'Avg Cost (Malicious)': avg_cost_malicious,
        'Gini (Benign Sellers)': gini
    }

def plot_economic_impact(data: pd.DataFrame):
    """Generates a two-panel plot for economic and fairness analysis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Panel 1: Cost Composition (Stacked Bar Chart) ---
    cost_df = data[['Scenario', 'Avg Cost (Benign)', 'Avg Cost (Malicious)']]
    cost_df.set_index('Scenario').plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color={'Avg Cost (Benign)': 'skyblue', 'Avg Cost (Malicious)': 'salmon'}
    )
    ax1.set_title('Attacks Siphon Payments from Benign Sellers', fontsize=14, weight='bold')
    ax1.set_ylabel('Average Payments per Round', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(["Payments to Benign Sellers", "Payments to Malicious Sellers"])

    # --- Panel 2: Gini Coefficient (Point Plot) ---
    sns.pointplot(data=data, x='Scenario', y='Gini (Benign Sellers)', ax=ax2, color='darkgreen', markers='D', scale=1.2)
    ax2.set_title('Sybil Attacks Deceptively "Equalize" Benign Seller Payments', fontsize=14, weight='bold')
    ax2.set_ylabel('Gini Coefficient (0=Equal, 1=Unequal)', fontsize=12)
    ax2.set_xlabel('Attack Scenario', fontsize=12)
    ax2.set_ylim(bottom=0)

    fig.suptitle('Economic and Fairness Impact of Marketplace Attacks', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figure_6_economic_impact.png", dpi=300)
    print("Plot saved to figure_6_economic_impact.png")
    plt.close()

if __name__ == '__main__':
    # --- Step 1: Update paths to experiments with a fixed adversary rate (e.g., 0.3) ---
    experiment_dirs = {
        "No Attack": Path("./results/sybil_baseline_cifar10_cnn_adv0.0/run_0_seed_42"), # Assuming you have a no-attack run
        "Standard Backdoor": Path("./results/sybil_baseline_cifar10_cnn_adv0.3/run_0_seed_42"),
        "Sybil Backdoor (Mimic)": Path("./results/sybil_mimic_cifar10_cnn_adv0.3/run_0_seed_42"),
    }

    all_data = []
    for scenario, path in experiment_dirs.items():
        processed_data = load_and_process_economic_data(path, scenario)
        if processed_data:
            all_data.append(processed_data)

    if not all_data:
        print("Error: No data was loaded. Please check your paths.")
    else:
        final_df = pd.DataFrame(all_data)
        print("\n--- Data for Plotting ---")
        print(final_df)
        print("-------------------------\n")

        plot_economic_impact(final_df)