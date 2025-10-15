import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def simulate_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Applies different payment models to the log data post-hoc."""

    # Model 1: Quality-Based Pricing
    # Payment = similarity_score * base_rate (e.g., 1.0)
    # Only selected sellers are eligible for payment.
    df['payment_quality'] = df.apply(
        lambda row: max(0, row['sim_to_buyer_root']) if row['selected'] else 0,
        axis=1
    )

    # Model 2: Risk-Adjusted Payment
    # Payment = base_payment (1.0) if selected, but penalized if an outlier.
    def risk_adjusted_payment(row):
        if not row['selected']:
            return 0.0
        # Give a base payment, but apply a heavy penalty if flagged as an outlier
        penalty = 0.9 if row['outlier'] else 0.0
        return 1.0 * (1 - penalty)

    df['payment_risk_adjusted'] = df.apply(risk_adjusted_payment, axis=1)

    # Model 3: Contribution-Weighted Reward
    # Payment is inversely proportional to the seller's training loss.
    # Lower loss -> higher contribution -> higher reward.
    max_loss = df['train_loss'].max() # Find max loss for normalization

    def contribution_payment(row):
        if not row['selected'] or pd.isna(row['train_loss']):
            return 0.0
        # Invert and normalize the loss to create a reward score
        # We add a small epsilon to avoid division by zero
        reward_score = (max_loss - row['train_loss']) / (max_loss + 1e-6)
        return max(0, reward_score) # Ensure payment is non-negative

    df['payment_contribution'] = df.apply(contribution_payment, axis=1)

    return df

def analyze_economic_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates total payments and Gini for each payment model."""

    df['seller_type'] = df['seller_id'].apply(lambda sid: 'Malicious' if 'adv' in sid else 'Benign')
    payment_models = ['payment_quality', 'payment_risk_adjusted', 'payment_contribution']

    results = []

    for model in payment_models:
        # Calculate total payments per group
        total_payments = df.groupby('seller_type')[model].sum()

        # Calculate Gini for benign sellers
        benign_payments = df[df['seller_type'] == 'Benign'].groupby('seller_id')[model].sum().values
        gini = calculate_gini(benign_payments) if len(benign_payments) > 1 else 0

        results.append({
            'Payment Model': model.replace('payment_', '').replace('_', ' ').title(),
            'Total Benign Payments': total_payments.get('Benign', 0),
            'Total Malicious Payments': total_payments.get('Malicious', 0),
            'Gini (Benign Sellers)': gini
        })

    return pd.DataFrame(results)

def calculate_gini(x):
    """Helper function to calculate the Gini coefficient."""
    x = np.asarray(x, dtype=float) + 1e-9 # Add epsilon to avoid all zeros
    x = np.sort(x)
    n = x.shape[0]
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

def plot_payment_model_comparison(data: pd.DataFrame):
    """Plots the comparison of different payment models."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Who Gets Paid? (Stacked Bar) ---
    cost_df = data.melt(
        id_vars='Payment Model',
        value_vars=['Total Benign Payments', 'Total Malicious Payments'],
        var_name='Recipient', value_name='Total Payment'
    )
    sns.barplot(
        data=cost_df, x='Payment Model', y='Total Payment', hue='Recipient',
        ax=ax1, palette={'Total Benign Payments': 'skyblue', 'Total Malicious Payments': 'salmon'}
    )
    ax1.set_title('Distribution of Total Payments by Model', fontsize=14, weight='bold')
    ax1.set_ylabel('Total Simulated Payment', fontsize=12)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=15)

    # --- Panel 2: Is it Fair? (Bar) ---
    sns.barplot(data=data, x='Payment Model', y='Gini (Benign Sellers)', ax=ax2, color='darkseagreen')
    ax2.set_title('Fairness Among Benign Sellers', fontsize=14, weight='bold')
    ax2.set_ylabel('Gini Coefficient (0=Equal, 1=Unequal)', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='x', rotation=15)

    fig.suptitle('Post-Hoc Economic Analysis of Simulated Payment Models', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("final_payment_analysis.png", dpi=300)
    print("Plot saved to final_payment_analysis.png")
    plt.show()


if __name__ == '__main__':
    # --- Step 1: Point this to the log file from a revealing experiment ---
    # A Sybil attack run is perfect for this analysis (e.g., adv_rate=0.3)
    experiment_log_path = Path("./results/sybil_mimic_cifar10_cnn_adv0.3/run_0_seed_42/seller_round_metrics.csv")

    if not experiment_log_path.exists():
        print(f"Error: Log file not found at {experiment_log_path}")
    else:
        # Load the raw log data
        log_df = pd.read_csv(experiment_log_path)

        # Run the payment simulation
        simulated_df = simulate_payments(log_df)

        # Analyze the outcomes of the simulation
        analysis_results = analyze_economic_outcomes(simulated_df)

        print("\n--- Economic Analysis Results ---")
        print(analysis_results)
        print("---------------------------------\n")

        # Plot the final comparison
        plot_payment_model_comparison(analysis_results)