import pandas as pd
from pathlib import Path

def calculate_selection_rates(exp_path: Path) -> dict:
    """
    Calculates the average selection rate for benign and malicious sellers
    from a single experiment's log file.
    """
    log_file = exp_path / "seller_round_metrics.csv"
    if not log_file.exists():
        # Return default values if log file is missing
        return {'Benign': float('nan'), 'Malicious': float('nan')}

    df = pd.read_csv(log_file)
    if 'selected' not in df.columns:
        return {'Benign': float('nan'), 'Malicious': float('nan')}

    df['seller_type'] = df['seller_id'].apply(lambda sid: 'Malicious' if 'adv' in sid else 'Benign')

    # Calculate mean selection rate for each group that exists in the data
    rates = df.groupby('seller_type')['selected'].mean().to_dict()

    # Ensure both keys exist for consistency
    rates.setdefault('Benign', float('nan'))
    rates.setdefault('Malicious', float('nan'))

    return rates

if __name__ == '__main__':
    # --- Step 1: Update these paths to your experiment output folders ---
    # This requires a broad set of runs:
    # 1. Baseline (no-attack) runs for each dataset to get the BSR.
    # 2. Attack runs for each combination of aggregator and dataset to get the MSR.

    datasets = ["CIFAR-10", "TREC"]
    aggregators = ["MartFL", "FLTrust", "SkyMask"]

    experiment_paths = {
        "Baselines": {
            "CIFAR-10": Path("./results/baseline_cifar10_cnn/run_0_seed_42"),
            "TREC": Path("./results/baseline_trec/run_0_seed_42"),
        },
        "Attacks": {
            "MartFL": {
                "CIFAR-10": Path("./results/main_summary_cifar10_cnn_martfl/run_0_seed_42"),
                "TREC": Path("./results/main_summary_trec_martfl/run_0_seed_42"),
            },
            "FLTrust": {
                "CIFAR-10": Path("./results/main_summary_cifar10_cnn_fltrust/run_0_seed_42"),
                "TREC": Path("./results/main_summary_trec_fltrust/run_0_seed_42"),
            },
            "SkyMask": {
                # SkyMask is typically for images, so we only include CIFAR-10
                "CIFAR-10": Path("./results/main_summary_cifar10_cnn_skymask/run_0_seed_42"),
            }
        }
    }

    results = []

    for aggregator in aggregators:
        for dataset in datasets:
            # Skip combinations that don't exist (e.g., SkyMask on TREC)
            if dataset not in experiment_paths["Attacks"][aggregator]:
                continue

            # Get BSR from the corresponding baseline run
            baseline_path = experiment_paths["Baselines"].get(dataset)
            bsr = calculate_selection_rates(baseline_path)['Benign'] if baseline_path else float('nan')

            # Get MSR from the attack run
            attack_path = experiment_paths["Attacks"][aggregator].get(dataset)
            msr = calculate_selection_rates(attack_path)['Malicious'] if attack_path else float('nan')

            results.append({
                'Dataset': dataset,
                'Aggregator': aggregator,
                'BSR (No Attack)': bsr,
                'MSR (Under Attack)': msr
            })

    if not results:
        print("Error: No data was loaded. Please check your paths.")
    else:
        final_df = pd.DataFrame(results)

        # Pivot the table for a cleaner, paper-ready format
        pivot_df = final_df.pivot_table(
            index='Dataset',
            columns='Aggregator',
            values=['BSR (No Attack)', 'MSR (Under Attack)']
        ).sort_index(axis=1, level=1) # Sort by aggregator name

        print("\n--- Cross-Dataset Filtering Performance ---")
        # Format the numbers to 2 decimal places for clarity
        print(pivot_df.to_string(float_format="%.2f"))
        print("-------------------------------------------\n")

        # Save the table to a CSV file
        pivot_df.to_csv("figure_7_cross_dataset_table.csv")
        print("Table saved to figure_7_cross_dataset_table.csv")