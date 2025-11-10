import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- 1. CONFIGURATION: SET THESE VALUES ---

# Set the path to your main results directory
# This script will search for all 'round_records.jsonl' files recursively
RESULTS_DIR = "./results"

# Define the seller IDs of your malicious/sybil sellers
# Example: MALICIOUS_SELLER_IDS = {"seller_adv_0", "seller_adv_1", "sybil_0"}
MALICIOUS_SELLER_IDS: Set[str] = {
    # Add your malicious/sybil seller IDs here
    # e.g., "seller_adv_0", "seller_adv_1"
}

# Define the valuation scores you want to analyze
# These MUST match the keys in your 'detailed_seller_metrics'
PROXY_SCORE_KEY = 'sim_to_oracle'
PERFORMANCE_SCORE_KEY = 'influence_score'
GROUND_TRUTH_SCORE_KEY = 'marginal_contrib_loo' # This may only have data on some rounds


# --- 2. DATA LOADING ---

def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Loads all 'round_records.jsonl' files from a results directory.

    This function assumes you save your records as a JSONL file
    (one JSON object per round, per line), which is a common practice.
    """
    all_round_records = []
    jsonl_files = list(Path(results_dir).rglob("round_records.jsonl"))

    if not jsonl_files:
        print(f"Error: No 'round_records.jsonl' files found in {results_dir}")
        return []

    print(f"Found {len(jsonl_files)} result files. Loading...")

    for file_path in jsonl_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            all_round_records.append(record)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping malformed line in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_round_records)} total round records.")
    return all_round_records

def preprocess_data(all_round_records: List[Dict[str, Any]], malicious_ids: Set[str]) -> pd.DataFrame:
    """
    Converts the raw list of round records into a clean, flat pandas DataFrame
    ready for analysis.
    """
    processed_data = []

    for round_record in all_round_records:
        round_num = round_record.get('round')
        selected_ids = set(round_record.get('selected_ids', []))
        outlier_ids = set(round_record.get('outlier_ids', []))

        seller_metrics_list = round_record.get('detailed_seller_metrics', [])

        for seller_data in seller_metrics_list:
            seller_id = seller_data.get('seller_id')
            if not seller_id:
                continue

            # Determine selection status
            if seller_id in selected_ids:
                selection_status = "Selected"
            elif seller_id in outlier_ids:
                selection_status = "Rejected"
            else:
                selection_status = "Not Participating" # Or "Not Selected"

            # Determine seller type
            seller_type = "Malicious" if seller_id in malicious_ids else "Benign"

            # Create a flat row
            row = {
                'round': round_num,
                'seller_id': seller_id,
                'selection_status': selection_status,
                'seller_type': seller_type,
                # Add all other scores from the record
                PROXY_SCORE_KEY: seller_data.get(PROXY_SCORE_KEY),
                PERFORMANCE_SCORE_KEY: seller_data.get(PERFORMANCE_SCORE_KEY),
                GROUND_TRUTH_SCORE_KEY: seller_data.get(GROUND_TRUTH_SCORE_KEY),
                # You can add more keys here
                'sim_to_agg': seller_data.get('sim_to_agg'),
                'l2_norm': seller_data.get('l2_norm'),
            }
            processed_data.append(row)

    df = pd.DataFrame(processed_data)

    # Convert types for plotting
    df['round'] = pd.to_numeric(df['round'])
    df[PROXY_SCORE_KEY] = pd.to_numeric(df[PROXY_SCORE_KEY])
    df[PERFORMANCE_SCORE_KEY] = pd.to_numeric(df[PERFORMANCE_SCORE_KEY])
    df[GROUND_TRUTH_SCORE_KEY] = pd.to_numeric(df[GROUND_TRUTH_SCORE_KEY])

    print(f"Pre-processing complete. Created DataFrame with {len(df)} rows.")
    return df

# --- 3. ANALYSIS & PLOTTING FUNCTIONS ---

def plot_aggregator_rationality(df: pd.DataFrame, score_to_plot: str):
    """
    Analysis 1: Is the aggregator rational?
    Compares the valuation score of selected vs. rejected sellers.
    """
    print(f"\n--- Analysis 1: Aggregator Rationality (using {score_to_plot}) ---")
    plt.figure(figsize=(10, 6))

    # Filter to only sellers who were either selected or rejected
    plot_df = df[df['selection_status'].isin(["Selected", "Rejected"])]

    if plot_df.empty:
        print("Skipping plot: No 'Selected' or 'Rejected' sellers found.")
        return

    sns.boxplot(
        data=plot_df,
        x='selection_status',
        y=score_to_plot,
        palette={'Selected': 'g', 'Rejected': 'r'}
    )

    plt.title(f'Aggregator Rationality: {score_to_plot} of Selected vs. Rejected Sellers', fontsize=16)
    plt.xlabel('Selection Status', fontsize=12)
    plt.ylabel(f'Valuation Score ({score_to_plot})', fontsize=12)
    plt.tight_layout()
    plt.savefig("analysis_1_aggregator_rationality.png")
    print("Saved 'analysis_1_aggregator_rationality.png'")
    print("Insight: A good aggregator should select sellers with a higher score. "
          "The 'Selected' box should be significantly higher than 'Rejected'.")

def plot_valuation_correlation(df: pd.DataFrame):
    """
    Analysis 2: Are the valuation methods good?
    Compares the cheap proxy score vs. the expensive "ground truth" score.
    """
    print(f"\n--- Analysis 2: Valuation Method Correlation ---")

    # Filter to only rounds where the expensive (LOO) score was calculated
    plot_df = df.dropna(subset=[PROXY_SCORE_KEY, GROUND_TRUTH_SCORE_KEY])

    if plot_df.empty:
        print(f"Skipping plot: No data found with both '{PROXY_SCORE_KEY}' and '{GROUND_TRUTH_SCORE_KEY}'.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x=PROXY_SCORE_KEY,
        y=GROUND_TRUTH_SCORE_KEY,
        hue='seller_type',
        palette={'Benign': 'blue', 'Malicious': 'red'},
        alpha=0.6
    )

    plt.title(f'Correlation: Proxy ({PROXY_SCORE_KEY}) vs. Ground Truth ({GROUND_TRUTH_SCORE_KEY})', fontsize=16)
    plt.xlabel(f'Proxy Score ({PROXY_SCORE_KEY})', fontsize=12)
    plt.ylabel(f'Ground Truth Score ({GROUND_TRUTH_SCORE_KEY})', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("analysis_2_valuation_correlation.png")
    print("Saved 'analysis_2_valuation_correlation.png'")
    print(f"Insight: If '{PROXY_SCORE_KEY}' is a good proxy, you should see a "
          "clear positive correlation (dots go from bottom-left to top-right).")

def plot_attacker_detection_timeseries(df: pd.DataFrame, score_to_plot: str):
    """
    Analysis 3: Can the valuation metrics detect attackers?
    Plots the average score of benign vs. malicious sellers over time.
    """
    print(f"\n--- Analysis 3: Attacker Detection (using {score_to_plot}) ---")
    plt.figure(figsize=(12, 6))

    if df.empty or df[score_to_plot].isnull().all():
        print(f"Skipping plot: No data found for '{score_to_plot}'.")
        return

    # Calculate the average score per round for each seller type
    avg_df = df.groupby(['round', 'seller_type'])[score_to_plot].mean().reset_index()

    sns.lineplot(
        data=avg_df,
        x='round',
        y=score_to_plot,
        hue='seller_type',
        palette={'Benign': 'blue', 'Malicious': 'red'},
        marker='o'
    )

    plt.title(f'Attacker Detection: Average {score_to_plot} Over Time', fontsize=16)
    plt.xlabel('Round Number', fontsize=12)
    plt.ylabel(f'Average Score ({score_to_plot})', fontsize=12)
    plt.legend(title='Seller Type')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("analysis_3_attacker_detection_timeseries.png")
    print("Saved 'analysis_3_attacker_detection_timeseries.png'")
    print("Insight: A good valuation metric should clearly separate the two lines. "
          "The 'Benign' line should be high, and 'Malicious' should be low.")

def plot_sybil_effectiveness(df: pd.DataFrame):
    """
    Analysis 4: How effective is the Sybil attack?
    Compares the sybils' proxy score (deception) vs. their ground truth score (impact).
    """
    print(f"\n--- Analysis 4: Sybil Attack Effectiveness ---")

    # Filter to *only* malicious sellers and *only* rounds with ground truth data
    plot_df = df[
        (df['seller_type'] == 'Malicious')
    ].dropna(subset=[PROXY_SCORE_KEY, GROUND_TRUTH_SCORE_KEY])

    if plot_df.empty:
        print(f"Skipping plot: No Malicious sellers found with both '{PROXY_SCORE_KEY}' and '{GROUND_TRUTH_SCORE_KEY}'.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x=PROXY_SCORE_KEY,
        y=GROUND_TRUTH_SCORE_KEY,
        color='red',
        alpha=0.7,
        s=80 # larger dots
    )

    plt.title('Sybil Attack Effectiveness: Deception vs. True Impact', fontsize=16)
    plt.xlabel(f'Deception (Proxy Score: {PROXY_SCORE_KEY})', fontsize=12)
    plt.ylabel(f'True Impact (Ground Truth Score: {GROUND_TRUTH_SCORE_KEY})', fontsize=12)

    # Add lines to show quadrants
    plt.axhline(0, color='black', linestyle='--', lw=1)
    plt.axvline(plot_df[PROXY_SCORE_KEY].mean(), color='grey', linestyle='--', lw=1, label=f'Avg. Proxy Score')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("analysis_4_sybil_effectiveness.png")
    print("Saved 'analysis_4_sybil_effectiveness.png'")
    print("Insight: The 'perfect' sybil attack would be in the top-right quadrant: "
          "high deception score (fooled the proxy) and low/negative impact score (hurt the model).")

# --- 4. MAIN EXECUTION ---

def main():
    # Set plotting style
    sns.set_theme(style="whitegrid")

    # 1. Load Data
    all_records = load_all_results(RESULTS_DIR)
    if not all_records:
        print("No data loaded. Exiting.")
        return

    # 2. Pre-process Data
    main_df = preprocess_data(all_records, MALICIOUS_SELLER_IDS)
    if main_df.empty:
        print("No data to analyze after pre-processing. Exiting.")
        return

    print("\nDataFrame sample:")
    print(main_df.head())

    # 3. Run Analyses

    # Analysis 1: Use the performance score (e.g., influence)
    plot_aggregator_rationality(main_df, score_to_plot=PERFORMANCE_SCORE_KEY)

    # Analysis 2:
    plot_valuation_correlation(main_df)

    # Analysis 3: Use the performance score (e.g., influence)
    plot_attacker_detection_timeseries(main_df, score_to_plot=PERFORMANCE_SCORE_KEY)

    # Analysis 4:
    plot_sybil_effectiveness(main_df)

    print("\n--- Analysis Complete ---")
    print("All plots saved to the current directory.")
    plt.show() # Display the plots

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()