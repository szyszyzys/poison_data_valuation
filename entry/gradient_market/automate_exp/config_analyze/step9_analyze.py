import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- 1. CONFIGURATION: SET THESE VALUES ---

# Set the path to your main results directory
RESULTS_DIR = "./results"
SUMMARY_FILENAME = "summary.json"
ROUNDS_FILENAME = "round_records.jsonl"

# --- DEFINE YOUR SELLER GROUND TRUTH ---
# This is the benign seller being copied
TARGET_SELLER_ID = "bn_0"

# Define the seller IDs of your malicious/sybil sellers
# This script will label any seller_id starting with 'adv' as a mimic.
# If your mimics have different names, list them here.
MIMICKING_SELLER_IDS: Set[str] = {
    # e.g., "adv_seller_0", "adv_seller_1"
    # This is often empty if you use the 'adv_' prefix
}

# Define the valuation scores you want to analyze
PERFORMANCE_SCORE_KEY = 'influence_score'
GROUND_TRUTH_SCORE_KEY = 'marginal_contrib_loo'


# --- 2. DATA LOADING & PARSING ---

def parse_path_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses the directory path to extract defense and adv_rate.

    Example path:
    .../step9_comp_mimicry_noisy_copy_fltrust_CIFAR100/adv_rate_0.1/.../summary.json
    """
    parts = set(file_path.parts)
    metadata = {}

    # Regex to find the defense and adv_rate
    defense_re = re.compile(r"step9_comp_mimicry_noisy_copy_([a-zA-Z0-9]+)_")
    adv_rate_re = re.compile(r"adv_rate_([0-9.]+)")

    for part in parts:
        defense_match = defense_re.search(part)
        adv_rate_match = adv_rate_re.search(part)

        if defense_match and 'defense' not in metadata:
            metadata['defense'] = defense_match.group(1)

        if adv_rate_match and 'adv_rate' not in metadata:
            metadata['adv_rate'] = float(adv_rate_match.group(1))

    if 'defense' in metadata and 'adv_rate' in metadata:
        return metadata

    return None

def load_final_summary_data(results_dir: str) -> pd.DataFrame:
    """
    Loads all 'summary.json' files for the final outcome plot.
    """
    all_results = []
    summary_files = list(Path(results_dir).rglob(f"step9_comp_mimicry*/{SUMMARY_FILENAME}"))

    print(f"Found {len(summary_files)} summary files. Loading...")

    for file_path in summary_files:
        metadata = parse_path_metadata(file_path)
        if not metadata:
            print(f"Warning: Skipping {file_path} - could not parse metadata.")
            continue

        try:
            with open(file_path, 'r') as f:
                summary_data = json.load(f)

            row = {
                'defense': metadata['defense'],
                'adv_rate': metadata['adv_rate'],
                'test_acc': summary_data.get('test_acc'),
                'seed': summary_data.get('seed'),
            }
            all_results.append(row)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_results)} summary data points.")
    return pd.DataFrame(all_results)

def get_seller_type(seller_id: str) -> str:
    """Helper to categorize sellers for analysis."""
    if seller_id == TARGET_SELLER_ID:
        return "Target (Benign)"
    if seller_id.startswith("adv_") or seller_id in MIMICKING_SELLER_IDS:
        return "Mimic (Malicious)"
    return "Other (Benign)"

def load_round_valuation_data(results_dir: str) -> pd.DataFrame:
    """
    Loads all 'round_records.jsonl' files for valuation analysis.
    """
    all_round_data = []
    round_files = list(Path(results_dir).rglob(f"step9_comp_mimicry*/{ROUNDS_FILENAME}"))

    print(f"Found {len(round_files)} round record files. Loading...")

    for file_path in round_files:
        metadata = parse_path_metadata(file_path)
        if not metadata:
            print(f"Warning: Skipping {file_path} - could not parse metadata.")
            continue

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    selected_ids = set(record.get('selected_ids', []))

                    for seller_data in record.get('detailed_seller_metrics', []):
                        seller_id = seller_data.get('seller_id')
                        if not seller_id:
                            continue

                        row = {
                            'defense': metadata['defense'],
                            'adv_rate': metadata['adv_rate'],
                            'round': record.get('round'),
                            'seller_id': seller_id,
                            'seller_type': get_seller_type(seller_id),
                            'was_selected': seller_id in selected_ids,
                            PERFORMANCE_SCORE_KEY: seller_data.get(PERFORMANCE_SCORE_KEY),
                            GROUND_TRUTH_SCORE_KEY: seller_data.get(GROUND_TRUTH_SCORE_KEY),
                        }
                        all_round_data.append(row)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_round_data)} detailed seller-round records.")
    df = pd.DataFrame(all_round_data)

    # Convert types
    for col in [PERFORMANCE_SCORE_KEY, GROUND_TRUTH_SCORE_KEY, 'adv_rate', 'round']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    return df

# --- 3. ANALYSIS & PLOTTING FUNCTIONS ---

def plot_final_outcomes(df: pd.DataFrame):
    """
    Plots the final test accuracy as the number of mimics increases.
    """
    print("\n--- Plotting 1: Final Outcome (Test Accuracy vs. # Mimics) ---")
    if df.empty:
        print("No summary data to plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='adv_rate',
        y='test_acc',
        hue='defense',
        marker='o',
        linewidth=2.5,
        markersize=8
    )
    plt.title('Attack Impact: Test Accuracy vs. Proportion of Mimics', fontsize=16, fontweight='bold')
    plt.xlabel('Proportion of Mimicking Sellers (adv_rate)', fontsize=12)
    plt.ylabel('Final Test Accuracy', fontsize=12)
    plt.legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("analysis_mimicry_accuracy.png")
    print("Saved 'analysis_mimicry_accuracy.png'")
    print("Insight: This shows which defenses are robust. A good defense line stays flat and high.")

def plot_valuation_comparison(df: pd.DataFrame, score_to_plot: str):
    """
    Plots the valuation scores for the Target, Mimics, and Other Benign sellers.
    This is the most important plot.
    """
    print(f"\n--- Plotting 2: Valuation Analysis (using {score_to_plot}) ---")
    plot_df = df.dropna(subset=[score_to_plot])
    if plot_df.empty:
        print(f"No data found for score '{score_to_plot}'. Skipping plot.")
        return

    g = sns.catplot(
        data=plot_df,
        x='seller_type',
        y=score_to_plot,
        col='defense',
        kind='box',
        order=['Target (Benign)', 'Mimic (Malicious)', 'Other (Benign)'],
        palette={'Target (Benign)': 'blue', 'Mimic (Malicious)': 'red', 'Other (Benign)': 'grey'}
    )
    g.fig.suptitle(f'Valuation Metric ({score_to_plot}) vs. Seller Type (Faceted by Defense)', fontsize=16, y=1.03)
    g.set_axis_labels("Seller Type", f"Valuation Score ({score_to_plot})")
    plt.tight_layout()
    plt.savefig(f"analysis_mimicry_valuation_{score_to_plot}.png")
    print(f"Saved 'analysis_mimicry_valuation_{score_to_plot}.png'")
    print("Insight: This shows if the valuation metric is fooled. "
          "If the 'Target' and 'Mimic' boxes look identical, the metric is fooled. "
          "A good metric would show a high score for 'Target' and a low score for 'Mimic'.")

def plot_selection_rates(df: pd.DataFrame):
    """
    Plots the % of rounds each seller type was selected.
    """
    print("\n--- Plotting 3: Selection Rate Analysis ---")
    if df.empty:
        print("No round data to plot.")
        return

    # Calculate selection rate per group
    selection_df = df.groupby(['defense', 'seller_type'])['was_selected'].mean().reset_index()
    selection_df.rename(columns={'was_selected': 'selection_rate'}, inplace=True)

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=selection_df,
        x='defense',
        y='selection_rate',
        hue='seller_type',
        palette={'Target (Benign)': 'blue', 'Mimic (Malicious)': 'red', 'Other (Benign)': 'grey'}
    )

    plt.title('Selection Rate by Defense and Seller Type', fontsize=16, fontweight='bold')
    plt.xlabel('Defense', fontsize=12)
    plt.ylabel('Average Selection Rate', fontsize=12)
    plt.legend(title='Seller Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("analysis_mimicry_selection_rate.png")
    print("Saved 'analysis_mimicry_selection_rate.png'")
    print("Insight: This shows if the *aggregator* is fooled. "
          "A robust defense (like FLTrust) should select the 'Target' but reject the 'Mimic'. "
          "A naive defense (like FedAvg) will likely select both.")

# --- 4. MAIN EXECUTION ---

def main():
    sns.set_theme(style="whitegrid", palette="colorblind")

    # --- Part 1: Final Outcome Analysis ---
    summary_df = load_final_summary_data(RESULTS_DIR)
    if not summary_df.empty:
        plot_final_outcomes(summary_df)

    # --- Part 2: Valuation & Selection Analysis ---
    rounds_df = load_round_valuation_data(RESULTS_DIR)
    if not rounds_df.empty:
        plot_valuation_comparison(rounds_df, score_to_plot=PERFORMANCE_SCORE_KEY)
        plot_selection_rates(rounds_df)

    if summary_df.empty and rounds_df.empty:
        print("No data was loaded. Please check your RESULTS_DIR.")
        return

    print("\n--- Analysis Complete ---")
    plt.show()

if __name__ == "__main__":
    main()