import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- 1. CONFIGURATION: SET THESE VALUES ---

# Set the path to your main results directory
# This script will search for 'step10_scalability' subfolders
RESULTS_DIR = "./results"

# The name of the final summary file
SUMMARY_FILENAME = "summary.json"

# --- 2. DATA LOADING ---

def parse_path_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses the directory path to extract defense and n_sellers.

    Example path:
    .../step10_scalability_fltrust_cifar100/n_sellers_50/.../summary.json
    """
    parts = set(file_path.parts)
    metadata = {}

    # Regex to find the defense and n_sellers
    defense_re = re.compile(r"step10_scalability_([a-zA-Z0-9]+)_")
    sellers_re = re.compile(r"n_sellers_(\d+)")

    for part in parts:
        defense_match = defense_re.search(part)
        sellers_match = sellers_re.search(part)

        if defense_match and 'defense' not in metadata:
            metadata['defense'] = defense_match.group(1)

        if sellers_match and 'n_sellers' not in metadata:
            metadata['n_sellers'] = int(sellers_match.group(1))

    # Only return if we found both key pieces of info
    if 'defense' in metadata and 'n_sellers' in metadata:
        return metadata

    return None

def load_scalability_data(results_dir: str) -> pd.DataFrame:
    """
    Loads all 'summary.json' files and extracts metadata from their paths.
    """
    all_results = []
    # Find all summary files within step10 experiment folders
    summary_files = list(Path(results_dir).rglob(f"step10_scalability*/{SUMMARY_FILENAME}"))

    if not summary_files:
        print(f"Error: No '{SUMMARY_FILENAME}' files found under 'step10_scalability' in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(summary_files)} summary files. Loading...")

    for file_path in summary_files:
        # Extract metadata (defense, n_sellers) from the path
        metadata = parse_path_metadata(file_path)
        if not metadata:
            print(f"Warning: Skipping {file_path} - could not parse defense/n_sellers from path.")
            continue

        try:
            # Load the final metrics from the summary file
            with open(file_path, 'r') as f:
                summary_data = json.load(f)

            row = {
                'defense': metadata['defense'],
                'n_sellers': metadata['n_sellers'],
                'test_acc': summary_data.get('test_acc'),
                'backdoor_asr': summary_data.get('backdoor_asr'),
                'seed': summary_data.get('seed'), # For aggregation
            }

            # Ensure we have the data we need
            if row['test_acc'] is None or row['backdoor_asr'] is None:
                print(f"Warning: Skipping {file_path} - 'test_acc' or 'backdoor_asr' missing.")
                continue

            all_results.append(row)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Successfully loaded and parsed {len(all_results)} experiment runs.")
    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    return df

# --- 3. ANALYSIS & PLOTTING FUNCTIONS ---

def plot_scalability_impact(df: pd.DataFrame):
    """
    Plots Test Accuracy and Backdoor ASR as a function of
    the number of sellers for each defense.
    """
    if df.empty:
        print("No data to plot.")
        return

    print("\n--- Plotting Scalability Impact ---")

    # Set the plotting theme
    sns.set_theme(style="whitegrid", palette="colorblind")

    # --- PLOT 1: Test Accuracy ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='n_sellers',
        y='test_acc',
        hue='defense',
        marker='o',
        linewidth=2.5,
        markersize=8
    )
    plt.title('Impact of Marketplace Size on Test Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Total Number of Sellers in Marketplace', fontsize=12)
    plt.ylabel('Final Test Accuracy', fontsize=12)
    plt.legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("analysis_scalability_accuracy.png")
    print("Saved 'analysis_scalability_accuracy.png'")

    # --- PLOT 2: Backdoor ASR ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='n_sellers',
        y='backdoor_asr',
        hue='defense',
        marker='o',
        linewidth=2.5,
        markersize=8
    )
    plt.title('Impact of Marketplace Size on Backdoor ASR', fontsize=16, fontweight='bold')
    plt.xlabel('Total Number of Sellers in Marketplace', fontsize=12)
    plt.ylabel('Final Backdoor ASR', fontsize=12)
    plt.legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("analysis_scalability_asr.png")
    print("Saved 'analysis_scalability_asr.png'")

    print("\n--- Plotting Complete ---")
    print("Insight: Look for defenses (lines) that stay high in the accuracy plot "
          "and low in the ASR plot as the number of sellers increases (moving left to right).")

# --- 4. MAIN EXECUTION ---

def main():
    # 1. Load Data
    main_df = load_scalability_data(RESULTS_DIR)

    if main_df.empty:
        print("No data loaded. Exiting.")
        return

    print("\n--- Data Loaded Successfully ---")
    print(main_df.head())

    # 2. Show a summary table in the console
    # Note: We sort by n_sellers to make the table readable
    summary_table = main_df.groupby(['defense', 'n_sellers'])[['test_acc', 'backdoor_asr']].mean()
    print("\n--- Average Metrics Table ---")
    print(summary_table.to_string(float_format="%.4f"))

    # 3. Run Analysis
    plot_scalability_impact(main_df)

    plt.show() # Display the plots

if __name__ == "__main__":
    main()