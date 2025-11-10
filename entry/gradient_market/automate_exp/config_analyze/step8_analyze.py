import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- 1. CONFIGURATION: SET THESE VALUES ---

# Set the path to your main results directory
RESULTS_DIR = "./results"

SUMMARY_FILENAME = "summary.json"
ROUNDS_FILENAME = "round_records.jsonl"

# Define the valuation score you want to analyze
PERFORMANCE_SCORE_KEY = 'influence_score'
GROUND_TRUTH_SCORE_KEY = 'marginal_contrib_loo'

# --- 2. DATA LOADING & PARSING ---

def parse_path_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses the directory path to extract defense and attack_type.

    Example path:
    .../step8_buyer_attack_dos_fltrust_CIFAR100/.../summary.json
    """
    parts = set(file_path.parts)
    metadata = {}

    # Regex to find the attack_type and defense
    # Format: step8_buyer_attack_{ATTACK_TAG}_{DEFENSE_NAME}_
    re_pattern = re.compile(r"step8_buyer_attack_([a-zA-Z0-9_]+)_([a-zA-Z0-9]+)_")

    for part in parts:
        match = re_pattern.search(part)
        if match and 'defense' not in metadata:
            metadata['attack_type'] = match.group(1)
            metadata['defense'] = match.group(2)
            break

    if 'defense' in metadata and 'attack_type' in metadata:
        return metadata

    return None

def load_summary_data(results_dir: str) -> pd.DataFrame:
    """
    Loads all 'summary.json' files for the final outcome heatmap.
    """
    all_results = []
    summary_files = list(Path(results_dir).rglob(f"step8_buyer_attack*/{SUMMARY_FILENAME}"))

    if not summary_files:
        print(f"Error: No '{SUMMARY_FILENAME}' files found under 'step8_buyer_attack' in {results_dir}")
        return pd.DataFrame()

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
                'attack_type': metadata['attack_type'],
                'test_acc': summary_data.get('test_acc'),
                'seed': summary_data.get('seed'),
            }
            all_results.append(row)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_results)} summary data points.")
    return pd.DataFrame(all_results)

def load_round_data(results_dir: str) -> pd.DataFrame:
    """
    Loads all 'round_records.jsonl' files for valuation/selection analysis.
    """
    all_round_data = []
    round_files = list(Path(results_dir).rglob(f"step8_buyer_attack*/{ROUNDS_FILENAME}"))

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

                    # 1. Get round-level stats
                    all_round_data.append({
                        'defense': metadata['defense'],
                        'attack_type': metadata['attack_type'],
                        'round': record.get('round'),
                        'seed': metadata.get('seed', 0), # simplified
                        'num_selected': record.get('num_selected'),
                        'metric_type': 'selection'
                    })

                    # 2. Get seller-level valuation stats
                    for seller_data in record.get('detailed_seller_metrics', []):
                        all_round_data.append({
                            'defense': metadata['defense'],
                            'attack_type': metadata['attack_type'],
                            'round': record.get('round'),
                            'seed': metadata.get('seed', 0),
                            'score': seller_data.get(PERFORMANCE_SCORE_KEY),
                            'metric_type': PERFORMANCE_SCORE_KEY
                        })

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_round_data)} round/valuation data points.")
    return pd.DataFrame(all_round_data)

# --- 3. ANALYSIS & PLOTTING FUNCTIONS ---

def plot_final_accuracy_heatmap(df: pd.DataFrame):
    """
    Analysis 1: Final Outcome.
    Plots a heatmap of Test Accuracy vs. (Defense, Attack Type).
    """
    print("\n--- Plotting 1: Final Accuracy Heatmap ---")
    if df.empty:
        print("No summary data to plot.")
        return

    # Aggregate over seeds
    heatmap_df = df.groupby(['defense', 'attack_type'])['test_acc'].mean().reset_index()

    # Pivot for heatmap
    try:
        heatmap_pivot = heatmap_df.pivot(index='defense', columns='attack_type', values='test_acc')
    except Exception as e:
        print(f"Error pivoting data for heatmap: {e}. Check your data.")
        print(heatmap_df)
        return

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag_r", # Red (low) to Blue (high)
        linewidths=.5,
        cbar_kws={'label': 'Final Test Accuracy'}
    )
    plt.title('Final Test Accuracy vs. Defense and Buyer Attack', fontsize=16, fontweight='bold')
    plt.xlabel('Buyer Attack Type', fontsize=12)
    plt.ylabel('Defense', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("analysis_buyer_attack_heatmap.png")
    print("Saved 'analysis_buyer_attack_heatmap.png'")
    print("Insight: A good defense will have high accuracy (e.g., blue) across its entire row. "
          "A devastating attack will have low accuracy (e.g., red) down its entire column.")

def plot_selection_rate(df: pd.DataFrame):
    """
    Analysis 2: Selection / DoS Impact.
    Plots the average number of selected sellers per round.
    """
    print("\n--- Plotting 2: Average Seller Selection Rate ---")
    plot_df = df[df['metric_type'] == 'selection'].copy()

    if plot_df.empty:
        print("No selection data to plot.")
        return

    plot_df['num_selected'] = pd.to_numeric(plot_df['num_selected'])

    # Get average selection count per experiment
    avg_selection_df = plot_df.groupby(['defense', 'attack_type', 'seed'])['num_selected'].mean().reset_index()

    plt.figure(figsize=(16, 8))
    g = sns.catplot(
        data=avg_selection_df,
        x='attack_type',
        y='num_selected',
        col='defense',
        kind='bar',
        col_wrap=2,
        palette='Blues',
        height=5,
        aspect=1.5
    )
    g.fig.suptitle('Impact of Buyer Attacks on Seller Selection (DoS Analysis)', fontsize=16, y=1.03)
    g.set_axis_labels("Buyer Attack Type", "Avg. # Sellers Selected")
    g.set_xticklabels(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("analysis_buyer_attack_selection_rate.png")
    print("Saved 'analysis_buyer_attack_selection_rate.png'")
    print("Insight: This shows Denial-of-Service. A 'dos' attack should cause 'num_selected' to drop to 0. "
          "Robust defenses (like FLTrust) should maintain a stable selection rate.")

def plot_valuation_stability(df: pd.DataFrame):
    """
    Analysis 3: Valuation Manipulation.
    Plots the distribution of seller valuation scores.
    """
    print("\n--- Plotting 3: Valuation Score Stability ---")
    plot_df = df[df['metric_type'] == PERFORMANCE_SCORE_KEY].copy()

    if plot_df.empty:
        print(f"No '{PERFORMANCE_SCORE_KEY}' data to plot.")
        return

    plot_df['score'] = pd.to_numeric(plot_df['score'])

    plt.figure(figsize=(16, 8))
    g = sns.catplot(
        data=plot_df,
        x='attack_type',
        y='score',
        col='defense',
        kind='box',
        col_wrap=2,
        palette='Greens',
        height=5,
        aspect=1.5,
        showfliers=False # Hide outliers for a cleaner plot
    )
    g.fig.suptitle(f'Impact of Buyer Attacks on Seller Valuation ({PERFORMANCE_SCORE_KEY})', fontsize=16, y=1.03)
    g.set_axis_labels("Buyer Attack Type", "Valuation Score Distribution")
    g.set_xticklabels(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("analysis_buyer_attack_valuation_stability.png")
    print("Saved 'analysis_buyer_attack_valuation_stability.png'")
    print(f"Insight: This shows manipulation. Attacks like 'erosion' might drive all scores to 0. "
          f"Attacks like 'oscillating' might make the score distribution (the box) very large and chaotic.")

# --- 4. MAIN EXECUTION ---

def main():
    sns.set_theme(style="whitegrid")

    # --- Part 1: Final Outcome Analysis ---
    summary_df = load_summary_data(RESULTS_DIR)
    if not summary_df.empty:
        plot_final_accuracy_heatmap(summary_df)

    # --- Part 2: Valuation & Selection Analysis ---
    rounds_df = load_round_data(RESULTS_DIR)
    if not rounds_df.empty:
        plot_selection_rate(rounds_df)
        plot_valuation_stability(rounds_df)

    if summary_df.empty and rounds_df.empty:
        print("No data was loaded. Please check your RESULTS_DIR.")
        return

    print("\n--- Analysis Complete ---")
    plt.show()

if __name__ == "__main__":
    main()