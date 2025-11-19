import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_figures"
TARGET_VICTIM_ID = "bn_5"  # Used for the Targeted Attack Plot


# --- End Configuration ---


def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """Parses folder names like: step8_buyer_attack_dos_martfl_CIFAR100"""
    try:
        pattern = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "attack": match.group(1),
                "defense": match.group(2),
                "dataset": match.group(3),
            }
        return {"scenario": scenario_name}
    except Exception as e:
        print(f"Warning parsing {scenario_name}: {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    """
    Loads performance metrics AND per-seller selection rates.
    Returns a list of rows (one per benign seller) for detailed analysis.
    """
    run_records = []
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Normalize Accuracy (0-1 range)
        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        base_metrics['acc'] = acc
        base_metrics['rounds'] = metrics.get('completed_rounds', 0)
    except Exception:
        return []

    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists():
            # If no report, return one record with just global metrics (no seller info)
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})

        found_sellers = False
        for seller_id, seller_data in sellers.items():
            # In Step 8, ALL sellers are benign (adv_rate=0),
            # but we verify type just in case.
            if seller_data.get('type') == 'benign':
                found_sellers = True
                record = base_metrics.copy()
                record['seller_id'] = seller_id
                record['selection_rate'] = seller_data.get('selection_rate', 0.0)
                run_records.append(record)

        if not found_sellers:
            return [base_metrics]

        return run_records
    except Exception:
        return [base_metrics]


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_seller_runs = []
    base_path = Path(base_dir)

    # Find all Step 8 folders
    scenario_folders = [f for f in base_path.glob("step8_buyer_attack_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario directories.")

    for scenario_path in scenario_folders:
        run_scenario_base = parse_scenario_name(scenario_path.name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            per_seller_records = load_run_data(metrics_file)
            for r in per_seller_records:
                all_seller_runs.append({**run_scenario_base, **r})

    return pd.DataFrame(all_seller_runs)


def plot_buyer_attack_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Fig 1: Boxplots of Selection Rate Distribution.
    Shows how specific attacks change the distribution of wealth/selection.
    """
    print("\n--- Plotting Selection Rate Distributions (Fig 1) ---")
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Exclude 'pivot' because it's a targeted attack (outliers matter more than distribution)
    generic_attacks = [a for a in df['attack'].unique() if 'pivot' not in str(a)]

    for attack in generic_attacks:
        attack_df = df[df['attack'] == attack]
        if attack_df.empty: continue

        # Ensure dataset exists
        dataset = attack_df['dataset'].iloc[0] if 'dataset' in attack_df.columns else "Unknown"

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            data=attack_df,
            x='defense',
            y='selection_rate',
            order=defense_order,
            palette="viridis"
        )

        plt.title(f'Selection Rate Distribution\nAttack: {attack}', fontsize=14)
        plt.ylabel("Selection Rate (0.0 - 1.0)")
        plt.ylim(-0.05, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        fname = output_dir / f"Step8_SELECTION_DIST_{attack}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def plot_targeted_attack_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Fig 1.5: Specialized plot for Targeted Exclusion (Orthogonal Pivot).
    Compares Victim (bn_5) vs Average Other.
    """
    print("\n--- Plotting Targeted Attack Breakdown (Fig 1.5) ---")

    # Filter for pivot attacks
    pivot_df = df[df['attack'].str.contains("pivot", case=False)].copy()

    if pivot_df.empty:
        print("  No 'pivot' attacks found. Skipping.")
        return

    # Categorize Sellers
    pivot_df['Status'] = pivot_df['seller_id'].apply(
        lambda x: 'Victim (bn_5)' if str(x) == TARGET_VICTIM_ID else 'Other Benign'
    )

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=pivot_df,
        x='defense',
        y='selection_rate',
        hue='Status',
        order=['fedavg', 'fltrust', 'martfl', 'skymask'],
        palette={'Victim (bn_5)': '#e74c3c', 'Other Benign': '#95a5a6'},  # Red vs Grey
        errorbar='sd'
    )

    plt.title("Targeted Exclusion Success (Orthogonal Pivot)\n(Goal: Red Bar -> 0.0)", fontsize=14)
    plt.ylabel("Selection Rate")
    plt.ylim(0, 1.05)
    plt.legend(title=None)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    fname = output_dir / "Step8_SELECTION_TARGETED_PIVOT.pdf"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname.name}")


def plot_buyer_attack_performance(df: pd.DataFrame, output_dir: Path):
    """
    Fig 2: Global Model Performance (Accuracy & Rounds).
    """
    print("\n--- Plotting Performance Metrics (Fig 2) ---")

    # Deduplicate: We only need one row per 'run' (seed), not per seller
    # Grouping by these keys ensures we get unique experiment results
    run_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds', 'attack', 'defense'])

    metrics = [('acc', 'Global Accuracy'), ('rounds', 'Rounds to Converge')]
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Loop through attacks to make separate plots
    attacks = sorted(run_df['attack'].unique())

    for attack in attacks:
        attack_df = run_df[run_df['attack'] == attack]
        if attack_df.empty: continue

        # Melt for FacetGrid (Accuracy vs Rounds)
        melted = attack_df.melt(
            id_vars=['defense'],
            value_vars=['acc', 'rounds'],
            var_name='MetricKey',
            value_name='Value'
        )

        # Rename metrics for display
        melted['Metric'] = melted['MetricKey'].map({'acc': 'Accuracy', 'rounds': 'Rounds'})

        g = sns.catplot(
            data=melted,
            x='defense',
            y='Value',
            col='Metric',
            kind='bar',
            order=defense_order,
            palette='viridis',
            height=4,
            aspect=1.2,
            sharey=False
        )

        g.fig.suptitle(f'Global Model Performance\nAttack: {attack}', y=1.05)
        g.set_axis_labels("Defense", "Value")
        g.set_titles("{col_name}")

        fname = output_dir / f"Step8_PERFORMANCE_{attack}.pdf"
        g.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Load Step 8 Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No Step 8 data found. Check your results directory.")
        return

    # 2. Save Summary CSV
    df.to_csv(output_dir / "step8_full_summary.csv", index=False)

    # 3. Generate Plots (No External Baseline)
    plot_buyer_attack_distribution(df, output_dir)
    plot_targeted_attack_breakdown(df, output_dir)
    plot_buyer_attack_performance(df, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()