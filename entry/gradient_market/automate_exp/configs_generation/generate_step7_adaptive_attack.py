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
TARGET_VICTIM_ID = "bn_5"


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses folder names. Handles BOTH Step 8 (Attacks) and Step 7 (Baseline).
    """
    try:
        # 1. Check for Step 8 (Buyer Attacks)
        pattern_step8 = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match8 = re.search(pattern_step8, scenario_name)
        if match8:
            return {
                "scenario": scenario_name,
                "type": "attack",
                "attack": match8.group(1),
                "defense": match8.group(2),
                "dataset": match8.group(3),
            }

        # 2. Check for Step 7 (Baseline - No Attack)
        pattern_step7 = r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask)_(.*)'
        match7 = re.search(pattern_step7, scenario_name)
        if match7:
            return {
                "scenario": scenario_name,
                "type": "baseline",
                "attack": "0. Baseline",  # Label for plotting
                "defense": match7.group(1),
                "dataset": match7.group(2),
            }

        return {"scenario": scenario_name, "type": "unknown"}
    except Exception as e:
        print(f"Warning parsing {scenario_name}: {e}")
        return {"scenario": scenario_name, "type": "unknown"}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    """Loads performance metrics and per-seller selection rates."""
    run_records = []
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        base_metrics['acc'] = acc
        base_metrics['rounds'] = metrics.get('completed_rounds', 0)
    except Exception:
        return []

    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists():
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})

        found_sellers = False
        for seller_id, seller_data in sellers.items():
            # In Step 8 & Step 7, all sellers are benign/honest.
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


def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)

    # Look for BOTH Step 8 and Step 7 folders
    folders = [f for f in base_path.glob("step8_buyer_attack_*") if f.is_dir()]
    folders += [f for f in base_path.glob("step7_baseline_no_attack_*") if f.is_dir()]

    print(f"Found {len(folders)} scenario directories (Step 8 + Step 7).")

    for scenario_path in folders:
        run_info = parse_scenario_name(scenario_path.name)
        if run_info.get("type") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            records = load_run_data(metrics_file)
            for r in records:
                all_records.append({**run_info, **r})

    return pd.DataFrame(all_records)


def get_baseline_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Creates a lookup dict for the baseline selection rate.
    Key: (defense, dataset) -> Value: avg_selection_rate
    """
    baseline_df = df[df['attack'] == '0. Baseline']
    if baseline_df.empty:
        print("⚠️ No Step 7 Baseline data found. Plots will rely only on Step 8 data.")
        return {}

    # Average across seeds and sellers
    lookup = baseline_df.groupby(['defense', 'dataset'])['selection_rate'].mean().to_dict()
    print(f"✅ Baseline calculated for {len(lookup)} configs.")
    return lookup


def plot_buyer_attack_distribution(df: pd.DataFrame, baseline_lookup: Dict, output_dir: Path):
    """
    Fig 1: Selection Rate Distribution (Boxplot) + Baseline Reference (Red Line).
    """
    print("\n--- Plotting Selection Distributions (Fig 1) ---")
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Filter for Step 8 attacks (excluding Pivot and Baseline itself)
    step8_attacks = [a for a in df['attack'].unique()
                     if 'pivot' not in str(a) and a != '0. Baseline']

    for attack in step8_attacks:
        attack_df = df[df['attack'] == attack]
        if attack_df.empty: continue

        dataset = attack_df['dataset'].iloc[0] if 'dataset' in attack_df.columns else "Unknown"

        plt.figure(figsize=(7, 5))

        ax = sns.boxplot(
            data=attack_df,
            x='defense',
            y='selection_rate',
            order=defense_order,
            palette="viridis"
        )

        # --- ADD BASELINE REFERENCE LINE ---
        # This shows "What the selection rate SHOULD be in a healthy market"
        for i, defense in enumerate(defense_order):
            base_val = baseline_lookup.get((defense, dataset))
            if base_val is not None:
                # Draw a red dashed line at the baseline level
                ax.hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4,
                          color='red', linestyle='--', lw=2,
                          label='Healthy Baseline' if i == 0 else "")

        if baseline_lookup:
            plt.legend(loc='best')

        plt.title(f'Impact on Seller Selection\nAttack: {attack}', fontsize=14)
        plt.ylabel("Selection Rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        fname = output_dir / f"Step8_SELECTION_{attack}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def plot_targeted_attack_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Fig 1.5: Targeted Exclusion (Pivot). Compares Victim vs Others.
    """
    print("\n--- Plotting Targeted Breakdown (Fig 1.5) ---")
    pivot_df = df[df['attack'].str.contains("pivot", case=False)].copy()

    if pivot_df.empty: return

    pivot_df['Status'] = pivot_df['seller_id'].apply(
        lambda x: 'Victim (bn_5)' if str(x) == TARGET_VICTIM_ID else 'Other Benign'
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=pivot_df, x='defense', y='selection_rate', hue='Status',
        order=['fedavg', 'fltrust', 'martfl', 'skymask'],
        palette={'Victim (bn_5)': '#e74c3c', 'Other Benign': '#95a5a6'},
        errorbar='sd'
    )

    plt.title("Targeted Exclusion Success (Orthogonal Pivot)", fontsize=14)
    plt.ylabel("Selection Rate")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    fname = output_dir / "Step8_SELECTION_TARGETED_PIVOT.pdf"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname.name}")


def plot_buyer_attack_performance(df: pd.DataFrame, output_dir: Path):
    """
    Fig 2: Performance (Acc/Rounds). Includes '0. Baseline' as the first bar.
    """
    print("\n--- Plotting Performance Metrics (Fig 2) ---")

    # Deduplicate to get one row per experiment seed
    run_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds', 'attack', 'defense'])

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Get attacks and ensure Baseline is first
    attacks = [a for a in run_df['attack'].unique() if a != '0. Baseline']
    attacks.sort()

    # Only loop through actual attacks (we will merge baseline into them)
    baseline_df = run_df[run_df['attack'] == '0. Baseline']

    for attack in attacks:
        current_attack_df = run_df[run_df['attack'] == attack]

        # Combine Baseline + Current Attack for side-by-side comparison
        combined_df = pd.concat([baseline_df, current_attack_df], ignore_index=True)
        if combined_df.empty: continue

        # Melt
        melted = combined_df.melt(
            id_vars=['attack', 'defense'],
            value_vars=['acc', 'rounds'],
            var_name='MetricKey', value_name='Value'
        )
        melted['Metric'] = melted['MetricKey'].map({'acc': 'Accuracy', 'rounds': 'Rounds'})

        # Plot
        g = sns.catplot(
            data=melted, x='defense', y='Value',
            hue='attack',  # Compare Baseline vs Attack
            col='Metric', kind='bar',
            order=defense_order,
            height=4, aspect=1.2, sharey=False,
            palette={'0. Baseline': 'grey', attack: 'red'}  # Baseline Grey, Attack Red
        )

        g.fig.suptitle(f'Marketplace Damage Assessment\nAttack: {attack}', y=1.05)
        g.set_axis_labels("Defense", "Value")

        fname = output_dir / f"Step8_PERFORMANCE_{attack}.pdf"
        g.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots saved to: {output_dir.resolve()}")

    # 1. Load Data (Step 8 + Step 7)
    df = collect_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    # 2. Calculate Baseline for Reference Lines
    baseline_lookup = get_baseline_lookup(df)

    # 3. Generate Plots
    plot_buyer_attack_distribution(df, baseline_lookup, output_dir)
    plot_targeted_attack_breakdown(df, output_dir)
    plot_buyer_attack_performance(df, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()