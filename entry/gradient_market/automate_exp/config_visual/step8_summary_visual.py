import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_summary_profiles"


# ---------------------

# --- Reuse your existing parsers/loaders ---
def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        # Step 8
        match8 = re.search(r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)', scenario_name)
        if match8:
            return {"type": "attack", "attack": match8.group(1), "defense": match8.group(2), "dataset": match8.group(3)}

        # Step 7 (Baseline)
        match7 = re.search(r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask)_(.*)', scenario_name)
        if match7:
            return {"type": "baseline", "attack": "0. Baseline", "defense": match7.group(1), "dataset": match7.group(2)}

        return {"type": "unknown"}
    except:
        return {"type": "unknown"}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    run_records = []
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0
        base_metrics['acc'] = acc
    except:
        return []

    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists(): return [base_metrics]
        with open(report_file, 'r') as f:
            report = json.load(f)
        sellers = report.get('seller_summaries', {})

        found = False
        for sid, sdata in sellers.items():
            if sdata.get('type') == 'benign':
                found = True
                rec = base_metrics.copy()
                rec['seller_id'] = sid
                rec['selection_rate'] = sdata.get('selection_rate', 0.0)
                run_records.append(rec)
        return run_records if found else [base_metrics]
    except:
        return [base_metrics]


def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)
    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))

    print(f"Found {len(folders)} directories.")
    for folder in folders:
        info = parse_scenario_name(folder.name)
        if info.get("type") == "unknown": continue
        for f in folder.rglob("final_metrics.json"):
            for r in load_run_data(f):
                all_records.append({**info, **r})
    return pd.DataFrame(all_records)


# ==========================================================================
# NEW SUMMARY PLOTTER
# ==========================================================================

def plot_defense_vulnerability_profile(df: pd.DataFrame, output_dir: Path):
    """
    Generates one Summary Figure per Defense/Dataset.
    Shows impact of ALL attacks on Selection (Top) and Accuracy (Bottom)
    relative to the Baseline.
    """
    print("\n--- Generating Defense Vulnerability Profiles ---")

    # Get unique combinations
    combinations = df[['defense', 'dataset']].drop_duplicates().values

    for defense, dataset in combinations:
        print(f"  Processing: {defense} on {dataset}")

        # Filter data
        subset = df[(df['defense'] == defense) & (df['dataset'] == dataset)].copy()

        # 1. Extract Baseline Stats
        baseline_data = subset[subset['attack'] == '0. Baseline']
        if baseline_data.empty:
            print(f"    Skipping {defense} (No Baseline found)")
            continue

        base_sel_mean = baseline_data['selection_rate'].mean()
        base_acc_mean = baseline_data['acc'].mean()

        # 2. Extract Attack Data (Exclude baseline rows from the bars/boxes)
        attack_data = subset[subset['attack'] != '0. Baseline'].copy()
        if attack_data.empty: continue

        # Clean up attack names for X-axis
        # e.g., "oscillating_binary" -> "Oscillating\nBinary"
        def clean_label(s):
            return s.replace('_', '\n').title()

        attack_data['attack_label'] = attack_data['attack'].apply(clean_label)

        # Determine sort order (alphabetical or logical)
        attack_order = sorted(attack_data['attack_label'].unique())

        # --- CREATE COMPOSITE PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Vulnerability Profile: {defense.upper()} ({dataset})", fontsize=16, y=0.95)

        # --- PLOT 1: SELECTION RATE (Fairness/Economy) ---
        # We use a Boxplot to show variance (e.g. Starvation selects some, rejects others)
        sns.boxplot(
            data=attack_data, x='attack_label', y='selection_rate',
            order=attack_order, ax=ax1, palette="viridis", hue='attack_label', legend=False
        )

        # Add Baseline Line
        ax1.axhline(y=base_sel_mean, color='red', linestyle='--', linewidth=2, label='Baseline (No Attack)')
        ax1.text(x=len(attack_order) - 0.5, y=base_sel_mean + 0.02, s="Baseline", color='red', ha='right', fontsize=10)

        ax1.set_ylabel("Selection Rate", fontsize=12)
        ax1.set_title("Economic Impact: Benign Seller Selection Rate", fontsize=12)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # --- PLOT 2: ACCURACY (Utility) ---
        # We use a Barplot (mean with CI)
        # We aggregate first to avoid calculating mean of selection rate rows (duplicates)
        acc_agg = attack_data.drop_duplicates(subset=['attack_label', 'acc'])

        sns.barplot(
            data=acc_agg, x='attack_label', y='acc',
            order=attack_order, ax=ax2, palette="viridis", hue='attack_label', legend=False
        )

        # Add Baseline Line
        ax2.axhline(y=base_acc_mean, color='blue', linestyle='--', linewidth=2, label='Baseline Accuracy')
        ax2.text(x=len(attack_order) - 0.5, y=base_acc_mean + 0.02, s="Baseline Acc", color='blue', ha='right',
                 fontsize=10)

        ax2.set_ylabel("Model Accuracy", fontsize=12)
        ax2.set_title("Utility Impact: Global Model Accuracy", fontsize=12)
        ax2.set_xlabel("Buyer Attack Type", fontsize=12)
        ax2.set_ylim(0, 1.05)  # Acc is 0-1
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        # Add value labels to bars
        for p in ax2.patches:
            if p.get_height() > 0:
                ax2.annotate(f'{p.get_height():.2f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, -12),
                             textcoords='offset points',
                             color='white', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle

        fname = output_dir / f"Summary_Profile_{defense}_{dataset}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fname.name}")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    plot_defense_vulnerability_profile(df, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()