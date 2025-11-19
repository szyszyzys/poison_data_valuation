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

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        # Step 8: Buyer Attacks
        match8 = re.search(r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)', scenario_name)
        if match8:
            return {
                "type": "attack",
                "attack": match8.group(1),
                "defense": match8.group(2),
                "dataset": match8.group(3).strip()  # Strip whitespace
            }

        # Step 7: Baseline
        match7 = re.search(r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask)_(.*)', scenario_name)
        if match7:
            return {
                "type": "baseline",
                "attack": "0. Baseline",
                "defense": match7.group(1),
                "dataset": match7.group(2).strip()  # Strip whitespace
            }

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

    # Debug counters
    count_step8 = 0
    count_step7 = 0

    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))

    print(f"Found {len(folders)} total directories.")

    for folder in folders:
        info = parse_scenario_name(folder.name)

        if info.get("type") == "unknown":
            print(f"  [WARN] Could not parse folder: {folder.name}")
            continue

        # Check if metrics exist
        metric_files = list(folder.rglob("final_metrics.json"))
        if not metric_files:
            # Only warn if it's a baseline folder (since that's what's missing)
            if info['type'] == 'baseline':
                print(f"  [WARN] Baseline folder found but NO metrics: {folder.name}")
            continue

        if info['type'] == 'attack': count_step8 += 1
        if info['type'] == 'baseline': count_step7 += 1

        for f in metric_files:
            for r in load_run_data(f):
                all_records.append({**info, **r})

    print(f"\n--- Data Loading Summary ---")
    print(f"  Step 8 (Attacks) folders processed: {count_step8}")
    print(f"  Step 7 (Baseline) folders processed: {count_step7}")

    return pd.DataFrame(all_records)


def plot_defense_vulnerability_profile(df: pd.DataFrame, output_dir: Path):
    print("\n--- Generating Defense Vulnerability Profiles ---")

    # 1. DIAGNOSTIC: Print available Baselines vs Attacks
    print("\n[Diagnostic] Available Data:")
    pivot = df.pivot_table(index=['defense', 'dataset'], columns='type', values='acc', aggfunc='count')
    print(pivot)
    print("-" * 40)

    combinations = df[['defense', 'dataset']].drop_duplicates().values

    for defense, dataset in combinations:
        print(f"  Processing: {defense} on {dataset}")

        subset = df[(df['defense'] == defense) & (df['dataset'] == dataset)].copy()

        # 1. Extract Baseline Stats
        baseline_data = subset[subset['attack'] == '0. Baseline']

        if baseline_data.empty:
            print(f"    Skipping {defense} on {dataset} (No Baseline rows found)")
            # Check if there is a naming mismatch
            possible_baselines = df[(df['type'] == 'baseline') & (df['defense'] == defense)]['dataset'].unique()
            if len(possible_baselines) > 0:
                print(f"    -> Did you mean one of these datasets? {possible_baselines}")
            continue

        base_sel_mean = baseline_data['selection_rate'].mean()
        base_acc_mean = baseline_data['acc'].mean()

        # 2. Extract Attack Data
        attack_data = subset[subset['attack'] != '0. Baseline'].copy()
        if attack_data.empty: continue

        def clean_label(s):
            return s.replace('_', '\n').title()

        attack_data['attack_label'] = attack_data['attack'].apply(clean_label)
        attack_order = sorted(attack_data['attack_label'].unique())

        # --- PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Vulnerability Profile: {defense.upper()} ({dataset})", fontsize=16, y=0.95)

        # Selection Rate
        sns.boxplot(
            data=attack_data, x='attack_label', y='selection_rate',
            order=attack_order, ax=ax1, palette="viridis", hue='attack_label', legend=False
        )
        ax1.axhline(y=base_sel_mean, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax1.text(x=len(attack_order) - 0.5, y=base_sel_mean + 0.02, s=f"Base: {base_sel_mean:.2f}", color='red',
                 ha='right')

        ax1.set_ylabel("Selection Rate", fontsize=12)
        ax1.set_title("Economic Impact", fontsize=12)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # Accuracy
        acc_agg = attack_data.drop_duplicates(subset=['attack_label', 'acc'])
        sns.barplot(
            data=acc_agg, x='attack_label', y='acc',
            order=attack_order, ax=ax2, palette="viridis", hue='attack_label', legend=False
        )
        ax2.axhline(y=base_acc_mean, color='blue', linestyle='--', linewidth=2, label='Baseline')
        ax2.text(x=len(attack_order) - 0.5, y=base_acc_mean + 0.02, s=f"Base: {base_acc_mean:.2f}", color='blue',
                 ha='right')

        ax2.set_ylabel("Model Accuracy", fontsize=12)
        ax2.set_title("Utility Impact", fontsize=12)
        ax2.set_xlabel("Buyer Attack Type", fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        for p in ax2.patches:
            if p.get_height() > 0:
                ax2.annotate(f'{p.get_height():.2f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, -12),
                             textcoords='offset points',
                             color='white', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

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