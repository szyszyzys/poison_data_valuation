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
TARGET_VICTIM_ID = "bn_5"  # The seller the attacker tries to exclude

# --- Styling ---
DEFENSE_PALETTE = {
    "fedavg": "#3498db",  # Blue
    "fltrust": "#e74c3c",  # Red
    "martfl": "#2ecc71",  # Green
    "skymask": "#9b59b6"  # Purple
}


def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        if 'step8_buyer_attack' in scenario_name:
            pattern = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
            match = re.search(pattern, scenario_name)
            if match:
                return {
                    "type": "attack",
                    "attack": match.group(1),
                    "defense": match.group(2),
                    "dataset": match.group(3),
                }
        elif 'step7_baseline_no_attack' in scenario_name:
            pattern = r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask)_(.*)'
            match = re.search(pattern, scenario_name)
            if match:
                return {
                    "type": "baseline",
                    "attack": "0. Baseline",
                    "defense": match.group(1),
                    "dataset": match.group(2),
                }
        return {"type": "unknown"}
    except:
        return {"type": "unknown"}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0
        base = {'acc': acc * 100, 'rounds': metrics.get('completed_rounds', 0)}

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            records = []
            for sid, sdata in report.get('seller_summaries', {}).items():
                if sdata.get('type') == 'benign':
                    rec = base.copy()
                    rec['seller_id'] = sid
                    rec['selection_rate'] = sdata.get('selection_rate', 0.0)
                    records.append(rec)
            return records if records else [base]

        return [base]
    except:
        return []


def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)

    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))
    print(f"Found {len(folders)} folders.")

    for folder in folders:
        meta = parse_scenario_name(folder.name)
        if meta['type'] == 'unknown': continue

        for mfile in folder.rglob("final_metrics.json"):
            data = load_run_data(mfile)
            for d in data:
                all_records.append({**meta, **d})

    return pd.DataFrame(all_records)


def get_baseline_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    base_df = df[df['attack'] == '0. Baseline']
    if base_df.empty: return {}
    # Get mean selection rate of the Victim in the baseline case
    victim_base = base_df[base_df['seller_id'] == TARGET_VICTIM_ID]
    if victim_base.empty:
        # Fallback to general average if victim specific logic fails
        return base_df.groupby(['defense', 'dataset'])['selection_rate'].mean().to_dict()
    return victim_base.groupby(['defense', 'dataset'])['selection_rate'].mean().to_dict()


# =============================================================================
# FIGURE 1: ATTACK COMPARISON SUMMARY (New Function)
# =============================================================================
def plot_all_attacks_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Creates a grouped bar chart:
    X-axis: Defense
    Bars (Hue): Attack Type (Lie, Reverse, Pivot, etc.)
    Y-axis: Selection Rate of the VICTIM (Lower is better for attacker)
    """
    print("\n--- Generating All-Attack Comparison Figure ---")
    set_plot_style()

    # Filter to only the Victim's data (since the goal is targeted exclusion)
    victim_df = df[df['seller_id'] == TARGET_VICTIM_ID].copy()

    if victim_df.empty:
        print("No data found for the specific victim ID. Plotting average of all benign instead.")
        victim_df = df.copy()  # Fallback

    # Remove baseline for this plot (we want to compare attacks)
    plot_df = victim_df[victim_df['attack'] != '0. Baseline'].copy()

    if plot_df.empty:
        print("No attack data found.")
        return

    # Clean up Attack Names for Legend
    plot_df['Attack Type'] = plot_df['attack'].apply(lambda x: x.replace('_', ' ').title())

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=plot_df,
        x='defense',
        y='selection_rate',
        hue='Attack Type',
        order=defense_order,
        palette='magma',  # Good contrast for multiple categories
        edgecolor='black',
        errorbar=('ci', 95)
    )

    # Add a line for "Ideal Safety" (Baseline ~1.0)
    plt.axhline(1.0, color='green', linestyle='--', linewidth=2, label="Ideal Safety (100%)")

    plt.title("Attack Effectiveness Comparison: Targeted Exclusion of Victim", fontsize=16, fontweight='bold')
    plt.ylabel("Victim Selection Rate (Lower = Attack Success)", fontsize=14)
    plt.xlabel("Defense Mechanism", fontsize=14)
    plt.ylim(0, 1.1)

    # Format X Axis
    ax = plt.gca()
    labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                          "SkyMask").replace(
        "Martfl", "MARTFL") for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')

    plt.legend(title="Attack Strategy", bbox_to_anchor=(1.02, 1), loc='upper left')

    fname = output_dir / "Step8_All_Attacks_Comparison.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved Comparison: {fname.name}")
    plt.close()


# =============================================================================
# FIGURE 2: DEEP DIVE (Your original requested figure)
# =============================================================================
def plot_detailed_pivot_breakdown(df: pd.DataFrame, baseline_lookup: Dict, output_dir: Path):
    print("\n--- Generating Detailed Deep Dive (Pivot) ---")
    set_plot_style()

    # Filter for Pivot Attack
    pivot_attacks = [a for a in df['attack'].unique() if 'pivot' in str(a).lower()]
    if not pivot_attacks: return

    attack_name = pivot_attacks[0]  # Use the first pivot attack found
    attack_df = df[df['attack'] == attack_name].copy()
    baseline_df = df[df['attack'] == '0. Baseline'].copy()

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    fig, axes = plt.subplots(1, 3, figsize=(22, 5), constrained_layout=True)

    # (a) Selection Distribution
    sns.boxplot(ax=axes[0], data=attack_df, x='defense', y='selection_rate', order=defense_order,
                palette=DEFENSE_PALETTE)
    for i, d in enumerate(defense_order):
        ds = attack_df['dataset'].iloc[0] if not attack_df.empty else "Unknown"
        base_val = baseline_lookup.get((d, ds))
        if base_val: axes[0].hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4, color='red', linestyle='--', lw=2.5)
    axes[0].set_title("(a) Selection Rate Variance", fontweight='bold')
    axes[0].set_ylabel("Selection Rate")

    # (b) Victim Isolation
    attack_df['Status'] = attack_df['seller_id'].apply(lambda x: 'Victim' if str(x) == TARGET_VICTIM_ID else 'Others')
    sns.barplot(ax=axes[1], data=attack_df, x='defense', y='selection_rate', hue='Status', order=defense_order,
                palette={'Victim': '#e74c3c', 'Others': '#95a5a6'})
    axes[1].set_title("(b) Victim Isolation (Targeted)", fontweight='bold')
    axes[1].set_ylabel("Selection Rate")

    # (c) Performance
    perf_df = pd.concat([baseline_df, attack_df]).drop_duplicates(subset=['defense', 'attack', 'acc'])
    perf_df['Condition'] = perf_df['attack'].apply(lambda x: 'Baseline' if 'Baseline' in x else 'Under Attack')
    sns.barplot(ax=axes[2], data=perf_df, x='defense', y='acc', hue='Condition', order=defense_order,
                palette={'Baseline': 'gray', 'Under Attack': '#e74c3c'})
    axes[2].set_title("(c) Utility Cost (Accuracy)", fontweight='bold')
    axes[2].set_ylabel("Accuracy (%)")

    # Cleanup
    for ax in axes:
        labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                              "SkyMask").replace(
            "Martfl", "MARTFL") for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fname = output_dir / "Step8_Detailed_Pivot_Breakdown.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved Deep Dive: {fname.name}")
    plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    baseline_lookup = get_baseline_lookup(df)

    # 1. Plot the Summary (All Attacks)
    plot_all_attacks_comparison(df, output_dir)

    # 2. Plot the Deep Dive (Pivot only)
    plot_detailed_pivot_breakdown(df, baseline_lookup, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()