import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8"
# >>> CHOOSE YOUR DEEP DIVE ATTACK HERE <<<
DEEP_DIVE_ATTACK = "Label Flip"

# --- Naming & Colors ---
PRETTY_NAMES = {
    "fedavg": "FedAvg", "fltrust": "FLTrust", "martfl": "MARTFL",
    "skymask": "SkyMask", "skymask_small": "SkyMask",
    "min_max": "Min-Max", "min_sum": "Min-Sum",
    "labelflip": "Label Flip", "label_flip": "Label Flip",
    "fang_krum": "Fang-Krum", "fang_trim": "Fang-Trim",
    "scaling": "Scaling", "dba": "DBA", "badnet": "BadNet",
    "pivot": "Pivot", "0. Baseline": "Baseline"
}

DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d", "FLTrust": "#3498db",
    "MARTFL": "#2ecc71", "SkyMask": "#e74c3c",
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.3)
    plt.rcParams.update({
        'font.weight': 'bold', 'axes.labelweight': 'bold',
        'axes.titleweight': 'bold', 'axes.titlesize': 16,
        'lines.linewidth': 2,
    })

# ==========================================
# 2. DATA LOADING (Standardized)
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    pattern = r'(step[78])_(baseline_no_attack|buyer_attack)_(?:(.+?)_)?(fedavg|martfl|fltrust|skymask|skymask_small)_(.*)'
    match = re.search(pattern, scenario_name)
    if match:
        _, mode, attack_raw, defense, dataset = match.groups()
        attack_name = "0. Baseline" if "baseline" in mode else attack_raw
        return {"type": "baseline" if "baseline" in mode else "attack",
                "attack": format_label(attack_name),
                "defense": format_label(defense),
                "dataset": dataset}
    return {"type": "unknown"}

def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    try:
        report_file = metrics_file.parent / "marketplace_report.json"
        if not report_file.exists(): return []
        with open(report_file, 'r') as f: report = json.load(f)
        records = []
        for sdata in report.get('seller_summaries', {}).values():
            if sdata.get('type') == 'benign':
                records.append({'selection_rate': sdata.get('selection_rate', 0.0)})
        return records
    except: return []

def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)
    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))
    print(f"Processing {len(folders)} directories...")
    for path in folders:
        info = parse_scenario_name(path.name)
        if info["type"] == "unknown": continue
        for mfile in path.rglob("final_metrics.json"):
            for r in load_run_data(mfile):
                all_records.append({**info, **r})
    return pd.DataFrame(all_records)

def get_baseline_averages(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates average healthy selection rate per defense."""
    base_df = df[df['attack'] == format_label("0. Baseline")]
    if base_df.empty: return {}
    return base_df.groupby('defense')['selection_rate'].mean().to_dict()

# ==========================================
# 3. THE COMBINED PLOTTING FUNCTION
# ==========================================

def plot_combined_summary_and_deep_dive(df: pd.DataFrame, deep_dive_attack: str, baseline_avgs: Dict, output_dir: Path):
    print(f"\n--- Generating Combined Figure (Deep Dive: {deep_dive_attack}) ---")

    # Setup the figure: 1 row, 2 columns.
    # width_ratios=[1.3, 1] makes the heatmap on the left slightly wider.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.3, 1]})
    plt.subplots_adjust(wspace=0.25) # Space between the two plots

    # ==========================
    # LEFT PANEL: OVERALL HEATMAP
    # ==========================
    ax_left = axes[0]

    # 1. Filter data: Remove Baseline rows (keeps heatmap focused on attacks)
    heatmap_df = df[df['attack'] != format_label("0. Baseline")].copy()

    # 2. Aggregate: Calculate mean selection rate
    agg_df = heatmap_df.groupby(['attack', 'defense'])['selection_rate'].mean().reset_index()

    # 3. Pivot for heatmap structure
    pivot_data = agg_df.pivot(index='attack', columns='defense', values='selection_rate')
    pivot_data = pivot_data.reindex(columns=DEFENSE_ORDER) # Ensure column order

    # 4. Plot Heatmap
    sns.heatmap(
        pivot_data, annot=True, fmt=".2f",
        cmap="RdYlGn", # Red=Low Selection (Attack wins), Green=High (Defense wins)
        linewidths=1, linecolor='white',
        cbar_kws={'label': 'Avg Selection Rate'},
        ax=ax_left
    )
    ax_left.set_title("A. Overall Attack Impact (Mean Selection Rate)")
    ax_left.set_ylabel("Attack Strategy")
    ax_left.set_xlabel("Defense Mechanism")

    # ==========================
    # RIGHT PANEL: DEEP DIVE BOXPLOT
    # ==========================
    ax_right = axes[1]
    target_attack_label = format_label(deep_dive_attack)

    # 1. Filter data for the specific attack
    subset_df = df[df['attack'] == target_attack_label].copy()

    if subset_df.empty:
        print(f"⚠️ Warning: No data found for attack '{target_attack_label}'. Skipping right panel.")
        ax_right.text(0.5, 0.5, "Data Not Found", ha='center')
    else:
        # 2. Plot Boxplot
        sns.boxplot(
            data=subset_df, x='defense', y='selection_rate',
            order=DEFENSE_ORDER, palette=DEFENSE_COLORS,
            linewidth=2, fliersize=4, ax=ax_right
        )

        # 3. Add Baseline Dashed Lines (optional but recommended)
        # This shows where the selection rate *should* be if there was no attack.
        for i, defense in enumerate(DEFENSE_ORDER):
             base_val = baseline_avgs.get(defense)
             if base_val is not None:
                 ax_right.hlines(y=base_val, xmin=i-0.4, xmax=i+0.4,
                                 colors='black', linestyles='--', lw=2, alpha=0.6,
                                 label="Healthy Baseline" if i == 0 else "")

        # Add legend just for the baseline line if it exists
        if baseline_avgs:
             ax_right.legend(loc='lower right', frameon=True, fontsize=11)

        ax_right.set_title(f"B. Deep Dive: {target_attack_label} Distribution")
        ax_right.set_ylim(-0.05, 1.05)
        ax_right.set_ylabel("Selection Rate")
        ax_right.set_xlabel("Defense Mechanism")

    # ==========================
    # FINAL SAVING
    # ==========================
    safe_name = re.sub(r'[^\w]', '', target_attack_label)
    fname = output_dir / f"Combined_Summary_{safe_name}.pdf"
    # use bbox_inches='tight' to ensure nothing gets cut off
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"  Saved: {fname.name}")

# ==========================================
# MAIN
# ==========================================
def main():
    set_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_data(BASE_RESULTS_DIR)
    if df.empty: return print("No data found.")

    # Calculate healthy baselines once
    baseline_avgs = get_baseline_averages(df)

    # Generate the combined plot
    plot_combined_summary_and_deep_dive(df, DEEP_DIVE_ATTACK, baseline_avgs, output_dir)

if __name__ == "__main__":
    main()