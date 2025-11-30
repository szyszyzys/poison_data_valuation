import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_percentage_distribution"

TARGET_DATASET = "CIFAR-100"

# METRICS TO PROCESS
# We treat 'selection_rate' as a special metric where Value = 1.0
METRICS_TO_PLOT = [
    "selection_rate",             # NEW: Participation as value
    "marginal_contrib_loo",       # Economic Value
    "kernelshap_score",           # Economic Value
    "influence_score"             # Economic Value
]

# COLORS (Traffic Light)
STACK_COLORS = {
    "Paid to Benign": "#2ca02c",      # Green (Efficient)
    "Discarded Benign": "#bbbbbb",    # Grey (Waste)
    "Paid to Adversary": "#d62728"    # Red (Theft)
}

PRETTY_NAMES = {
    "fedavg": "FedAvg", "fltrust": "FLTrust",
    "martfl": "MARTFL", "skymask": "SkyMask",
    "skymask_small": "SkyMask"
}
DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.weight': 'bold',
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'legend.fontsize': 14, 'figure.figsize': (10, 7),
        'axes.linewidth': 2.0, 'axes.edgecolor': '#333333',
        'pdf.fonttype': 42, 'ps.fonttype': 42
    })

# ==========================================
# 2. DATA LOADING (SUMMING TOTALS)
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_financial_totals(base_dir: Path, dataset: str, target_metric: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning for '{target_metric}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        defense_name = format_label(info['defense'])
        jsonl_files = list(folder.rglob("valuations.jsonl"))

        # Accumulators
        total_paid_benign = 0.0
        total_discarded_benign = 0.0
        total_paid_adv = 0.0
        round_count = 0

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                start_idx = max(0, int(len(lines) * 0.5)) # Converged stages

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # --- SPECIAL LOGIC FOR SELECTION RATE ---
                    if target_metric == "selection_rate":
                        round_valid = True
                        for sid in valuations.keys():
                            is_adv = str(sid).startswith('adv')
                            # Value is 1.0 if selected, 0.0 if not?
                            # Actually, for "distribution", every seller represents 1 unit of potential value.
                            val = 1.0

                            if is_adv:
                                if sid in selected_ids: total_paid_adv += val
                            else:
                                if sid in selected_ids: total_paid_benign += val
                                else: total_discarded_benign += val

                        if round_valid: round_count += 1
                        continue

                    # --- LOGIC FOR ECONOMIC METRICS ---
                    # Check existence
                    first_val = next(iter(valuations.values()))
                    if target_metric not in first_val: continue

                    round_valid = False
                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')
                        if target_metric in data and data[target_metric] is not None:
                            val = max(0, float(data[target_metric])) # Clamp negative values for pie chart logic
                            round_valid = True

                            if is_adv:
                                if sid in selected_ids: total_paid_adv += val
                            else:
                                if sid in selected_ids: total_paid_benign += val
                                else: total_discarded_benign += val

                    if round_valid: round_count += 1
            except: pass

        if round_count > 0:
            records.append({
                "defense": defense_name,
                "Paid to Benign": total_paid_benign,
                "Discarded Benign": total_discarded_benign,
                "Paid to Adversary": total_paid_adv
            })

    return pd.DataFrame(records)

# ==========================================
# 3. 100% STACKED BAR PLOTTING
# ==========================================

def plot_percentage_distribution(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty: return

    # Prepare DataFrame
    df = df.set_index("defense")
    existing_order = [d for d in DEFENSE_ORDER if d in df.index]
    df = df.loc[existing_order]

    cols = ["Paid to Benign", "Discarded Benign", "Paid to Adversary"]
    df = df[cols]

    # --- NORMALIZE TO PERCENTAGES (The Magic Step) ---
    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    # Create Plot
    ax = df_pct.plot(
        kind='bar', stacked=True, figsize=(10, 7),
        color=[STACK_COLORS[c] for c in cols],
        edgecolor='black', linewidth=1.2, width=0.65
    )

    # --- ADD LABELS ---
    for c in ax.containers:
        labels = []
        for v in c.datavalues:
            # Only label if segment is big enough (>3%)
            if v < 3.0:
                labels.append("")
            else:
                labels.append(f"{v:.0f}%")

        ax.bar_label(c, labels=labels, label_type='center', fontsize=14, fontweight='bold', color='white')

    # Titles & Labels
    clean_title = metric_name.replace("_", " ").title().replace("Loo", "LOO")
    if "Selection" in clean_title: clean_title = "Selection Count (Participation)"
    if "Kernelshap" in clean_title: clean_title = "Shapley Value"

    ax.set_ylabel("Percentage of Total Value Distributed", labelpad=10)
    ax.set_xlabel("")
    ax.set_title(f"Value Distribution: {clean_title}", pad=20)
    ax.set_ylim(0, 100)

    plt.xticks(rotation=0)

    # Legend
    plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', borderaxespad=0., ncol=3, frameon=False)

    plt.tight_layout()

    filename = f"Fig_PctDistribution_{metric_name}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- Starting Percentage Distribution Analysis ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        df = load_financial_totals(base_dir, TARGET_DATASET, metric)
        if not df.empty:
            plot_percentage_distribution(df, metric, output_dir)
        else:
            print(f"  -> Skipping {metric} (No data)")

if __name__ == "__main__":
    main()