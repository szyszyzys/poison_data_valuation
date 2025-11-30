import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_valuation_sparse"

# TARGET CONFIGURATION
TARGET_DATASET = "CIFAR-100"

# --- THE METRICS YOU WANT TO FIND ---
# The script will scan every round. If a round has these keys, it keeps it.
# If a round is missing them (e.g. Round 33), it ignores just that round.
REQUIRED_KEYS = [
    "marginal_contrib_loo_score",
    "kernel_shap_score",
    "influence_score"
]

# --- STYLING ---
PAYMENT_PALETTE = {"Benign": "#2ca02c", "Adversary": "#d62728"}
PRETTY_NAMES = {"fedavg": "FedAvg", "fltrust": "FLTrust", "martfl": "MARTFL", "skymask": "SkyMask", "skymask_small": "SkyMask"}
DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.weight': 'bold',
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.titlesize': 24, 'axes.labelsize': 20,
        'xtick.labelsize': 18, 'ytick.labelsize': 18,
        'legend.fontsize': 16, 'legend.title_fontsize': 18,
        'axes.linewidth': 2.0, 'axes.edgecolor': '#333333',
        'lines.linewidth': 3.0, 'figure.figsize': (12, 6),
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

# ==========================================
# 2. SPARSE DATA LOADING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_sparse_metrics(base_dir: Path, dataset: str, target_metric: str) -> pd.DataFrame:
    records = []

    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning for '{target_metric}' in {dataset}...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Dataset Matching
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Check last 50% of rounds to ensure we hit some multiples of 10
                start_idx = max(0, int(len(lines) * 0.5))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # --- SPARSE CHECK ---
                    # Check if ANY seller has the target metric for this specific round
                    # If not (e.g. Round 33), skip this round silently
                    has_metric = False
                    for val in valuations.values():
                        if target_metric in val:
                            has_metric = True
                            break

                    if not has_metric:
                        continue # Skip Round 33, go to next line

                    # If we found the metric, record it!
                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        if target_metric in data and data[target_metric] is not None:
                            try:
                                raw_score = float(data[target_metric])
                                realized_payment = raw_score if sid in selected_ids else 0.0

                                records.append({
                                    "defense": format_label(info['defense']),
                                    "Type": "Adversary" if is_adv else "Benign",
                                    "Metric": target_metric,
                                    "Realized Payment": realized_payment
                                })
                            except: continue

            except Exception: pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def plot_metric(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty:
        print(f"⚠️  No data found for {metric_name} (Sparse load returned 0 records)")
        return

    defense_order = [d for d in DEFENSE_ORDER if d in df['defense'].unique()]
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df, x='defense', y='Realized Payment', hue='Type',
        order=defense_order, palette=PAYMENT_PALETTE,
        edgecolor='black', linewidth=1.2, capsize=0.1, errwidth=1.5, ax=ax
    )

    clean_title = metric_name.replace("_", " ").replace("sim", "Similarity").title()
    ax.set_ylabel("Avg Score", labelpad=10)
    ax.set_xlabel("")

    ax.axhline(0, color='black', linewidth=1.5, zorder=0)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_max <= 0: y_max = abs(y_min) * 0.2
    ax.set_ylim(y_min, y_max + (y_range * 0.35))

    ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=16, columnspacing=1.5)
    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    filename = f"Fig_Sparse_{metric_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"✅ Saved {filename} (based on {len(df)} sparse records)")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================

def main():
    print("--- Starting Sparse Valuation Analysis ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loop through the keys you expect to exist (sparsely)
    for metric in REQUIRED_KEYS:
        df = load_sparse_metrics(base_dir, TARGET_DATASET, metric)
        plot_metric(df, metric, output_dir)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()