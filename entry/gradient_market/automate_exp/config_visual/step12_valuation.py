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
FIGURE_OUTPUT_DIR = "./figures/step12_final_benchmark"

# TARGET CONFIGURATION
TARGET_DATASET = "CIFAR-100"

# --- EXACT KEYS FROM YOUR JSON ---
# The script will look for these specific keys.
# Note: I removed '_score' from LOO based on your JSON snippet.
METRICS_TO_PLOT = [
    "marginal_contrib_loo",
    "kernelshap_score",
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
# 2. DATA LOADING (SPARSE + ZEROING)
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_realized_payments(base_dir: Path, dataset: str, target_key: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning for metric: '{target_key}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Dataset Match
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Check last 50% of rounds
                start_idx = max(0, int(len(lines) * 0.5))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # --- CHECK: Does this round have the metric? ---
                    # (e.g. Round 140 has it, Round 141 does not)
                    first_seller = next(iter(valuations.values()))
                    if target_key not in first_seller:
                        continue # Skip this round

                    # --- EXTRACT DATA ---
                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        if target_key in data and data[target_key] is not None:
                            try:
                                raw_score = float(data[target_key])
                                # Realized Payment Logic: 0 if filtered
                                realized_payment = raw_score if sid in selected_ids else 0.0

                                records.append({
                                    "defense": format_label(info['defense']),
                                    "Type": "Adversary" if is_adv else "Benign",
                                    "Realized Payment": realized_payment
                                })
                            except: continue
            except Exception: pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING
# ==========================================

def plot_compact_payment(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty:
        print(f"⚠️  No data found for {metric_name}")
        return

    defense_order = [d for d in DEFENSE_ORDER if d in df['defense'].unique()]
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df, x='defense', y='Realized Payment', hue='Type',
        order=defense_order, palette=PAYMENT_PALETTE,
        edgecolor='black', linewidth=1.2, capsize=0.1, errwidth=1.5, ax=ax
    )

    # Clean Title
    clean_title = metric_name.replace("_score", "").replace("_", " ").title()
    if "Loo" in clean_title: clean_title = "LOO Contribution"
    if "Kernelshap" in clean_title: clean_title = "Shapley Value"

    ax.set_ylabel("Avg Score", labelpad=10)
    ax.set_xlabel("")

    # Internal Legend Layout
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_max <= 0: y_max = abs(y_min) * 0.2

    # Add headroom for legend
    ax.set_ylim(y_min, y_max + (y_range * 0.35))

    ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=16, columnspacing=1.5)
    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    filename = f"Fig3_Payment_{metric_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"✅ Saved {filename} ({len(df)} records)")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================

def main():
    print("--- Starting Final Benchmark Visualization ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        df = load_realized_payments(base_dir, TARGET_DATASET, metric)
        plot_compact_payment(df, metric, output_dir)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()