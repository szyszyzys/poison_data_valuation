import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. CONFIGURATION & COMPACT STYLING
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_valuation_comparison"

# TARGET CONFIGURATION
TARGET_DATASET = "CIFAR-100"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",
}

# --- Color Standards (Green=Benign, Red=Adversary) ---
PAYMENT_PALETTE = {
    "Benign": "#2ca02c",   # Green
    "Adversary": "#d62728" # Red
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    """Sets the 'Compact & Bold' style you requested."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'legend.title_fontsize': 18,
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.0,
        'figure.figsize': (12, 6), # Compact Size
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# ==========================================
# 2. DATA LOADING (Updated for your JSON keys)
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {
                "defense": parts[idx + 1],
                "dataset": parts[idx + 3]
            }
    except Exception:
        pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_realized_payments(base_dir: Path, dataset: str, target_key: str) -> pd.DataFrame:
    """
    Reads valuations.jsonl and looks for the 'target_key'
    (e.g., 'influence_score', 'sim_to_buyer', 'marginal_contrib_loo_score').
    """
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning for metric key: '{target_key}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Loose matching for dataset name
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()

                # Analyze converged state (last 20%)
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        # --- KEY CHANGE: Direct Key Access ---
                        # We check if the specific key exists in the JSON dictionary
                        if target_key in data and data[target_key] is not None:
                            raw_score = float(data[target_key])

                            # Realized Payment Logic: 0 if not selected
                            realized_payment = raw_score if sid in selected_ids else 0.0

                            records.append({
                                "defense": format_label(info['defense']),
                                "Type": "Adversary" if is_adv else "Benign",
                                "Realized Payment": realized_payment
                            })
            except Exception:
                pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING (Compact Style)
# ==========================================

def plot_compact_payment(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty:
        print(f"  [Warning] No data found for {metric_name}")
        return

    # Filter Defenses
    defense_order = [d for d in DEFENSE_ORDER if d in df['defense'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df,
        x='defense',
        y='Realized Payment',
        hue='Type',
        order=defense_order,
        palette=PAYMENT_PALETTE,
        edgecolor='black',
        linewidth=1.2,
        capsize=0.1,
        errwidth=1.5,
        ax=ax
    )

    # --- Labels ---
    clean_title = metric_name.replace("_", " ").replace("sim", "Similarity").title()
    ax.set_ylabel("Avg Score", labelpad=10)
    ax.set_xlabel("")

    # --- Compact Internal Legend ---
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    # Dynamic Y-Limits to fit legend inside
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_max <= 0: y_max = abs(y_min) * 0.2 # Handle negative-only plots

    # Add 35% headroom
    ax.set_ylim(y_min, y_max + (y_range * 0.35))

    ax.legend(
        loc='upper center',
        ncol=2,
        frameon=False,
        fontsize=16,
        columnspacing=1.5
    )

    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    filename = f"Fig_Valuation_{metric_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  Saved {filename}")
    plt.close()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # --- METRICS TO PLOT ---
    # These keys MUST match the keys in your JSONL file exactly
    METRICS_TO_PROCESS = [
        "influence_score",              # Found in your snippet
        "sim_to_buyer",                 # Found in your snippet
        "sim_to_oracle",                # Found in your snippet
        "marginal_contrib_loo_score",   # Try this if you have LOO data
        "kernel_shap_score"             # Try this if you have Shapley data
    ]

    print(f"--- Generating Comparisons for {len(METRICS_TO_PROCESS)} Metrics ---")

    for metric_key in METRICS_TO_PROCESS:
        df = load_realized_payments(base_dir, TARGET_DATASET, metric_key)

        if not df.empty:
            plot_compact_payment(df, metric_key, output_dir)
        else:
            print(f"⚠️  Skipping {metric_key} (Key not found in JSON data)")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()