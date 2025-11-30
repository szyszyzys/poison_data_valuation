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
FIGURE_OUTPUT_DIR = "./figures/step12_final_impact"

TARGET_DATASET = "CIFAR-100"

# KEYS TO LOOK FOR (Auto-Discovery)
# The script will try to generate a figure for each of these if found.
REQUIRED_KEYS = [
    "marginal_contrib_loo",  # Note: removed _score based on your data
    "marginal_contrib_loo_score", # Fallback if naming changes
    "kernelshap_score",
    "influence_score",
    "sim_to_buyer"
]

# STYLING
PAYMENT_PALETTE = {"Benign": "#2ca02c", "Adversary": "#d62728"}
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
    """Sets a compact, bold style with large fonts for papers."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.weight': 'bold',
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.titlesize': 20, 'axes.labelsize': 18,
        'xtick.labelsize': 16, 'ytick.labelsize': 16,
        'legend.fontsize': 16, 'legend.title_fontsize': 18,
        'axes.linewidth': 2.0, 'axes.edgecolor': '#333333',
        'lines.linewidth': 2.0, 'figure.figsize': (10, 6),
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

def load_sparse_metrics(base_dir: Path, dataset: str, target_metric: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))

    print(f"Scanning for '{target_metric}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Dataset Matching (Loose)
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Check last 50% of rounds (Converged Stage)
                start_idx = max(0, int(len(lines) * 0.5))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # --- SPARSE CHECK ---
                    # Check if this specific round actually has the metric
                    # (e.g. Round 140 has it, Round 141 does not)
                    first_val = next(iter(valuations.values()))
                    if target_metric not in first_val:
                        continue # Skip this round

                    # --- EXTRACT & ZERO OUT ---
                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        if target_metric in data and data[target_metric] is not None:
                            try:
                                raw_score = float(data[target_metric])

                                # === THE CORE LOGIC ===
                                # If filtered (not in selected_ids), Payment = 0.0
                                # This visualizes "Discarding"
                                realized_payment = raw_score if sid in selected_ids else 0.0

                                records.append({
                                    "defense": format_label(info['defense']),
                                    "Type": "Adversary" if is_adv else "Benign",
                                    "Metric": target_metric,
                                    "Realized Payment": realized_payment
                                })
                            except: continue
            except: pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING (WITH DATA LABELS)
# ==========================================

def plot_readable_impact(df: pd.DataFrame, metric_name: str, output_dir: Path):
    if df.empty: return

    defense_order = [d for d in DEFENSE_ORDER if d in df['defense'].unique()]

    # Create Figure
    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot Bar Chart (Average Paycheck)
    sns.barplot(
        data=df, x='defense', y='Realized Payment', hue='Type',
        order=defense_order, palette=PAYMENT_PALETTE,
        edgecolor='black', linewidth=1.5, capsize=0.1, errwidth=1.5, ax=ax
    )

    # --- ADD DATA LABELS (The Readability Fix) ---
    # This prints the exact value on top of the bar
    for container in ax.containers:
        # Determine format based on scale
        values = [v if not np.isnan(v) else 0 for v in container.datavalues]
        max_val = np.max(np.abs(values))

        if max_val < 0.01: fmt = '%.1e'   # Scientific for Influence
        elif max_val < 1.0: fmt = '%.3f'  # 3 decimals for small LOO
        else: fmt = '%.2f'                # 2 decimals for large LOO/Shapley

        ax.bar_label(container, fmt=fmt, padding=3, fontsize=13, fontweight='bold')

    # Titles & Labels
    clean_title = metric_name.replace("_", " ").title().replace("Loo", "LOO")
    if "Kernelshap" in clean_title: clean_title = "Shapley Value (Realized Payment)"

    ax.set_ylabel("Avg Payment Score", labelpad=10)
    ax.set_xlabel("")
    ax.set_title(clean_title, pad=15)

    # Reference Line
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    # Layout Adjustment for Labels
    # Add 15% headroom so labels don't get cut off
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_max <= 0: y_max = abs(y_min) * 0.2
    ax.set_ylim(y_min, y_max + (y_range * 0.25))

    # Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=14)

    sns.despine(top=True, right=True)
    plt.tight_layout()

    filename = f"Fig_FinalImpact_{metric_name}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved figure: {save_path}")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- Starting Final Payment Impact Analysis ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all potential metrics
    for metric in REQUIRED_KEYS:
        df = load_sparse_metrics(base_dir, TARGET_DATASET, metric)

        if not df.empty:
            print(f"  -> Found {len(df)} records for {metric}. Plotting...")
            plot_readable_impact(df, metric, output_dir)
        else:
            print(f"  -> No data found for {metric} (skipping).")

    print("\n✅ All figures generated.")

if __name__ == "__main__":
    main()