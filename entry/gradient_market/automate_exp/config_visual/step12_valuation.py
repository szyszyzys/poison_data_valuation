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
FIGURE_OUTPUT_DIR = "./figures/step12_final_payment_impact_v2"

TARGET_DATASET = "CIFAR-100"

METRIC_CONFIGS = {
    # MODE B: Participation (Uses ALL rounds)
    "selection_rate": {
        "title": "Participation Rate (Avg Selection Frequency)",
        "ylabel": "Selection Rate (0-1)",
        "is_count": True
    },
    # MODE A: Economic (Uses Sparse rounds)
    "marginal_contrib_loo": {
        "title": "LOO Contribution (Realized Payment)",
        "ylabel": "Avg Payment Score",
        "is_count": False
    },
    "kernelshap_score": {
        "title": "Shapley Value (Realized Payment)",
        "ylabel": "Avg Payment Score",
        "is_count": False
    },
    "influence_score": {
        "title": "Influence Function (Realized Payment)",
        "ylabel": "Avg Payment Score",
        "is_count": False
    }
}

PAYMENT_PALETTE = {"Benign": "#2ca02c", "Adversary": "#d62728"}
PRETTY_NAMES = {"fedavg": "FedAvg", "fltrust": "FLTrust", "martfl": "MARTFL", "skymask": "SkyMask", "skymask_small": "SkyMask"}
DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.weight': 'bold',
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.titlesize': 20, 'axes.labelsize': 18,
        'xtick.labelsize': 16, 'ytick.labelsize': 16,
        'legend.fontsize': 16, 'legend.title_fontsize': 18,
        'axes.linewidth': 2.0, 'axes.edgecolor': '#333333',
        'lines.linewidth': 2.0, 'figure.figsize': (11, 6),
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

# ==========================================
# 2. DATA LOADING (DUAL MODE)
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_metrics(base_dir: Path, dataset: str, target_metric: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))

    print(f"Scanning for '{target_metric}'...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if dataset.lower().replace("-", "") not in info['dataset'].lower().replace("-", ""):
            continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Use last 50% of rounds for stability
                start_idx = max(0, int(len(lines) * 0.5))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # --- MODE B: SELECTION RATE (DENSE) ---
                    # We process this round regardless of whether valuation data exists
                    if target_metric == "selection_rate":
                        for sid in valuations.keys():
                            is_adv = str(sid).startswith('adv')
                            # Value is 1.0 if selected, 0.0 if not
                            payment = 1.0 if sid in selected_ids else 0.0
                            records.append({
                                "defense": format_label(info['defense']),
                                "Type": "Adversary" if is_adv else "Benign",
                                "Realized Payment": payment
                            })
                        continue

                    # --- MODE A: ECONOMIC METRICS (SPARSE) ---
                    # Only process if this specific metric exists in this round
                    first_val = next(iter(valuations.values()))
                    if target_metric not in first_val:
                        continue

                    for sid, data in valuations.items():
                        is_adv = str(sid).startswith('adv')
                        if target_metric in data and data[target_metric] is not None:
                            try:
                                raw_score = float(data[target_metric])
                                realized_payment = raw_score if sid in selected_ids else 0.0
                                records.append({
                                    "defense": format_label(info['defense']),
                                    "Type": "Adversary" if is_adv else "Benign",
                                    "Realized Payment": realized_payment
                                })
                            except: continue
            except: pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING
# ==========================================

def plot_payment_impact(df: pd.DataFrame, metric_key: str, output_dir: Path):
    if df.empty: return

    config = METRIC_CONFIGS.get(metric_key, {"title": metric_key, "ylabel": "Score"})
    defense_order = [d for d in DEFENSE_ORDER if d in df['defense'].unique()]

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=df, x='defense', y='Realized Payment', hue='Type',
        order=defense_order, palette=PAYMENT_PALETTE,
        edgecolor='black', linewidth=1.5, capsize=0.1,
        err_kws={'linewidth': 1.5},
        ax=ax
    )

    # Add Data Labels
    for container in ax.containers:
        values = [v if not np.isnan(v) else 0 for v in container.datavalues]
        max_val = np.max(np.abs(values))

        if config.get("is_count"): fmt = '%.2f'   # For Selection Rate (e.g. 0.95)
        elif max_val < 0.01: fmt = '%.1e'         # Scientific
        elif max_val < 1.0: fmt = '%.3f'          # Small decimals
        else: fmt = '%.2f'                        # Large numbers

        ax.bar_label(container, fmt=fmt, padding=4, fontsize=13, fontweight='bold')

    ax.set_ylabel(config["ylabel"], labelpad=10)
    ax.set_xlabel("")
    ax.set_title(config["title"], pad=15)
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    # Headroom adjustment
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_max <= 0: y_max = abs(y_min) * 0.2

    # Selection Rate usually needs to go up to 1.1 or 1.2
    if config.get("is_count"):
        ax.set_ylim(0, 1.25)
    else:
        ax.set_ylim(y_min, y_max + (y_range * 0.25))

    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=14, edgecolor='black')

    sns.despine(top=True, right=True)
    plt.tight_layout()

    filename = f"Fig_Impact_{metric_key}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved figure: {save_path}")
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- Starting Enhanced Payment Impact Analysis ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for metric in METRIC_CONFIGS.keys():
        df = load_metrics(base_dir, TARGET_DATASET, metric)

        if not df.empty:
            print(f"  -> Found {len(df)} records for {metric}. Plotting...")
            plot_payment_impact(df, metric, output_dir)
        else:
            print(f"  -> No data found for {metric} (skipping).")

    print("\n✅ All figures generated successfully.")

if __name__ == "__main__":
    main()