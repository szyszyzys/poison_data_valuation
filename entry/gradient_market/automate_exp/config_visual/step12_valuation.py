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
FIGURE_OUTPUT_DIR = "./figures/step12_valuation_comparison"

# TARGET CONFIGURATION
TARGET_DATASET = "CIFAR-100"

# KEYS TO IGNORE (Don't plot these as valuation metrics)
IGNORE_KEYS = {
    "selected", "outlier", "train_loss", "num_samples",
    "selection_score", "gradient_norm", "price_paid", "round"
}

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",
}

PAYMENT_PALETTE = {
    "Benign": "#2ca02c",   # Green
    "Adversary": "#d62728" # Red
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    """Sets the Compact & Bold style."""
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
        'figure.figsize': (12, 6),
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# ==========================================
# 2. AUTO-DISCOVERY DATA LOADING
# ==========================================

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            idx = parts.index('summary')
            return {"defense": parts[idx + 1], "dataset": parts[idx + 3]}
    except: pass
    return {"defense": "unknown", "dataset": "unknown"}

def load_all_metrics_auto(base_dir: Path, dataset: str) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning {len(scenario_folders)} folders for {dataset}...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Loose dataset matching
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

                        # --- AUTO DISCOVERY LOGIC ---
                        # Loop through ALL keys in the seller's dictionary
                        for key, value in data.items():
                            # Skip non-metric keys (metadata, booleans, nulls)
                            if key in IGNORE_KEYS or value is None or isinstance(value, (bool, str)):
                                continue

                            try:
                                raw_score = float(value)

                                # Realized Payment Logic: 0 if filtered
                                realized_payment = raw_score if sid in selected_ids else 0.0

                                records.append({
                                    "defense": format_label(info['defense']),
                                    "Type": "Adversary" if is_adv else "Benign",
                                    "Metric": key,  # Capture the metric name dynamically
                                    "Realized Payment": realized_payment
                                })
                            except:
                                continue
            except: pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def plot_metric(df: pd.DataFrame, metric_name: str, output_dir: Path):
    """Plots a single metric from the large dataframe."""
    print(f"  -> Plotting {metric_name}...")

    # Filter for just this metric
    subset = df[df['Metric'] == metric_name].copy()
    if subset.empty: return

    defense_order = [d for d in DEFENSE_ORDER if d in subset['defense'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=subset,
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

    # Clean Labels
    clean_title = metric_name.replace("_", " ").replace("sim", "Similarity").title()
    if "Loo" in clean_title: clean_title = clean_title.replace("Loo", "LOO")

    ax.set_ylabel("Avg Score", labelpad=10)
    ax.set_xlabel("")
    # ax.set_title(clean_title, pad=15) # Optional title

    # Layout: Internal Legend + Zero Line
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_max <= 0: y_max = abs(y_min) * 0.2

    # Add headroom for legend
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
    print("--- Starting Auto-Discovery Visualization ---")
    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load EVERYTHING
    df = load_all_metrics_auto(base_dir, TARGET_DATASET)

    if df.empty:
        print("❌ No data found.")
        return

    # 2. Identify Unique Metrics Found
    found_metrics = df['Metric'].unique()
    print(f"✅ Found {len(df)} records across these metrics: {found_metrics}")

    # 3. Plot Each Metric Found
    for metric in found_metrics:
        plot_metric(df, metric, output_dir)

    print("\n✅ All figures generated.")

if __name__ == "__main__":
    main()