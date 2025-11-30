import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_main_summary"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",
}

# --- Color Standards ---
METRIC_PALETTE = {
    "Accuracy": "#2ca02c",      # Green (Good)
    "ASR": "#d62728",           # Red (Bad)
    "Benign Select": "#1f77b4", # Blue (Neutral)
    "Adv Select": "#ff7f0e"     # Orange (Warning)
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    """Standardizes names."""
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    """Sets the 'Big & Bold' professional style globally."""
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
        'legend.fontsize': 16,     # Slightly smaller to fit inside
        'legend.title_fontsize': 18,
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.0,
        # Tighter figure size since legend is inside
        'figure.figsize': (12, 6),
    })

# ==========================================
# 2. DATA LOADING & PARSING
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


def load_metrics_from_csv(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if df.empty or 'seller_id' not in df.columns: return pd.DataFrame()

        df['type'] = df['seller_id'].apply(
            lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign'
        )

        if 'selected' in df.columns:
            df['selected'] = df['selected'].astype(int)
            summary = df.groupby('type')[['selected']].mean().reset_index()
            return summary
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for Step 12 results in {base_path}...")

    scenario_folders = [f for f in base_path.glob("step12_*") if f.is_dir()]

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("defense") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_dir = metrics_file.parent
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                asr = metrics.get('asr', 0)
            except:
                acc = 0; asr = 0

            df_val = load_metrics_from_csv(run_dir)
            flat_record = {**run_scenario, "acc": acc, "asr": asr}

            if not df_val.empty:
                for _, row in df_val.iterrows():
                    s_type = row['type']
                    if 'selected' in row:
                        flat_record[f"{s_type}_selected"] = row['selected']

            all_runs.append(flat_record)

    df = pd.DataFrame(all_runs)
    if not df.empty:
        df['defense'] = df['defense'].apply(format_label)

    return df

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_grouped_benchmark(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Generates the Grouped Bar Chart with INTERNAL LEGEND."""
    print(f"\n--- Generating Grouped Benchmark for {dataset} ---")

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Normalize
    if subset['acc'].max() <= 1.0: subset['acc'] *= 100
    if subset['asr'].max() <= 1.0: subset['asr'] *= 100
    if 'Benign_selected' in subset.columns: subset['Benign_selected'] *= 100
    if 'Adversary_selected' in subset.columns: subset['Adversary_selected'] *= 100

    # Aggregate
    agg_df = subset.groupby('defense').mean(numeric_only=True).reset_index()

    # Melt
    rename_map = {
        'acc': 'Accuracy',
        'asr': 'ASR',
        'Benign_selected': 'Benign Select',
        'Adversary_selected': 'Adv Select'
    }
    cols_to_melt = [c for c in rename_map.keys() if c in agg_df.columns]

    melted = agg_df.melt(
        id_vars=['defense'],
        value_vars=cols_to_melt,
        var_name='Metric Type',
        value_name='Percentage'
    )
    melted['Metric Type'] = melted['Metric Type'].map(rename_map)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6)) # Compact size

    defense_order = [d for d in DEFENSE_ORDER if d in melted['defense'].unique()]

    sns.barplot(
        data=melted,
        x='defense',
        y='Percentage',
        hue='Metric Type',
        order=defense_order,
        palette=METRIC_PALETTE,
        edgecolor='black', # Consistent black edge
        linewidth=1.2,
        ax=ax
    )

    # Labels
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)", labelpad=10)

    # --- SPACE SAVING LEGEND CONFIGURATION ---
    # 1. Increase Y-Limit to make room INSIDE the axes
    ax.set_ylim(0, 135)

    # 2. Place Legend INSIDE (upper center)
    ax.legend(
        loc='upper center',
        ncol=4,                 # Horizontal layout
        frameon=False,          # Cleaner look without box
        fontsize=15,
        columnspacing=1.2,
        handletextpad=0.4
    )

    # Bar Annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=13, fontweight='bold')

    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Save
    fname = output_dir / f"Step12_Grouped_Benchmark_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  Saved plot to: {fname}")
    plt.close()


def main():
    set_publication_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data found.")
        return

    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_grouped_benchmark(df, dataset, output_dir)

    print("\n✅ Compact Grouped Benchmark Generated.")


if __name__ == "__main__":
    main()