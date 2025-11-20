import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_main_summary"

# --- VISUAL STYLING ---
# Metric-based Palette (Matches your screenshot)
METRIC_PALETTE = {
    "Accuracy": "#2ca02c",  # Green (Good)
    "ASR": "#d62728",  # Red (Bad)
    "Benign Select": "#1f77b4",  # Blue (Neutral)
    "Adv Select": "#ff7f0e"  # Orange (Warning)
}


def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 2.5


# --- DATA LOADING FUNCTIONS (From Step 12) ---

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        parts = scenario_name.split('_')
        if 'step12' in parts and 'main' in parts:
            try:
                idx = parts.index('summary')
                return {
                    "defense": parts[idx + 1],
                    "dataset": parts[idx + 3]
                }
            except IndexError:
                pass
    except:
        pass
    return {"defense": "unknown", "dataset": "unknown"}


def load_metrics_from_csv(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if df.empty or 'seller_id' not in df.columns: return pd.DataFrame()
        df['type'] = df['seller_id'].apply(lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign')

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

    return pd.DataFrame(all_runs)


# --- PLOTTING FUNCTION ---

def plot_grouped_benchmark(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates the Grouped Bar Chart (4 bars per defense).
    """
    print(f"\n--- Generating Grouped Benchmark for {dataset} ---")

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # 1. Prepare Data
    # Convert to percentages
    if subset['acc'].max() <= 1.0: subset['acc'] *= 100
    if subset['asr'].max() <= 1.0: subset['asr'] *= 100

    if 'Benign_selected' in subset.columns: subset['Benign_selected'] *= 100
    if 'Adversary_selected' in subset.columns: subset['Adversary_selected'] *= 100

    # Aggregate means across seeds
    agg_df = subset.groupby('defense').mean(numeric_only=True).reset_index()

    # 2. Melt into Long Format for Seaborn
    # We map column names to the Legend Labels we want
    rename_map = {
        'acc': 'Accuracy',
        'asr': 'ASR',
        'Benign_selected': 'Benign Select',
        'Adversary_selected': 'Adv Select'
    }

    # Filter only columns that exist
    cols_to_melt = [c for c in rename_map.keys() if c in agg_df.columns]

    melted = agg_df.melt(
        id_vars=['defense'],
        value_vars=cols_to_melt,
        var_name='Metric Type',
        value_name='Percentage'
    )

    # Rename the values in the column to nice names
    melted['Metric Type'] = melted['Metric Type'].map(rename_map)

    # 3. Setup Plot
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    defense_order = [d for d in defense_order if d in melted['defense'].unique()]

    # 4. Draw Grouped Bars
    sns.barplot(
        data=melted,
        x='defense',
        y='Percentage',
        hue='Metric Type',
        order=defense_order,
        palette=METRIC_PALETTE,
        edgecolor='white',
        linewidth=1,
        ax=ax
    )

    # 5. Styling & Labels
    # ax.set_title(f"Main Benchmark: Defense Capabilities vs. Backdoor Attack ({dataset})",
    #              fontweight='bold', fontsize=16, pad=20)

    # Format Defense Names on X Axis
    labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                          "SkyMask").replace(
        "Martfl", "MARTFL") for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_xlabel("", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=14)
    ax.set_ylim(0, 115)  # Extra space for legend/labels

    # 6. Add Value Labels on Bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10)

    # 7. Legend Configuration (Top Center)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        title=None,
        fontsize=12
    )

    # 8. Final Cleanups
    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 9. Save
    fname = output_dir / f"Step12_Grouped_Benchmark_{dataset}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  Saved plot to: {fname}")
    plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir.resolve()}")

    # 1. Load Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data found.")
        return

    # 2. Generate Plot for each dataset
    for dataset in df['dataset'].unique():
        if dataset != 'unknown':
            plot_grouped_benchmark(df, dataset, output_dir)

    print("\n✅ Grouped Benchmark Figure Generated.")


if __name__ == "__main__":
    main()