import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_paper_simple"

# --- TARGET CONFIGURATION ---
# Only process CIFAR-100 by default as requested.
TARGET_DATASETS = ["CIFAR-100"]

# --- VISUAL CONSISTENCY ---
# Standard paper color palette
CUSTOM_PALETTE = {
    "fedavg": "#4c72b0",   # Blue
    "fltrust": "#dd8452",  # Orange
    "martfl": "#55a868",   # Green
    "skymask": "#c44e52",  # Red
    "skymask-s": "#8c564b" # Brown
}

# High contrast for the Payment Intuition plot
TYPE_PALETTE = {
    "Benign": "#2ca02c",   # Vivid Green
    "Adversary": "#d62728" # Vivid Red
}

def set_plot_style():
    """Sets a clean, simple, rectangular plotting style."""
    sns.set_theme(style="whitegrid", font="serif")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams.update({
        'axes.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'grid.color': '#cccccc',
        'grid.linewidth': 1.0,
        # CRITICAL FOR LATEX: ensures text is searchable and clear
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        # Ensure bars/boxes have clean black outlines
        'patch.edgecolor': 'black',
        'patch.linewidth': 1.2
    })

# ==========================================
# 2. DATA LOADING (Robust & Filtered)
# ==========================================
def parse_scenario_name(folder_name: str) -> Dict[str, str]:
    parts = folder_name.split('_')
    info = {"folder": folder_name, "defense": "unknown", "dataset": "unknown"}
    try:
        if 'summary' in parts:
            idx = parts.index('summary')
            remaining = parts[idx+1:]
            modality_idx = -1
            for m in ['image', 'text', 'tabular']:
                if m in remaining:
                    modality_idx = remaining.index(m)
                    break
            if modality_idx > -1:
                defense_str = "_".join(remaining[:modality_idx])
                info['defense'] = "skymask-s" if "skymask" in defense_str and "small" in defense_str else defense_str
                if len(remaining) > modality_idx + 1:
                    raw_ds = remaining[modality_idx+1]
                    info['dataset'] = raw_ds.upper().replace("CIFAR", "CIFAR-") if "cifar" in raw_ds.lower() else raw_ds.capitalize()
    except Exception: pass
    return info

def load_data(base_dir: Path, target_datasets: List[str]) -> (pd.DataFrame, pd.DataFrame):
    global_records, raw_val_records = [], []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning {len(scenario_folders)} folders...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if info['dataset'] == 'unknown': continue
        if target_datasets and info['dataset'] not in target_datasets: continue

        # Get Global Metrics
        metrics_files = list(folder.rglob("final_metrics.json"))
        seeds_acc, seeds_rounds = [], []
        for m_file in metrics_files:
            try:
                with open(m_file, 'r') as f: d = json.load(f)
                seeds_acc.append(d.get('acc', 0))
                seeds_rounds.append(d.get('completed_rounds', 0))
            except: pass
        global_records.append({
            **info,
            "acc": np.mean(seeds_acc) if seeds_acc else 0,
            "rounds": np.mean(seeds_rounds) if seeds_rounds else 0
        })

        # Get Valuation Data (Converged rounds only)
        jsonl_files = list(folder.rglob("valuations.jsonl"))
        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                start_idx = max(0, int(len(lines) * 0.8))
                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    for sid, scores in rec.get('seller_valuations', {}).items():
                        s_type = 'Adversary' if str(sid).startswith('adv') else 'Benign'
                        for k, v in scores.items():
                            if k in ['round', 'seller_id'] or v is None: continue
                            metric = k.replace('_score', '').replace('val_', '')
                            raw_val_records.append({**info, "type": s_type, "metric": metric, "score": float(v)})
            except: pass
    return pd.DataFrame(global_records), pd.DataFrame(raw_val_records)

# ==========================================
# 3. CLEAN, RECTANGULAR PLOTTING
# ==========================================

def plot_utility_bars(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Fig 1: Simple, side-by-side bar charts for Accuracy and Efficiency."""
    print(f"  -> Generating Fig 1 (Utility Bars) for {dataset}")
    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    subset['acc'] = subset['acc'].apply(lambda x: x*100 if x <= 1.0 else x)
    defense_order = [d for d in ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s'] if d in subset['defense'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Accuracy Bar Plot
    sns.barplot(ax=axes[0], data=subset, x='defense', y='acc', order=defense_order, palette=CUSTOM_PALETTE)
    axes[0].set_title("Final Model Accuracy", fontweight='bold', pad=10)
    axes[0].set_ylabel("Test Accuracy (%)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 100)

    # 2. Rounds Bar Plot
    sns.barplot(ax=axes[1], data=subset, x='defense', y='rounds', order=defense_order, palette=CUSTOM_PALETTE)
    axes[1].set_title("Training Efficiency", fontweight='bold', pad=10)
    axes[1].set_ylabel("Rounds to Converge", fontweight='bold')
    axes[1].set_xlabel("")

    for ax in axes:
        labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
        ax.grid(axis='y', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / f"Fig1_Utility_Bars_{dataset}.pdf", bbox_inches='tight')
    plt.close()

def plot_payment_box(df_raw: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Fig 2: Grouped Box Plot.
    This is the best rectangular design to stress the "payment" intuition.
    It visually demonstrates how adversarial payments are squashed to zero.
    """
    print(f"  -> Generating Fig 2 (Payment Box Plot) for {dataset}")
    subset = df_raw[df_raw['dataset'] == dataset].copy()
    if subset.empty: return

    defense_order = [d for d in ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s'] if d in subset['defense'].unique()]

    # Iterate through metrics (e.g., shapley, influence)
    for metric in subset['metric'].unique():
        m_subset = subset[subset['metric'] == metric]
        plt.figure(figsize=(12, 6.5))

        # --- THE CORE VISUALIZATION ---
        # A Grouped Box Plot clearly compares the distributions side-by-side.
        ax = sns.boxplot(
            data=m_subset, x='defense', y='score', hue='type',
            order=defense_order, palette=TYPE_PALETTE,
            linewidth=1.8,    # Thicker lines for a "solid" look
            fliersize=4,      # Visible outlier dots
            saturation=1.0    # Solid, vivid colors
        )

        # Add a thick zero line to emphasize no-payment zone
        ax.axhline(0, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=0)

        # Labels stressing the payment intuition
        ax.set_title(f"Approximated Payment Distribution ({dataset})", fontweight='bold', fontsize=16, pad=15)
        # Clear Y-axis label linking score to payment
        ax.set_ylabel("Valuation Score (Approx. Payment)", fontweight='bold', fontsize=14)
        ax.set_xlabel("", fontsize=14)

        # Clean X-axis labels
        labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontweight='bold', fontsize=13)

        # Prominent Legend
        plt.legend(title="Participant Type", title_fontsize='13', fontsize='12', loc='upper right',
                   bbox_to_anchor=(1.15, 1.0), borderaxespad=0.)

        plt.grid(axis='y', linestyle='-', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for legend
        plt.savefig(output_dir / f"Fig2_Payment_Box_{dataset}_{metric}.pdf", bbox_inches='tight')
        plt.close()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("--- Generating Simplified Paper Figures ---")
    set_plot_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    df_global, df_raw = load_data(Path(BASE_RESULTS_DIR), TARGET_DATASETS)
    if df_global.empty:
        print("❌ No data found for the specified configuration.")
        return

    # 2. Generate Figures
    for ds in df_global['dataset'].unique():
        print(f"\nProcessing Dataset: {ds}")
        # Figure 1: The "It works" charts (Accuracy & Rounds)
        plot_utility_bars(df_global, ds, output_dir)
        # Figure 2: The "Intuition" chart (Payment Box Plot)
        plot_payment_box(df_raw, ds, output_dir)

    print(f"\n✅ Done. Simplified figures saved to: {FIGURE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()