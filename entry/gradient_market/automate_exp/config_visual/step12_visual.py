import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Dict, Any

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_main_summary"

# --- VISUAL CONSISTENCY ---
CUSTOM_PALETTE = {
    "fedavg": "#4c72b0",   # Deep Blue
    "fltrust": "#dd8452",  # Deep Orange
    "martfl": "#55a868",   # Deep Green
    "skymask": "#c44e52",  # Deep Red
    "skymask-s": "#c44e52" # Handle alias
}

TYPE_PALETTE = {
    "Benign": "#2ca02c",   # Green
    "Adversary": "#d62728" # Red
}

def set_plot_style():
    """Sets professional plotting style with Type 42 fonts for LaTeX."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 2.5,
        # CRITICAL FOR LATEX:
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'cm'
    })

# ==========================================
# 2. ROBUST PARSING LOGIC
# ==========================================

def parse_scenario_name(folder_name: str) -> Dict[str, str]:
    """
    Robustly parses folder names like:
    - step12_main_summary_fedavg_image_cifar100_cnn
    - step12_main_summary_skymask_small_image_CIFAR10_cnn
    """
    parts = folder_name.split('_')
    info = {"folder": folder_name, "defense": "unknown", "modality": "unknown", "dataset": "unknown"}

    try:
        if 'summary' in parts:
            idx = parts.index('summary')
            remaining = parts[idx+1:]

            # Find index of the modality to handle "skymask" vs "skymask_small"
            modality_idx = -1
            for m in ['image', 'text', 'tabular']:
                if m in remaining:
                    modality_idx = remaining.index(m)
                    break

            if modality_idx > -1:
                # Defense is everything before modality
                defense_parts = remaining[:modality_idx]
                defense_str = "_".join(defense_parts)
                # Normalize Defense
                if "skymask" in defense_str and "small" in defense_str:
                    info['defense'] = "skymask-s"
                else:
                    info['defense'] = defense_str

                info['modality'] = remaining[modality_idx]

                # Dataset is usually immediately after modality
                if len(remaining) > modality_idx + 1:
                    raw_ds = remaining[modality_idx+1]
                    # Normalize Dataset Name (CIFAR100 vs cifar100)
                    if "cifar" in raw_ds.lower():
                        info['dataset'] = raw_ds.upper().replace("CIFAR", "CIFAR-")
                    else:
                        info['dataset'] = raw_ds.capitalize()
    except Exception:
        pass

    return info

def load_all_data(base_dir: Path) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads TWO DataFrames:
    1. df_global: Accuracy, Rounds, Selection Rates (One row per experiment)
    2. df_raw_vals: Individual valuation scores (One row per seller per metric)
    """
    global_records = []
    raw_val_records = []

    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning {len(scenario_folders)} scenario folders...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if info['dataset'] == 'unknown': continue

        # --- A. Get Global Metrics (Accuracy) ---
        # Look for final_metrics.json recursively (in case of seed folders)
        metrics_files = list(folder.rglob("final_metrics.json"))

        # We average metrics across seeds for the "Performance" plots
        seeds_acc = []
        seeds_rounds = []

        for m_file in metrics_files:
            try:
                with open(m_file, 'r') as f:
                    d = json.load(f)
                seeds_acc.append(d.get('acc', 0))
                seeds_rounds.append(d.get('completed_rounds', 0))
            except: pass

        avg_acc = np.mean(seeds_acc) if seeds_acc else 0
        avg_rounds = np.mean(seeds_rounds) if seeds_rounds else 0

        # --- B. Get Valuation Data (Detailed) ---
        jsonl_files = list(folder.rglob("valuations.jsonl"))

        scenario_benign_sel = []
        scenario_adv_sel = []

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f:
                    lines = f.readlines()

                # Take last 20% of rounds for "Converged" view
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    for sid, scores in valuations.items():
                        s_type = 'Adversary' if str(sid).startswith('adv') else 'Benign'

                        # Selection Tracking
                        is_sel = 1 if sid in selected_ids else 0
                        if s_type == 'Adversary': scenario_adv_sel.append(is_sel)
                        else: scenario_benign_sel.append(is_sel)

                        # Raw Score Tracking
                        for k, v in scores.items():
                            if k in ['round', 'seller_id'] or v is None: continue
                            # Clean metric name
                            metric = k.replace('_score', '').replace('val_', '')

                            raw_val_records.append({
                                **info,
                                "type": s_type,
                                "metric": metric,
                                "score": float(v)
                            })
            except: pass

        # Consolidate Global Record
        ben_rate = np.mean(scenario_benign_sel) if scenario_benign_sel else 0
        adv_rate = np.mean(scenario_adv_sel) if scenario_adv_sel else 0

        global_records.append({
            **info,
            "acc": avg_acc,
            "rounds": avg_rounds,
            "Benign_selected": ben_rate,
            "Adversary_selected": adv_rate
        })

    return pd.DataFrame(global_records), pd.DataFrame(raw_val_records)

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_performance_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Fig 1: Accuracy, Rounds, and Selection Gap."""
    print(f"  -> Generating Performance Row for {dataset}")
    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Calculate Gap
    subset['Selection Gap'] = (subset['Benign_selected'] - subset['Adversary_selected']) * 100
    subset['acc'] = subset['acc'].apply(lambda x: x*100 if x<=1.0 else x)

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 1. Accuracy
    sns.barplot(ax=axes[0], data=subset, x='defense', y='acc', order=defense_order, palette=CUSTOM_PALETTE)
    axes[0].set_title("Global Accuracy (%)", fontweight='bold')
    axes[0].set_ylabel("")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 100)

    # 2. Rounds
    sns.barplot(ax=axes[1], data=subset, x='defense', y='rounds', order=defense_order, palette=CUSTOM_PALETTE)
    axes[1].set_title("Rounds to Converge", fontweight='bold')
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")

    # 3. Selection Gap
    sns.barplot(ax=axes[2], data=subset, x='defense', y='Selection Gap', order=defense_order, palette=CUSTOM_PALETTE)
    axes[2].set_title("Selection Advantage\n(Benign - Adversary)", fontweight='bold')
    axes[2].set_ylabel("Percentage Points")
    axes[2].set_xlabel("")
    axes[2].axhline(0, color='black', linewidth=1.5)

    # Clean Labels
    for ax in axes:
        labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

    plt.savefig(output_dir / f"Step12_Performance_{dataset}.pdf", bbox_inches='tight')
    plt.close()


def plot_valuation_distribution(df_raw: pd.DataFrame, dataset: str, output_dir: Path):
    """Fig 2: Strip Plot showing Fairness (Spread) and Security (Clamping)."""
    print(f"  -> Generating Valuation Distribution for {dataset}")
    subset = df_raw[df_raw['dataset'] == dataset].copy()
    if subset.empty: return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    metrics = subset['metric'].unique()

    for metric in metrics:
        m_subset = subset[subset['metric'] == metric]

        plt.figure(figsize=(12, 6))

        # Strip Plot (Individual Dots)
        ax = sns.stripplot(
            data=m_subset, x='defense', y='score', hue='type',
            order=defense_order, palette=TYPE_PALETTE,
            dodge=True, alpha=0.5, jitter=0.25, size=5, zorder=0
        )

        # Box Plot (Summary) - Transparent
        sns.boxplot(
            data=m_subset, x='defense', y='score', hue='type',
            order=defense_order, palette=TYPE_PALETTE,
            dodge=True, ax=ax, boxprops={'facecolor':'none', 'linewidth':1.5},
            fliersize=0, zorder=10
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Seller Type", loc='upper right')

        labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        ax.set_title(f"Valuation Distribution: {metric} ({dataset})", fontweight='bold')
        ax.set_ylabel("Attribution Score")
        ax.set_xlabel("")
        ax.axhline(0, color='black', linestyle='--')

        plt.savefig(output_dir / f"Step12_Dist_{dataset}_{metric}.pdf", bbox_inches='tight')
        plt.close()


def plot_payout_ratio(df_raw: pd.DataFrame, dataset: str, output_dir: Path):
    """Fig 3: Total Payout (Bar Chart) - The Security Argument."""
    print(f"  -> Generating Payout Ratio for {dataset}")
    subset = df_raw[df_raw['dataset'] == dataset].copy()
    if subset.empty: return

    # Calculate Mean Score by Type
    agg = subset.groupby(['defense', 'type'])['score'].mean().reset_index()

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in agg['defense'].unique()]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=agg, x='defense', y='score', hue='type',
        order=defense_order, palette=TYPE_PALETTE
    )

    labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    ax.set_title(f"Average Payout Ratio ({dataset})", fontweight='bold')
    ax.set_ylabel("Mean Valuation Score")
    ax.set_xlabel("")
    ax.axhline(0, color='black', linewidth=1)

    plt.savefig(output_dir / f"Step12_Payout_{dataset}.pdf", bbox_inches='tight')
    plt.close()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Step 12 Full Visualization ---")
    set_plot_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    df_global, df_raw = load_all_data(Path(BASE_RESULTS_DIR))

    if df_global.empty:
        print("❌ No data found. Check your directory structure (look for seed_x folders).")
        return

    # Save Summaries
    df_global.to_csv(output_dir / "summary_global.csv", index=False)
    df_raw.to_csv(output_dir / "summary_raw_valuations.csv", index=False)
    print(f"Data Loaded. Global: {len(df_global)} rows, Raw: {len(df_raw)} rows.")

    # 2. Generate Plots per Dataset
    datasets = df_global['dataset'].unique()

    for ds in datasets:
        print(f"\nProcessing Dataset: {ds}")
        try:
            # Figure 1: Performance (Acc, Rounds, Select)
            plot_performance_row(df_global, ds, output_dir)

            # Figure 2 & 3: Financials (Distribution & Payout)
            plot_valuation_distribution(df_raw, ds, output_dir)
            plot_payout_ratio(df_raw, ds, output_dir)

        except Exception as e:
            print(f"  Error processing {ds}: {e}")

    print("\n✅ All Figures Generated Successfully.")

if __name__ == "__main__":
    main()