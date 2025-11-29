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
FIGURE_OUTPUT_DIR = "./figures/step12_paper_ready"

# --- TARGET CONFIGURATION ---
# Only process these datasets. Leave empty [] to process all.
TARGET_DATASETS = ["CIFAR-100"]

# --- VISUAL CONSISTENCY ---
# Matches standard academic color schemes
CUSTOM_PALETTE = {
    "fedavg": "#4c72b0",   # Blue
    "fltrust": "#dd8452",  # Orange
    "martfl": "#55a868",   # Green
    "skymask": "#c44e52",  # Red
    "skymask-s": "#8c564b" # Brown (Distinct from SkyMask)
}

# High contrast for the "Intuition" plot
TYPE_PALETTE = {
    "Benign": "#2ca02c",   # Vivid Green
    "Adversary": "#d62728" # Vivid Red
}

def set_plot_style():
    """Sets professional plotting style with Type 42 fonts for LaTeX/ACM/IEEE."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.6) # Increased font scale for readability in two-column papers
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 2.0,
        'grid.alpha': 0.3,
        # CRITICAL FOR LATEX SUBMISSION:
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'cm',
        'legend.frameon': True,
        'legend.framealpha': 0.9
    })

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_scenario_name(folder_name: str) -> Dict[str, str]:
    """Robustly parses folder names."""
    parts = folder_name.split('_')
    info = {"folder": folder_name, "defense": "unknown", "modality": "unknown", "dataset": "unknown"}

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
                defense_parts = remaining[:modality_idx]
                defense_str = "_".join(defense_parts)
                # Normalize Defense
                if "skymask" in defense_str and "small" in defense_str:
                    info['defense'] = "skymask-s"
                else:
                    info['defense'] = defense_str

                # Normalize Dataset
                if len(remaining) > modality_idx + 1:
                    raw_ds = remaining[modality_idx+1]
                    if "cifar" in raw_ds.lower():
                        info['dataset'] = raw_ds.upper().replace("CIFAR", "CIFAR-")
                    else:
                        info['dataset'] = raw_ds.capitalize()
    except Exception:
        pass
    return info

def load_data(base_dir: Path, target_datasets: List[str]) -> (pd.DataFrame, pd.DataFrame):
    """Loads and filters data based on configuration."""
    global_records = []
    raw_val_records = []

    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning {len(scenario_folders)} folders...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # FILTERING LOGIC
        if info['dataset'] == 'unknown': continue
        if target_datasets and info['dataset'] not in target_datasets: continue

        # 1. Global Metrics (Accuracy/Rounds)
        metrics_files = list(folder.rglob("final_metrics.json"))
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

        global_records.append({**info, "acc": avg_acc, "rounds": avg_rounds})

        # 2. Valuation Data (The "Payment" Intuition)
        jsonl_files = list(folder.rglob("valuations.jsonl"))
        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f:
                    lines = f.readlines()
                # Only analyze the last 20% of rounds (converged state)
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    valuations = rec.get('seller_valuations', {})
                    for sid, scores in valuations.items():
                        s_type = 'Adversary' if str(sid).startswith('adv') else 'Benign'
                        for k, v in scores.items():
                            if k in ['round', 'seller_id'] or v is None: continue
                            # We only care about the main score for the main intuition plot
                            metric = k.replace('_score', '').replace('val_', '')
                            if metric == "shapley" or metric == "data_shapley" or metric == "influence":
                                # Assuming 'shapley' or similar is the primary metric.
                                # If you have multiple, the plot loop handles them.
                                pass

                            raw_val_records.append({
                                **info, "type": s_type, "metric": metric, "score": float(v)
                            })
            except: pass

    return pd.DataFrame(global_records), pd.DataFrame(raw_val_records)

# ==========================================
# 3. PAPER-READY PLOTTING
# ==========================================

def plot_utility_summary(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Figure 1: Side-by-Side Accuracy and Efficiency.
    Shows that the defense does not destroy model utility.
    """
    print(f"  -> Generating Fig 1 (Utility) for {dataset}")
    subset = df[df['dataset'] == dataset].copy()
    if subset.empty: return

    # Normalize Accuracy for display (0-100)
    subset['acc'] = subset['acc'].apply(lambda x: x*100 if x <= 1.0 else x)

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Accuracy
    sns.barplot(ax=axes[0], data=subset, x='defense', y='acc', order=defense_order, palette=CUSTOM_PALETTE, capsize=.1)
    axes[0].set_title(f"Model Utility ({dataset})", fontweight='bold')
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 100)

    # Subplot 2: Efficiency (Rounds)
    sns.barplot(ax=axes[1], data=subset, x='defense', y='rounds', order=defense_order, palette=CUSTOM_PALETTE, capsize=.1)
    axes[1].set_title("Convergence Speed", fontweight='bold')
    axes[1].set_ylabel("Rounds to Converge")
    axes[1].set_xlabel("")

    # Formatting Labels
    for ax in axes:
        labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / f"Fig1_Utility_{dataset}.pdf", bbox_inches='tight')
    plt.close()

def plot_payment_intuition(df_raw: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Figure 2: Split Violin Plot.
    THIS IS THE KEY INTUITION PLOT.
    It shows the distribution of "payments" (valuations).
    Goal: Show Benign (Green) is high, Adversary (Red) is pushed to zero.
    """
    print(f"  -> Generating Fig 2 (Payment Intuition) for {dataset}")
    subset = df_raw[df_raw['dataset'] == dataset].copy()
    if subset.empty: return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    # Filter for the most relevant metric if multiple exist (usually the first one found is primary)
    # Or iterate if you want one plot per metric type
    metrics = subset['metric'].unique()

    for metric in metrics:
        m_subset = subset[subset['metric'] == metric]

        plt.figure(figsize=(12, 6))

        # Split Violin Plot: The best way to show separation of distributions
        ax = sns.violinplot(
            data=m_subset, x='defense', y='score', hue='type',
            split=True,           # This splits the violin: Left=Benign, Right=Adv
            inner="quart",        # Shows quartiles inside the violin
            order=defense_order,
            palette=TYPE_PALETTE,
            linewidth=1.5,
            cut=0                 # Don't extend range beyond observed data (cleaner for 0 values)
        )

        # Aesthetics
        ax.set_title(f"Impact of Filtering on Participant Payments ({dataset})", fontweight='bold', pad=15)
        ax.set_ylabel("Approximated Payment\n(Valuation Score)", fontweight='bold')
        ax.set_xlabel("Defense Mechanism", fontweight='bold')

        # Legend
        plt.legend(title="Participant Type", loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

        # X-Axis Labels
        labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, size=12)

        # Zero line to emphasize adversaries getting 0 payment
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f"Fig2_Payment_Intuition_{dataset}_{metric}.pdf", bbox_inches='tight')
        plt.close()

# ==========================================
# 4. MAIN
# ==========================================

def main():
    print("--- Generating Paper-Ready Figures ---")
    set_plot_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    df_global, df_raw = load_data(Path(BASE_RESULTS_DIR), TARGET_DATASETS)

    if df_global.empty:
        print("❌ No data found for the requested configuration.")
        return

    # Generate Figures for each dataset found
    datasets = df_global['dataset'].unique()
    for ds in datasets:
        print(f"\nProcessing: {ds}")

        # Figure 1: Does the model work? (Accuracy/Rounds)
        plot_utility_summary(df_global, ds, output_dir)

        # Figure 2: Does the intuition hold? (Payment Separation)
        plot_payment_intuition(df_raw, ds, output_dir)

    print(f"\n✅ Done. Figures saved to {FIGURE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()