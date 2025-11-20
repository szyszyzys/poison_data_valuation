import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_sybil_comparison(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- 1. CONFIG ---
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    # --- 2. DATA PREP: Sophisticated Sorting ---
    # We want to group by Family (Standard vs Oracle) and then sort by Alpha
    def get_sort_key(label):
        # Returns tuple: (Family_Order, Alpha_Value)

        # 0. Baseline
        if label == 'baseline_no_sybil':
            return (0, 0.0)

        # 1. Standard Mimicry Family (Uses Historical/Estimated Centroid)
        if label == 'mimic':
            return (1, 0.1)      # Base Mimic
        if label == 'knock_out':
            return (1, 0.2)      # KnockOut is 2x Base
        if label == 'pivot':
            return (1, 1.0)      # Pivot is 100% Replacement

        # 2. Oracle Family (Uses True Centroid)
        if label.startswith('oracle_blend'):
            try:
                val = float(label.split('_')[-1])
                return (2, val)
            except:
                return (2, 0.5)

        return (3, 0.0) # Catch-all

    unique_labels = defense_df['strategy_label'].unique()
    sorted_labels = sorted(unique_labels, key=get_sort_key)

    # --- 3. DATA PREP: Metrics ---
    metric_map = {
        'acc': 'Model Accuracy',
        'asr': 'Attack Success Rate',
        'adv_selection_rate': 'Adv. Selection Rate',
        'benign_selection_rate': 'Benign Selection Rate'
    }
    metrics_to_plot = [m for m in metric_map.keys() if m in defense_df.columns]

    plot_df = defense_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- 4. PLOTTING ---
    plt.figure(figsize=(18, 8))

    ax = sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels,
        palette="deep",
        edgecolor="black",
        linewidth=1
    )

    # --- 5. STYLING ---
    plt.title(f'Impact of Centroid Knowledge & Aggressiveness ({defense.upper()})',
              fontsize=24, fontweight='bold', pad=20)
    plt.ylabel('Rate', fontsize=20, fontweight='bold')
    plt.xlabel('', fontsize=0)

    # --- 6. FORMATTING LABELS (The Scientific Naming) ---
    def format_label(l):
        # Baseline
        if l == 'baseline_no_sybil': return "Baseline"

        # --- FAMILY 1: STANDARD MIMIC (Estimated Centroid) ---
        if l == 'mimic':
            return "Mimic\n($\\alpha=0.1$)"
        if l == 'knock_out':
            return "Mimic\n($\\alpha=0.2$)" # Renamed from Knock-out
        if l == 'pivot':
            return "Mimic\n($\\alpha=1.0$)" # Renamed from Pivot

        # --- FAMILY 2: ORACLE MIMIC (True Centroid) ---
        if l.startswith('oracle_blend'):
            val = l.split('_')[-1]
            return f"Oracle\n($\\alpha={val}$)"

        return l.replace('_', '\n').title()

    formatted_labels = [format_label(l) for l in sorted_labels]

    ax.set_xticklabels(formatted_labels, rotation=0, fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)

    # --- 7. LEGEND & LAYOUT ---
    plt.legend(
        title=None,
        fontsize=16,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False
    )

    # Add a subtle vertical line to separate the two families visually (Optional)
    # Find the index where Oracle starts and draw a line
    # (This assumes Baseline is 0, Mimic family is 1..3, Oracle is 4..)
    # You can tune x-coord based on how many bars you have.

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    plot_file = output_dir / f"plot_sybil_effectiveness_{defense}.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')