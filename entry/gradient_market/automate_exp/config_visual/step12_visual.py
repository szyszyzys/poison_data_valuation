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
FIGURE_OUTPUT_DIR = "./figures/paper_final_impact"

# TARGET CONFIGURATION
TARGET_DATASET = "CIFAR-100"

# The Metric to Visualize (Choose ONE for the main paper figure)
# Options: 'marginal_contrib_loo', 'shapley', 'influence'
TARGET_METRIC = "marginal_contrib_loo"

# VISUAL SETTINGS
CUSTOM_PALETTE = {
    "Benign": "#2ca02c",   # Vivid Green (Safe/Good)
    "Adversary": "#d62728" # Vivid Red (Danger/Bad)
}

def set_plot_style():
    """Sets professional plotting style for academic papers (ACM/IEEE/VLDB)."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'grid.alpha': 0.3,
        # Font Type 42 is required for many conferences (avoids Type 3 fonts)
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'patch.edgecolor': 'black',
        'patch.linewidth': 1.2
    })

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_scenario_name(folder_name: str) -> Dict[str, str]:
    """
    Robustly parses folder names to extract Defense, Modality, and Dataset.
    Example: 'step12_main_summary_skymask_small_image_cifar100_cnn'
    """
    parts = folder_name.split('_')
    info = {"folder": folder_name, "defense": "unknown", "dataset": "unknown"}

    try:
        if 'summary' in parts:
            idx = parts.index('summary')
            remaining = parts[idx+1:]

            # Find modality index (image/text/tabular)
            modality_idx = -1
            for m in ['image', 'text', 'tabular']:
                if m in remaining:
                    modality_idx = remaining.index(m)
                    break

            if modality_idx > -1:
                # Extract Defense Name
                defense_parts = remaining[:modality_idx]
                defense_str = "_".join(defense_parts)

                # Normalize Defense Names
                if "skymask" in defense_str and "small" in defense_str:
                    info['defense'] = "skymask-s"
                else:
                    info['defense'] = defense_str

                # Extract Dataset Name
                if len(remaining) > modality_idx + 1:
                    raw_ds = remaining[modality_idx+1]
                    if "cifar" in raw_ds.lower():
                        info['dataset'] = raw_ds.upper().replace("CIFAR", "CIFAR-")
                    else:
                        info['dataset'] = raw_ds.capitalize()
    except Exception:
        pass

    return info

def load_realized_payments(base_dir: Path, dataset: str, metric_name: str) -> pd.DataFrame:
    """
    Loads valuation data and applies the FILTERING LOGIC.
    Rule: If a participant is NOT in 'selected_ids', their payment is 0.0.
    """
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning {len(scenario_folders)} folders for {dataset}...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Filter for the target dataset only
        if info['dataset'] != dataset:
            continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f:
                    lines = f.readlines()

                # Only analyze the converged state (last 20% of rounds)
                # This ensures we measure the stable market behavior
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)

                    # --- CRITICAL: Get the list of survivors ---
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    for sid, scores in valuations.items():
                        # Determine Type
                        is_adv = str(sid).startswith('adv')

                        # 1. Find the Raw Valuation Score (The "Theoretical" Value)
                        raw_score = 0.0
                        score_found = False

                        # Robust key search (handles 'val_shapley_score', 'shapley', etc.)
                        for k, v in scores.items():
                            if metric_name in k and v is not None:
                                raw_score = float(v)
                                score_found = True
                                break

                        if not score_found:
                            continue

                        # 2. APPLY THE ZEROING LOGIC (The "Realized" Value)
                        # If the defense filtered them out, they get PAID NOTHING.
                        realized_payment = raw_score if sid in selected_ids else 0.0

                        records.append({
                            **info,
                            "Type": "Adversary" if is_adv else "Benign",
                            "Realized Payment": realized_payment
                        })
            except Exception as e:
                # print(f"Error reading {j_file}: {e}")
                pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def plot_realized_payment_chart(df: pd.DataFrame, metric: str, output_dir: Path):
    """
    Generates the grouped bar chart with error bars.
    This visualizes the 'Average Paycheck' per participant type.
    """
    print(f"  -> Generating Realized Payment Chart for {metric}...")

    if df.empty:
        print("     [Warning] No data found to plot.")
        return

    # Filter Defenses (ensure consistent order)
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in df['defense'].unique()]

    plt.figure(figsize=(11, 6))

    # --- THE MAIN PLOT ---
    # sns.barplot automatically calculates the MEAN and Confidence Interval (Error Bar)
    ax = sns.barplot(
        data=df,
        x='defense',
        y='Realized Payment',
        hue='Type',
        order=defense_order,
        palette=CUSTOM_PALETTE,
        edgecolor='black',
        linewidth=1.2,
        capsize=0.15,     # Width of the error bar caps
        errwidth=1.5,     # Thickness of the error bar lines
        saturation=1.0    # Vivid colors
    )

    # --- AESTHETICS ---
    # Title and Labels
    clean_metric = metric.replace("_", " ").replace("loo", "LOO").replace("contrib", "Contribution").title()
    ax.set_title(f"Average Realized Payment per Round ({clean_metric})", fontweight='bold', pad=15)
    ax.set_ylabel("Mean Payment (Normalized Units)", fontweight='bold')
    ax.set_xlabel("", fontweight='bold')

    # Zero Line (Crucial for showing 0 payment)
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    # X-Axis Tick Formatting
    labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontweight='bold', fontsize=13)

    # Legend Formatting
    plt.legend(title="Participant Type", title_fontsize='12', fontsize='11', loc='upper right')

    # Grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Save
    filename = f"Fig3_Realized_Payment_{metric}.pdf"
    save_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"  -> Saved figure to {save_path}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Realized Payment Analysis ---")

    # Setup
    set_plot_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    df = load_realized_payments(base_dir, TARGET_DATASET, TARGET_METRIC)

    if df.empty:
        print("❌ Error: No data found. Please check BASE_RESULTS_DIR and TARGET_DATASET.")
        return
    else:
        print(f"✅ Loaded {len(df)} valuation records.")

    # 2. Generate Figure
    plot_realized_payment_chart(df, TARGET_METRIC, output_dir)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()