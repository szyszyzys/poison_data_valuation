import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/paper_final_impact"

# TARGET CONFIGURATION
TARGET_DATASET = "CIFAR-100"

# --- Naming Standards ---
PRETTY_NAMES = {
    "fedavg": "FedAvg",
    "fltrust": "FLTrust",
    "martfl": "MARTFL",
    "skymask": "SkyMask",
    "skymask_small": "SkyMask",
}

# --- Color Standards (Green for Good, Red for Bad) ---
PAYMENT_PALETTE = {
    "Benign": "#2ca02c",   # Green (Matches 'Accuracy' in your other script)
    "Adversary": "#d62728" # Red (Matches 'ASR' in your other script)
}

DEFENSE_ORDER = ["FedAvg", "FLTrust", "MARTFL", "SkyMask"]

def format_label(label: str) -> str:
    """Standardizes names."""
    if not isinstance(label, str): return str(label)
    return PRETTY_NAMES.get(label.lower(), label.replace("_", " ").title())

def set_publication_style():
    """Sets the 'Big & Bold' professional style (Matches your Benchmark script)."""
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
        # Compact figure size
        'figure.figsize': (12, 6),
        # Font Type 42 for papers
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
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

def load_realized_payments(base_dir: Path, dataset: str, metric_name: str) -> pd.DataFrame:
    """
    Loads valuation data and applies the ZEROING LOGIC.
    If a participant is NOT in 'selected_ids', their payment is 0.0.
    """
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning folders for {dataset} ({metric_name})...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)

        # Filter for the target dataset only
        if info['dataset'].lower() != dataset.lower().replace("-", "") and info['dataset'] != dataset:
             # handle loose matching (cifar100 vs CIFAR-100)
             if "cifar" in info['dataset'].lower() and "cifar" in dataset.lower():
                 pass # Acceptable match
             else:
                 continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f:
                    lines = f.readlines()

                # Analyze converged state (last 20%)
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    for sid, scores in valuations.items():
                        is_adv = str(sid).startswith('adv')

                        # 1. Find the Raw Score
                        raw_score = 0.0
                        score_found = False
                        for k, v in scores.items():
                            if metric_name in k and v is not None:
                                raw_score = float(v)
                                score_found = True
                                break

                        if not score_found: continue

                        # 2. APPLY ZEROING (Filter Logic)
                        realized_payment = raw_score if sid in selected_ids else 0.0

                        records.append({
                            "defense": format_label(info['defense']), # Apply formatting here
                            "dataset": info['dataset'],
                            "Type": "Adversary" if is_adv else "Benign",
                            "Realized Payment": realized_payment
                        })
            except Exception:
                pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING FUNCTION (UPDATED STYLE)
# ==========================================

def plot_compact_realized_payment(df: pd.DataFrame, metric: str, output_dir: Path):
    """
    Generates the Realized Payment Chart using the COMPACT STYLE.
    Legend is inside the chart, Y-axis has headroom.
    """
    print(f"  -> Plotting {metric}...")

    if df.empty:
        print("     [Warning] No data found.")
        return

    # Ensure consistent order
    defense_order = [d for d in DEFENSE_ORDER if d in df['defense'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- MAIN PLOT ---
    sns.barplot(
        data=df,
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

    # --- LABELS & TITLES ---
    clean_metric = metric.replace("_", " ").replace("loo", "LOO").replace("contrib", "Contribution").title()
    if "Kernelshap" in clean_metric: clean_metric = "Shapley Value"

    ax.set_ylabel("Avg Payment", labelpad=10) # Shortened for compactness
    ax.set_xlabel("")
    # ax.set_title(f"Realized Payment: {clean_metric}", pad=15) # Optional Title

    # --- LEGEND & LAYOUT ( The "Compact" Logic ) ---

    # 1. Zero Line
    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    # 2. Calculate dynamic Y-limits to fit legend
    # Get current limits
    y_min, y_max = ax.get_ylim()
    # If data is all negative or zero, we need room at top for legend
    if y_max <= 0: y_max = abs(y_min) * 0.2

    # Add 30% headroom for the internal legend
    headroom = (y_max - y_min) * 0.35
    ax.set_ylim(y_min, y_max + headroom)

    # 3. Internal Legend (Upper Center)
    ax.legend(
        loc='upper center',
        ncol=2,                 # 2 Columns (Benign, Adversary)
        frameon=False,
        fontsize=16,
        columnspacing=1.5,
        handletextpad=0.5
    )

    # 4. Despine
    sns.despine(left=False, bottom=False, right=True, top=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Save
    filename = f"Fig3_Realized_Payment_{metric}.pdf"
    save_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  Saved to {save_path}")
    plt.close()

# ==========================================
# 4. MAIN EXECUTION (LOOP OVER METRICS)
# ==========================================

def main():
    print("--- Starting Compact Realized Payment Analysis ---")

    set_publication_style()
    base_dir = Path(BASE_RESULTS_DIR)
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # THE LIST OF METRICS TO GENERATE
    METRICS_TO_PLOT = [
        "marginal_contrib_loo",  # LOO
        "kernelshap",            # Shapley
        "influence"              # Influence
    ]

    for metric in METRICS_TO_PLOT:
        # 1. Load
        df = load_realized_payments(base_dir, TARGET_DATASET, metric)

        # 2. Plot (if data exists)
        if not df.empty:
            plot_compact_realized_payment(df, metric, output_dir)
        else:
            print(f"⚠️  No data found for {metric}. Did you run the valuation experiment?")

    print("\n✅ All figures generated.")

if __name__ == "__main__":
    main()