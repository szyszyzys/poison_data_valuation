import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_average_paycheck"

TARGET_DATASETS = ["CIFAR-100"]
# The metric you want to highlight (change as needed)
TARGET_METRIC = "marginal_contrib_loo"  # or "shapley", "influence"

# --- STYLE & PALETTE ---
CUSTOM_PALETTE = {
    "Benign": "#2ca02c",   # Green (Good)
    "Adversary": "#d62728" # Red (Bad)
}

def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams.update({
        'axes.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'patch.edgecolor': 'black',
        'patch.linewidth': 1.2
    })

# ==========================================
# 2. DATA PROCESSING (The "Average" Logic)
# ==========================================
def parse_scenario_name(folder_name: str):
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
    except: pass
    return info

def load_average_data(base_dir: Path, dataset: str, metric_name: str):
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning folders for {dataset}...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if info['dataset'] != dataset: continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Analyze converged rounds (last 20%)
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # Accumulators for THIS specific round
                    adv_scores = []
                    benign_scores = []

                    for sid, scores in valuations.items():
                        # Determine Type
                        is_adv = str(sid).startswith('adv')

                        # Get Score
                        val = 0.0
                        for k, v in scores.items():
                            if metric_name in k and v is not None:
                                val = float(v)
                                break

                        # IMPORTANT: We track every individual score to calculate proper means/error bars
                        if is_adv:
                            adv_scores.append(val)
                        else:
                            benign_scores.append(val)

                    # Append individual data points (Let Seaborn handle the averaging)
                    # This preserves variance info for error bars
                    for s in adv_scores:
                        records.append({**info, "Type": "Adversary", "Score": s})
                    for s in benign_scores:
                        records.append({**info, "Type": "Benign", "Score": s})

            except: pass

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING THE INSIGHT
# ==========================================
def plot_average_paycheck(df: pd.DataFrame, metric: str, output_dir: Path):
    print(f"  -> Generating Average Paycheck Plot for {metric}")
    if df.empty: return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in df['defense'].unique()]

    plt.figure(figsize=(10, 6))

    # BAR CHART with ERROR BARS (ci=95)
    # This automatically calculates the MEAN of the 0.3 Adv and 0.7 Benign separately
    ax = sns.barplot(
        data=df,
        x='defense',
        y='Score',
        hue='Type',
        order=defense_order,
        palette=CUSTOM_PALETTE,
        edgecolor='black',
        capsize=0.1,    # Adds caps to error bars
        errwidth=1.5    # Thickness of error bars
    )

    # Styling
    clean_metric = metric.replace("_", " ").replace("loo", "LOO").title()
    ax.set_title(f"Average Payment Per Participant ({clean_metric})", fontweight='bold', pad=15)
    ax.set_ylabel("Mean Valuation Score", fontweight='bold')
    ax.set_xlabel("", fontweight='bold')

    # Zero line
    ax.axhline(0, color='black', linewidth=1.5)

    # Clean Labels
    labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)

    # Legend
    plt.legend(title="Participant Type", loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / f"Fig_AvgPaycheck_{metric}.pdf", bbox_inches='tight')
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    set_plot_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Load Data (Individual points, not pre-aggregated)
    df = load_average_data(Path(BASE_RESULTS_DIR), TARGET_DATASETS[0], TARGET_METRIC)

    if df.empty:
        print("❌ No data found.")
        return

    plot_average_paycheck(df, TARGET_METRIC, output_dir)
    print(f"✅ Created Average Paycheck figure in {FIGURE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()