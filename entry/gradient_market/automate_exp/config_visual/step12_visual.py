import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step12_financial_impact_by_metric"

# We only focus on CIFAR-100 as requested
TARGET_DATASETS = ["CIFAR-100"]

# The specific metrics you want to analyze
TARGET_METRICS = ["marginal_contrib_loo", "kernelshap", "influence"]

# --- COLOR PALETTE ---
# Green = Success (Paid to honest workers)
# Grey  = Collateral Damage (Honest workers we accidentally filtered)
# Red   = Security Failure (Money stolen by attackers)
FINANCIAL_PALETTE = {
    "Paid to Benign": "#2ca02c",      # Vivid Green
    "Discarded Benign": "#bbbbbb",    # Light Grey
    "Paid to Adversary": "#d62728"    # Vivid Red
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
        'patch.linewidth': 1.0
    })

# ==========================================
# 2. DATA PARSING
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

def load_financial_data_by_metric(base_dir: Path, dataset: str, target_metrics: List[str]) -> pd.DataFrame:
    records = []
    scenario_folders = list(base_dir.glob("step12_*"))
    print(f"Scanning {len(scenario_folders)} folders for dataset: {dataset}...")

    for folder in scenario_folders:
        info = parse_scenario_name(folder.name)
        if info['dataset'] != dataset: continue

        jsonl_files = list(folder.rglob("valuations.jsonl"))

        # We need to track totals PER METRIC for this specific scenario (Defense)
        # Structure: { metric_name: { 'paid_benign': 0.0, 'discarded_benign': 0.0, 'paid_adv': 0.0 } }
        scenario_totals = {m: {'paid_benign': 0.0, 'discarded_benign': 0.0, 'paid_adv': 0.0} for m in target_metrics}
        round_counts = {m: 0 for m in target_metrics}

        for j_file in jsonl_files:
            try:
                with open(j_file, 'r') as f: lines = f.readlines()
                # Analyze converged rounds (last 20%)
                start_idx = max(0, int(len(lines) * 0.8))

                for line in lines[start_idx:]:
                    rec = json.loads(line)
                    selected_ids = set(rec.get('selected_ids', []))
                    valuations = rec.get('seller_valuations', {})

                    # Check which metrics are available in this line
                    # Usually a run has 1 or 2 metrics. We update the ones we find.
                    found_metrics = set()

                    for sid, scores in valuations.items():
                        is_adv = str(sid).startswith('adv')
                        is_selected = sid in selected_ids

                        for m in target_metrics:
                            # Look for keys like 'val_shapley_score' or just 'shapley'
                            # The file might have 'marginal_contrib_loo_score'
                            # We search for the metric name in the keys
                            val = 0.0
                            metric_key_found = False

                            for k, v in scores.items():
                                if m in k and v is not None:
                                    val = max(0, float(v)) # Assume positive payment allocation
                                    metric_key_found = True
                                    break

                            if metric_key_found:
                                found_metrics.add(m)
                                if is_adv:
                                    if is_selected: scenario_totals[m]['paid_adv'] += val
                                else:
                                    if is_selected: scenario_totals[m]['paid_benign'] += val
                                    else:           scenario_totals[m]['discarded_benign'] += val

                    for m in found_metrics:
                        round_counts[m] += 1
            except: pass

        # Add records for this Defense
        for m in target_metrics:
            if round_counts[m] > 0:
                records.append({
                    "defense": info['defense'],
                    "metric": m,
                    "Paid to Benign": scenario_totals[m]['paid_benign'] / round_counts[m],
                    "Discarded Benign": scenario_totals[m]['discarded_benign'] / round_counts[m],
                    "Paid to Adversary": scenario_totals[m]['paid_adv'] / round_counts[m]
                })

    return pd.DataFrame(records)

# ==========================================
# 3. PLOTTING
# ==========================================
def plot_financial_impact(df: pd.DataFrame, metric: str, output_dir: Path):
    print(f"  -> Generating Chart for Metric: {metric}")
    subset = df[df['metric'] == metric].copy()
    if subset.empty:
        print(f"     [Warning] No data found for metric '{metric}'")
        return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask', 'skymask-s']
    defense_order = [d for d in defense_order if d in subset['defense'].unique()]

    # Prepare Data for Stacked Bar
    plot_df = subset.set_index('defense')[['Paid to Benign', 'Discarded Benign', 'Paid to Adversary']]
    plot_df = plot_df.loc[defense_order]

    # Plot
    ax = plot_df.plot(
        kind='bar',
        stacked=True,
        figsize=(9, 6),
        color=[FINANCIAL_PALETTE[x] for x in plot_df.columns],
        width=0.75,
        edgecolor='black'
    )

    # Clean Metric Name for Title
    pretty_name = metric.replace("_", " ").replace("loo", "LOO").replace("contrib", "Contribution").title()
    if "Kernelshap" in pretty_name: pretty_name = "Shapley Value (KernelShap)"

    ax.set_title(f"Financial Impact on {pretty_name}", fontweight='bold', pad=15)
    ax.set_ylabel("Average Total Payment (Per Round)", fontweight='bold')
    ax.set_xlabel("", fontweight='bold')

    # Labels
    labels = [l.get_text().replace("skymask-s", "SkyMask-S").title() for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0, fontweight='bold', fontsize=12)

    # Legend
    plt.legend(title="Budget Allocation", loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.savefig(output_dir / f"Fig3_Impact_{metric}.pdf", bbox_inches='tight')
    plt.close()

# ==========================================
# 4. MAIN
# ==========================================
def main():
    print("--- Generating Financial Impact Analysis ---")
    set_plot_style()
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    # We process CIFAR-100 as the primary benchmark
    dataset = "CIFAR-100"
    df_finance = load_financial_data_by_metric(Path(BASE_RESULTS_DIR), dataset, TARGET_METRICS)

    if df_finance.empty:
        print("❌ No data found. Check directory structure.")
        return

    # 2. Generate One Figure Per Metric
    for metric in TARGET_METRICS:
        plot_financial_impact(df_finance, metric, output_dir)

    print(f"\n✅ Done. Figures saved to {FIGURE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()