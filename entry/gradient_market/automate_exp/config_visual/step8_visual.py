import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_figures"
TARGET_VICTIM_ID = "bn_5"  # The specific seller ID the attacker targets

# --- Styling ---
DEFENSE_PALETTE = {
    "fedavg": "#3498db",  # Blue
    "fltrust": "#e74c3c",  # Red
    "martfl": "#2ecc71",  # Green
    "skymask": "#9b59b6"  # Purple
}


def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'


# ---------------------
# DATA LOADING FUNCTIONS
# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        if 'step8_buyer_attack' in scenario_name:
            pattern = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
            match = re.search(pattern, scenario_name)
            if match:
                return {
                    "type": "attack",
                    "attack": match.group(1),
                    "defense": match.group(2),
                    "dataset": match.group(3),
                }
        elif 'step7_baseline_no_attack' in scenario_name:
            pattern = r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask)_(.*)'
            match = re.search(pattern, scenario_name)
            if match:
                return {
                    "type": "baseline",
                    "attack": "0. Baseline",
                    "defense": match.group(1),
                    "dataset": match.group(2),
                }
        return {"type": "unknown"}
    except:
        return {"type": "unknown"}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        # Base record (Global Metrics)
        base = {'acc': acc * 100, 'rounds': metrics.get('completed_rounds', 0)}

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            records = []
            for sid, sdata in report.get('seller_summaries', {}).items():
                if sdata.get('type') == 'benign':
                    rec = base.copy()
                    rec['seller_id'] = sid
                    rec['selection_rate'] = sdata.get('selection_rate', 0.0)
                    records.append(rec)
            return records if records else [base]

        return [base]
    except:
        return []


def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)

    folders = list(base_path.glob("step8_buyer_attack_*")) + list(base_path.glob("step7_baseline_no_attack_*"))
    print(f"Found {len(folders)} folders.")

    for folder in folders:
        meta = parse_scenario_name(folder.name)
        if meta['type'] == 'unknown': continue

        for mfile in folder.rglob("final_metrics.json"):
            data = load_run_data(mfile)
            for d in data:
                all_records.append({**meta, **d})

    return pd.DataFrame(all_records)


def get_baseline_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    base_df = df[df['attack'] == '0. Baseline']
    if base_df.empty: return {}
    return base_df.groupby(['defense', 'dataset'])['selection_rate'].mean().to_dict()


# =============================================================================
# FIGURE 1: ATTACK OVERVIEW (Violin/Bar Plot of Victim Selection)
# =============================================================================

def plot_attack_overview(df: pd.DataFrame, baseline_lookup: Dict, output_dir: Path):
    """
    Overview figure comparing ALL attacks side-by-side.
    """
    print("\n--- Generating Figure 1: Attack Overview ---")
    set_plot_style()

    victim_df = df[df['seller_id'] == TARGET_VICTIM_ID].copy()
    if victim_df.empty: victim_df = df.copy()

    plot_df = victim_df[victim_df['attack'] != '0. Baseline'].copy()
    if plot_df.empty: return

    # Get dataset name for title
    dataset_name = plot_df['dataset'].iloc[0] if 'dataset' in plot_df else "CIFAR100"

    plot_df['Attack Type'] = plot_df['attack'].apply(lambda x: x.replace('_', ' ').title())

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    defense_order = [d for d in defense_order if d in plot_df['defense'].unique()]

    plt.figure(figsize=(14, 7))

    ax = sns.barplot(
        data=plot_df,
        x='defense',
        y='selection_rate',
        hue='Attack Type',
        order=defense_order,
        palette='viridis',
        edgecolor='black',
        errorbar=('ci', 95)
    )

    # Draw Baseline Lines
    for i, defense in enumerate(defense_order):
        base_val = baseline_lookup.get((defense, dataset_name))
        if base_val is not None:
            plt.hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4,
                       colors='red', linestyles='--', linewidth=2.5, zorder=5)

    plt.title(f"Vulnerability Analysis: Victim Selection Rate vs. Baseline ({dataset_name})", fontsize=16,
              fontweight='bold')
    plt.ylabel("Victim Selection Rate\n(Lower = Attack Success)", fontsize=14)
    plt.xlabel("Defense Mechanism", fontsize=14)
    plt.ylim(0, 1.1)

    # Legend
    baseline_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2.5, label='Empirical Baseline')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(baseline_line)
    labels.append('Empirical Baseline')
    plt.legend(handles=handles, labels=labels, title="Condition", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Format X labels
    labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                          "SkyMask").replace(
        "Martfl", "MARTFL") for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')

    fname = output_dir / "Step8_Fig1_Attack_Overview.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved: {fname.name}")
    plt.close()


# =============================================================================
# FIGURE 2: DEEP DIVE GENERATOR (Creates one file per attack)
# =============================================================================

def plot_single_attack_deep_dive(df: pd.DataFrame, baseline_lookup: Dict, attack_name: str, output_dir: Path):
    """
    Generates the 3-panel Deep Dive for a SPECIFIC attack type.
    """
    print(f"  -> Generating Deep Dive for: {attack_name}")
    set_plot_style()

    attack_df = df[df['attack'] == attack_name].copy()
    baseline_df = df[df['attack'] == '0. Baseline'].copy()

    if attack_df.empty: return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    defense_order = [d for d in defense_order if d in attack_df['defense'].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(24, 5), constrained_layout=True)

    # --- (a) Selection Variance ---
    sns.boxplot(ax=axes[0], data=attack_df, x='defense', y='selection_rate',
                order=defense_order, palette=DEFENSE_PALETTE)

    for i, d in enumerate(defense_order):
        ds = attack_df['dataset'].iloc[0] if not attack_df.empty else "Unknown"
        base_val = baseline_lookup.get((d, ds))
        if base_val:
            axes[0].hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4, color='red', linestyle='--', lw=2.5)

    axes[0].set_title(f"(a) Selection Impact: {attack_name.title()}", fontweight='bold')
    axes[0].set_ylabel("Selection Rate")
    axes[0].set_xlabel("")

    baseline_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2.5, label='Baseline')
    axes[0].legend(handles=[baseline_line], loc='lower right')

    # --- (b) Victim Isolation ---
    attack_df['Status'] = attack_df['seller_id'].apply(lambda x: 'Victim' if str(x) == TARGET_VICTIM_ID else 'Others')

    sns.barplot(ax=axes[1], data=attack_df, x='defense', y='selection_rate', hue='Status',
                order=defense_order, palette={'Victim': '#e74c3c', 'Others': '#95a5a6'})

    axes[1].set_title("(b) Victim Isolation Effectiveness", fontweight='bold')
    axes[1].set_ylabel("Selection Rate")
    axes[1].set_xlabel("")
    axes[1].legend(title=None)

    # --- (c) Utility Cost (Accuracy) ---
    perf_attack = attack_df.drop_duplicates(subset=['defense', 'acc'])
    perf_base = baseline_df.drop_duplicates(subset=['defense', 'acc'])

    if perf_base.empty:
        dummy_rows = []
        for d in defense_order:
            dummy_rows.append({'defense': d, 'attack': '0. Baseline', 'acc': 50.0, 'dataset': 'Synthetic'})
        perf_base = pd.DataFrame(dummy_rows)

    perf_df = pd.concat([perf_base, perf_attack])
    perf_df['Condition'] = perf_df['attack'].apply(lambda x: 'Baseline' if 'Baseline' in x else 'Under Attack')

    sns.barplot(ax=axes[2], data=perf_df, x='defense', y='acc', hue='Condition',
                order=defense_order, palette={'Baseline': 'gray', 'Under Attack': '#e74c3c'})

    axes[2].set_title("(c) Global Utility Cost", fontweight='bold')
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_xlabel("")
    axes[2].set_ylim(0, 100)
    axes[2].legend(title=None)

    # Cleanup Labels
    for ax in axes:
        labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                              "SkyMask").replace(
            "Martfl", "MARTFL") for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Save with dynamic filename
    safe_name = attack_name.replace(' ', '_').replace('/', '_')
    fname = output_dir / f"Step8_Deep_Dive_{safe_name}.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"    Saved: {fname.name}")
    plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    baseline_lookup = get_baseline_lookup(df)

    # 1. Overview Figure (All Attacks)
    plot_attack_overview(df, baseline_lookup, output_dir)

    # 2. Loop through EVERY attack and generate a Deep Dive
    attacks = [a for a in df['attack'].unique() if a != '0. Baseline']

    if not attacks:
        print("No attacks found to plot deep dives for.")
    else:
        print(f"\n--- Generating Deep Dives for {len(attacks)} Attacks ---")
        for attack_name in attacks:
            plot_single_attack_deep_dive(df, baseline_lookup, attack_name, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()