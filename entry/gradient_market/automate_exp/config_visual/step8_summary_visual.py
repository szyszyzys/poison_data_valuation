import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step8_figures"
TARGET_VICTIM_ID = "bn_5"

# --- Styling Helper ---
def set_plot_style():
    """Sets a consistent professional style with LARGE, BOLD fonts."""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2) # Increased base scale

    # Global updates for boldness and size
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',              # Bold text
        'axes.labelweight': 'bold',         # Bold labels
        'axes.titleweight': 'bold',         # Bold titles
        'axes.titlesize': 20,               # Big titles
        'axes.labelsize': 18,               # Big labels
        'xtick.labelsize': 14,              # Big ticks
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'axes.linewidth': 2,                # Thicker borders
        'lines.linewidth': 2.5,
    })

# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses folder names. Handles BOTH Step 8 (Attacks) and Step 7 (Baseline).
    """
    try:
        # 1. Check for Step 8 (Buyer Attacks)
        pattern_step8 = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match8 = re.search(pattern_step8, scenario_name)
        if match8:
            return {
                "scenario": scenario_name,
                "type": "attack",
                "attack": match8.group(1),
                "defense": match8.group(2),
                "dataset": match8.group(3),
            }

        # 2. Check for Step 7 (Baseline - No Attack)
        pattern_step7 = r'step7_baseline_no_attack_(fedavg|martfl|fltrust|skymask)_(.*)'
        match7 = re.search(pattern_step7, scenario_name)
        if match7:
            return {
                "scenario": scenario_name,
                "type": "baseline",
                "attack": "0. Baseline",
                "defense": match7.group(1),
                "dataset": match7.group(2),
            }

        return {"scenario": scenario_name, "type": "unknown"}
    except Exception as e:
        return {"scenario": scenario_name, "type": "unknown"}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    """Loads performance metrics and per-seller selection rates."""
    run_records = []
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        base_metrics['acc'] = acc
        base_metrics['rounds'] = metrics.get('completed_rounds', 0)
    except Exception:
        return []

    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists():
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})

        found_sellers = False
        for seller_id, seller_data in sellers.items():
            if seller_data.get('type') == 'benign':
                found_sellers = True
                record = base_metrics.copy()
                record['seller_id'] = seller_id
                record['selection_rate'] = seller_data.get('selection_rate', 0.0)
                run_records.append(record)

        if not found_sellers:
            return [base_metrics]

        return run_records
    except Exception:
        return [base_metrics]


def collect_data(base_dir: str) -> pd.DataFrame:
    all_records = []
    base_path = Path(base_dir)

    # Look for BOTH Step 8 and Step 7 folders
    folders = [f for f in base_path.glob("step8_buyer_attack_*") if f.is_dir()]
    folders += [f for f in base_path.glob("step7_baseline_no_attack_*") if f.is_dir()]

    print(f"Found {len(folders)} scenario directories (Step 8 + Step 7).")

    for scenario_path in folders:
        run_info = parse_scenario_name(scenario_path.name)
        if run_info.get("type") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            records = load_run_data(metrics_file)
            for r in records:
                all_records.append({**run_info, **r})

    return pd.DataFrame(all_records)


def get_baseline_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    baseline_df = df[df['attack'] == '0. Baseline']
    if baseline_df.empty:
        print("⚠️ No Step 7 Baseline data found. Plots will rely only on Step 8 data.")
        return {}

    lookup = baseline_df.groupby(['defense', 'dataset'])['selection_rate'].mean().to_dict()
    print(f"✅ Baseline calculated for {len(lookup)} configs.")
    return lookup


def plot_buyer_attack_distribution(df: pd.DataFrame, baseline_lookup: Dict, output_dir: Path):
    print("\n--- Plotting Selection Distributions (Fig 1) ---")
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    step8_attacks = [a for a in df['attack'].unique()
                     if 'pivot' not in str(a) and a != '0. Baseline']

    for attack in step8_attacks:
        attack_df = df[df['attack'] == attack]
        if attack_df.empty: continue

        dataset = attack_df['dataset'].iloc[0] if 'dataset' in attack_df.columns else "Unknown"

        # Explicitly create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6)) # Slightly larger for readability

        # Plot
        sns.boxplot(
            data=attack_df,
            x='defense',
            y='selection_rate',
            order=defense_order,
            palette="viridis",
            hue='defense',
            legend=False,
            ax=ax
        )

        # Baseline Lines
        for i, defense in enumerate(defense_order):
            base_val = baseline_lookup.get((defense, dataset))
            if base_val is not None:
                ax.hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4,
                          color='red', linestyle='--', lw=3, # Thicker line
                          label='Healthy Baseline' if i == 0 else "")

        # Legend
        if any(baseline_lookup.get((d, dataset)) for d in defense_order):
            handles, labels = ax.get_legend_handles_labels()
            baseline_handles = [h for h, l in zip(handles, labels) if "Baseline" in l]
            baseline_labels = [l for l in labels if "Baseline" in l]
            if baseline_handles:
                ax.legend(baseline_handles[:1], baseline_labels[:1], loc='best', fontsize=14)

        # BOLD TITLES AND LABELS
        ax.set_title(f'Impact on Seller Selection\nAttack: {attack}', fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel("Selection Rate", fontsize=18, fontweight='bold', labelpad=10)
        ax.set_xlabel("Defense Strategy", fontsize=18, fontweight='bold', labelpad=10)

        # Bigger Ticks
        ax.tick_params(axis='both', which='major', labelsize=14)

        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        fname = output_dir / f"Step8_SELECTION_{attack}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def plot_targeted_attack_breakdown(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Targeted Breakdown (Fig 1.5) ---")
    pivot_df = df[df['attack'].str.contains("pivot", case=False)].copy()

    if pivot_df.empty: return

    pivot_df['Status'] = pivot_df['seller_id'].apply(
        lambda x: 'Victim (bn_5)' if str(x) == TARGET_VICTIM_ID else 'Other Benign'
    )

    plt.figure(figsize=(9, 7)) # Larger figure
    ax = sns.barplot(
        data=pivot_df, x='defense', y='selection_rate', hue='Status',
        order=['fedavg', 'fltrust', 'martfl', 'skymask'],
        palette={'Victim (bn_5)': '#e74c3c', 'Other Benign': '#95a5a6'},
        errorbar='sd'
    )

    # BOLD TITLES AND LABELS
    plt.title("Targeted Exclusion Success (Orthogonal Pivot)", fontsize=20, fontweight='bold', pad=15)
    plt.ylabel("Selection Rate", fontsize=18, fontweight='bold', labelpad=10)
    plt.xlabel("Defense Strategy", fontsize=18, fontweight='bold', labelpad=10)

    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, title_fontsize=16)

    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    fname = output_dir / "Step8_SELECTION_TARGETED_PIVOT.pdf"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname.name}")


def plot_buyer_attack_performance(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Performance Metrics (Fig 2) ---")

    run_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds', 'attack', 'defense'])
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    attacks = [a for a in run_df['attack'].unique() if a != '0. Baseline']
    attacks.sort()

    baseline_df = run_df[run_df['attack'] == '0. Baseline']

    for attack in attacks:
        current_attack_df = run_df[run_df['attack'] == attack]

        combined_df = pd.concat([baseline_df, current_attack_df], ignore_index=True)
        if combined_df.empty: continue

        melted = combined_df.melt(
            id_vars=['attack', 'defense'],
            value_vars=['acc', 'rounds'],
            var_name='MetricKey', value_name='Value'
        )
        melted['Metric'] = melted['MetricKey'].map({'acc': 'Accuracy', 'rounds': 'Rounds'})

        g = sns.catplot(
            data=melted, x='defense', y='Value',
            hue='attack',
            col='Metric', kind='bar',
            order=defense_order,
            height=5, aspect=1.3, sharey=False, # Larger aspect
            palette={'0. Baseline': 'grey', attack: 'red'}
        )

        # BOLD TITLES AND LABELS
        g.fig.suptitle(f'Marketplace Damage Assessment\nAttack: {attack}', y=1.05, fontsize=22, fontweight='bold')
        g.set_axis_labels("Defense Strategy", "Value", fontsize=18, fontweight='bold')
        g.set_titles("{col_name}", size=18, fontweight='bold')

        for ax in g.axes.flat:
            ax.tick_params(labelsize=14)

        fname = output_dir / f"Step8_PERFORMANCE_{attack}.pdf"
        g.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname.name}")


def main():
    set_plot_style() # Apply global style
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots saved to: {output_dir.resolve()}")

    df = collect_data(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    baseline_lookup = get_baseline_lookup(df)

    plot_buyer_attack_distribution(df, baseline_lookup, output_dir)
    plot_targeted_attack_breakdown(df, output_dir)
    plot_buyer_attack_performance(df, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()