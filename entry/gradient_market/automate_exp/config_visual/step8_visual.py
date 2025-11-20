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
FIGURE_OUTPUT_DIR = "./figures/step8_figures_new"
TARGET_VICTIM_ID = "bn_5"

# --- Styling ---
DEFENSE_PALETTE = {
    "fedavg": "#3498db",  # Blue
    "fltrust": "#e74c3c",  # Red
    "martfl": "#2ecc71",  # Green
    "skymask": "#9b59b6"  # Purple
}
ATTACK_PALETTE = {
    "0. Baseline": "gray",
    "Targeted": "#e74c3c"  # Red for attack
}


def set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'


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

        # Normalize metrics
        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0
        base = {'acc': acc * 100, 'rounds': metrics.get('completed_rounds', 0)}  # Convert to %

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


# --- NEW COMBINED FIGURE FUNCTION ---

def plot_combined_figure(df: pd.DataFrame, baseline_lookup: Dict, output_dir: Path):
    """
    Generates a single, high-quality figure for the paper with 3 panels:
    (a) General Selection Impact (Boxplot)
    (b) Targeted Victim Isolation (Barplot)
    (c) Performance Degradation (Barplot)
    """
    print("\n--- Generating Combined Paper Figure ---")
    set_plot_style()

    # Filter relevant data
    # We assume we want to show the "Pivot" attack as the representative case
    pivot_attack_name = [a for a in df['attack'].unique() if 'pivot' in str(a).lower()]
    if not pivot_attack_name:
        print("No pivot attack data found for combined plot.")
        return

    attack_name = pivot_attack_name[0]
    attack_df = df[df['attack'] == attack_name].copy()
    baseline_df = df[df['attack'] == '0. Baseline'].copy()

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 5), constrained_layout=True)

    # --- Panel A: Selection Distribution ---
    sns.boxplot(
        ax=axes[0], data=attack_df, x='defense', y='selection_rate',
        order=defense_order, palette=DEFENSE_PALETTE
    )

    # Add Baseline Lines
    for i, defense in enumerate(defense_order):
        # Assume first dataset found is representative
        ds = attack_df['dataset'].iloc[0] if not attack_df.empty else "Unknown"
        base_val = baseline_lookup.get((defense, ds))
        if base_val is not None:
            axes[0].hlines(y=base_val, xmin=i - 0.4, xmax=i + 0.4, color='red', linestyle='--', lw=2.5)

    axes[0].set_title("(a) General Selection Impact", fontweight='bold')
    axes[0].set_ylabel("Selection Rate")
    axes[0].set_xlabel("")
    axes[0].set_ylim(-0.05, 1.05)

    # Legend hack for baseline line
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2.5, linestyle='--')]
    axes[0].legend(custom_lines, ['Healthy Baseline'], loc='lower right')

    # --- Panel B: Targeted Victim Isolation ---
    # Compare Victim vs Others within the attack data
    attack_df['Status'] = attack_df['seller_id'].apply(
        lambda x: 'Victim' if str(x) == TARGET_VICTIM_ID else 'Others'
    )

    sns.barplot(
        ax=axes[1], data=attack_df, x='defense', y='selection_rate', hue='Status',
        order=defense_order, palette={'Victim': '#e74c3c', 'Others': '#95a5a6'},
        errorbar=('ci', 95)
    )

    axes[1].set_title("(b) Targeted Victim Isolation", fontweight='bold')
    axes[1].set_ylabel("Selection Rate")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(title=None, loc='upper right')

    # --- Panel C: Performance Degradation ---
    # Combine Baseline and Attack for Side-by-Side
    # We only care about Accuracy for this summary plot
    perf_df = pd.concat([baseline_df, attack_df])
    # Deduplicate scenario runs (we don't need per-seller rows for accuracy)
    perf_df = perf_df.drop_duplicates(subset=['defense', 'attack', 'acc'])

    # Rename for legend
    perf_df['Condition'] = perf_df['attack'].apply(lambda x: 'Baseline' if 'Baseline' in x else 'Under Attack')

    sns.barplot(
        ax=axes[2], data=perf_df, x='defense', y='acc', hue='Condition',
        order=defense_order, palette={'Baseline': 'gray', 'Under Attack': '#e74c3c'}
    )

    axes[2].set_title("(c) Global Model Accuracy Impact", fontweight='bold')
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_xlabel("")
    axes[2].set_ylim(0, 105)  # Assuming %
    axes[2].legend(title=None, loc='lower right')

    # --- Final Touches ---
    # Clean X Labels
    for ax in axes:
        labels = [l.get_text().capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask",
                                                                                                              "SkyMask").replace(
            "Martfl", "MARTFL") for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Save
    fname = output_dir / "Step8_Combined_Paper_Figure.pdf"
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    print(f"  Saved Combined Figure: {fname.name}")
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

    # Generate the specific combined figure
    plot_combined_figure(df, baseline_lookup, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()