import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step2_figures_flexible"
RELATIVE_ACC_THRESHOLD = 0.90

# 1. DEFENSE LABELS (Pretty Names)
DEFENSE_LABELS = {
    'fedavg': 'FedAvg',
    'fltrust': 'FLTrust',
    'martfl': 'MARTFL',
    'skymask': 'SkyMask',
}

# 2. COLOR CONSISTENCY STANDARD
#    (Matches your previous request: Grey, Blue, Green, Red)
DEFENSE_COLORS = {
    "FedAvg": "#7f8c8d",  # Grey
    "FLTrust": "#3498db",  # Blue
    "MARTFL": "#2ecc71",  # Green
    "SkyMask": "#e74c3c",  # Red
}

# 3. PLOT ORDER
DEFENSE_ORDER = [
    'fedavg',
    'fltrust',
    'martfl',
    'skymask',
]


# ==========================================
# --- STYLING HELPER ---
# ==========================================

def set_publication_style():
    """Sets the 'Compact & Bold' professional style globally."""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'axes.titlepad': 6,  # TIGHT PADDING
        'axes.linewidth': 2.0,
        'axes.edgecolor': '#333333',
        'lines.linewidth': 2.5,
    })


# ==========================================
# --- DATA LOADING FUNCTIONS ---
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        pattern = r'step2\.5_find_hps_(?P<defense>.+?)_(?P<modality>image|text|tabular)_(?P<dataset>.+)'
        match = re.search(pattern, scenario_name)
        if match:
            return match.groupdict()
        else:
            return {}
    except Exception:
        return {}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if not report_file.exists():
            run_data['benign_selection_rate'] = np.nan
            run_data['adv_selection_rate'] = np.nan
            return run_data

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {}).values()
        adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
        ben_sellers = [s for s in sellers if s.get('type') == 'benign']

        run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
        run_data['benign_selection_rate'] = np.mean(
            [s['selection_rate'] for s in ben_sellers]) if ben_sellers else np.nan
        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if
                        f.is_dir() and not f.name.endswith("_nolocalclip")]

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if not run_scenario: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue
                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)
                if run_metrics:
                    all_runs.append({**run_scenario, **run_hps, **run_metrics})
            except Exception:
                continue

    if not all_runs:
        print("No run data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # Filter
    df = df[df['defense'].isin(DEFENSE_ORDER)].copy()
    if df.empty: return pd.DataFrame()

    # Calculate Thresholds
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    # --- STANDARDIZATION ---
    # Map raw defense names (e.g., 'skymask') to Pretty Names (e.g., 'SkyMask')
    # This enables the Color Dictionary to work
    df['defense'] = df['defense'].map(lambda x: DEFENSE_LABELS.get(x, x))

    return df


# ==========================================
# --- PLOTTING FUNCTION (COMPACT) ---
# ==========================================

def plot_composite_row(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Composite Row (Compact & Consistent) ---")

    set_publication_style()

    unique_datasets = df['dataset'].unique()
    markers = ['(a)', '(b)', '(c)', '(d)']

    for target_dataset in unique_datasets:
        print(f"   > Processing composite row for: {target_dataset}")

        subset = df[df['dataset'] == target_dataset]

        # Get Current Order using Pretty Names
        pretty_order = [DEFENSE_LABELS.get(d, d) for d in DEFENSE_ORDER]
        current_order = [d for d in pretty_order if d in subset['defense'].unique()]

        if not current_order: continue

        # --- LAYOUT: LOW HEIGHT (2.8) ---
        fig, axes = plt.subplots(1, 4, figsize=(28, 4.8), constrained_layout=True)

        # --- Data Prep ---
        d1 = subset.groupby('defense')['platform_usable'].mean().reindex(current_order).reset_index()
        d1['Value'] = d1['platform_usable'] * 100

        def calc_best_acc_comp(g):
            u = g[g['platform_usable'] == True]
            return u['acc'].mean() if not u.empty else g['acc'].max()

        d2 = subset.groupby('defense').apply(calc_best_acc_comp).reindex(current_order).reset_index(name='acc')
        d2['Value'] = d2['acc'] * 100

        d3 = subset[subset['platform_usable'] == True].groupby('defense')['rounds'].mean().reindex(
            current_order).reset_index()
        d3['Value'] = d3['rounds']

        d4 = subset.groupby('defense')[['benign_selection_rate', 'adv_selection_rate']].mean().reindex(
            current_order).reset_index()
        d4 = d4.melt(id_vars='defense', var_name='Type', value_name='Rate')
        d4['Rate'] *= 100
        d4['Type'] = d4['Type'].replace({'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})

        # --- Plotting ---

        # (a) Usability - Uses DEFENSE_COLORS
        sns.barplot(ax=axes[0], data=d1, x='defense', y='Value', order=current_order,
                    palette=DEFENSE_COLORS, edgecolor='black', linewidth=2)
        axes[0].set_title(f"{markers[0]} Usability Rate (%)", fontweight='bold', fontsize=20)
        axes[0].set_ylim(0, 105)

        # (b) Accuracy - Uses DEFENSE_COLORS
        sns.barplot(ax=axes[1], data=d2, x='defense', y='Value', order=current_order,
                    palette=DEFENSE_COLORS, edgecolor='black', linewidth=2)
        axes[1].set_title(f"{markers[1]} Avg. Usable Acc (%)", fontweight='bold', fontsize=20)
        axes[1].set_ylim(0, 105)

        # Hatching for zero usability
        for i, defense in enumerate(current_order):
            val = d1[d1['defense'] == defense]['Value'].values
            if len(val) > 0 and val[0] == 0:
                axes[1].patches[i].set_hatch('///')
                axes[1].patches[i].set_edgecolor('black')

        # (c) Cost - Uses DEFENSE_COLORS
        sns.barplot(ax=axes[2], data=d3, x='defense', y='Value', order=current_order,
                    palette=DEFENSE_COLORS, edgecolor='black', linewidth=2)
        axes[2].set_title(f"{markers[2]} Avg. Cost (Rounds)", fontweight='bold', fontsize=20)

        # (d) Selection - grouped bars (Uses Green/Red for Benign/Adv)
        sns.barplot(ax=axes[3], data=d4, x='defense', y='Rate', hue='Type', order=current_order,
                    palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor='black', linewidth=2)
        axes[3].set_title(f"{markers[3]} Avg. Selection Rates", fontweight='bold', fontsize=20)
        axes[3].set_ylim(0, 105)

        # --- LEGEND for Plot (d) ---
        # Placed ABOVE the 4th plot to save vertical space inside the tiny plot
        axes[3].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
                       ncol=2, frameon=False, fontsize=14)

        # --- Common Styling ---
        for ax in axes:
            # Bold X-Labels
            ax.set_xticklabels(current_order, fontsize=14, fontweight='bold')
            ax.set_xlabel("")

            # Bold Y-Labels
            ax.tick_params(axis='y', labelsize=14)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')

            ax.grid(axis='y', alpha=0.5, linewidth=1.5)

            # Annotations
            for p in ax.patches:
                h = p.get_height()
                if not np.isnan(h) and h > 0:
                    ax.annotate(f'{h:.0f}',
                                (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom',
                                fontsize=14, fontweight='bold',
                                xytext=(0, 4), textcoords='offset points')

        save_path = output_dir / f"plot_row_combined_{target_dataset}.pdf"
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
        print(f"     Saved composite row to: {save_path}")
        plt.close(fig)


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)
    if not df.empty:
        plot_composite_row(df, output_dir)
    else:
        print("No data found.")


if __name__ == "__main__":
    main()