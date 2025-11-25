import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
# New output directory for this specific comparison
FIGURE_OUTPUT_DIR = "./figures/skymask_comparison"

# Define the relative 'usability' threshold
RELATIVE_ACC_THRESHOLD = 0.90
# ---------------------

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
        # Matches both 'skymask' and 'skymask_small'
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask_small|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3),
            }
        else:
            raise ValueError(f"Pattern not matched for: {scenario_name}")
    except Exception as e:
        return {"scenario": scenario_name}

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
        run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else np.nan
        return run_data
    except Exception as e:
        return {}

def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir() and not f.name.endswith("_nolocalclip")]

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)
        if "defense" not in run_scenario: continue

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
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # --- CRITICAL CHANGE: NO SWAP LOGIC ---
    # We deliberately keep both 'skymask' and 'skymask_small'
    # so we can compare them.
    # --------------------------------------

    print("Calculating thresholds...")
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    return df

def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)

def print_text_comparison(df: pd.DataFrame):
    """Prints a numeric summary to console."""
    print("\n" + "="*50)
    print("NUMERIC COMPARISON: SkyMask vs SkyMask_Small")
    print("="*50)

    target_defenses = ['skymask', 'skymask_small']
    df_filtered = df[df['defense'].isin(target_defenses)].copy()

    if df_filtered.empty:
        print("No SkyMask data found.")
        return

    datasets = df_filtered['dataset'].unique()

    for ds in datasets:
        print(f"\nDataset: {ds}")
        subset = df_filtered[df_filtered['dataset'] == ds]

        # Calculate means
        means = subset.groupby('defense')[['acc', 'rounds', 'adv_selection_rate']].mean()

        if 'skymask' not in means.index or 'skymask_small' not in means.index:
            print("  -> Missing one of the pair, cannot compare.")
            continue

        orig = means.loc['skymask']
        small = means.loc['skymask_small']

        # Accuracy Delta
        acc_diff = (small['acc'] - orig['acc']) * 100
        print(f"  Accuracy:  Original {orig['acc']:.4f} -> Small {small['acc']:.4f} | Diff: {acc_diff:+.2f}%")

        # Rounds Delta
        rounds_diff = small['rounds'] - orig['rounds']
        print(f"  Rounds:    Original {orig['rounds']:.1f} -> Small {small['rounds']:.1f} | Diff: {rounds_diff:+.1f}")

        # Security Delta
        adv_diff = (small['adv_selection_rate'] - orig['adv_selection_rate']) * 100
        print(f"  Adv Rate:  Original {orig['adv_selection_rate']:.2f} -> Small {small['adv_selection_rate']:.2f} | Diff: {adv_diff:+.2f}%")

def plot_direct_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Plots ONLY SkyMask vs SkyMask_Small side-by-side for clear comparison.
    """
    print("\n--- Plotting Direct Comparison ---")

    # 1. Filter Data
    target_defenses = ['skymask', 'skymask_small']
    df_comp = df[df['defense'].isin(target_defenses)].copy()

    if df_comp.empty:
        print("No SkyMask data found to compare.")
        return

    # 2. Setup Plotting
    sns.set_theme(style="whitegrid", context="talk")
    datasets = df_comp['dataset'].unique()

    defense_labels = {
        'skymask': 'Original',
        'skymask_small': 'Small'
    }

    # Define colors: Grey for Original, Blue for Small (highlighting the new one)
    palette = {'skymask': '#95a5a6', 'skymask_small': '#3498db'}

    # 3. Create Plots for each Dataset
    for ds in datasets:
        ds_data = df_comp[df_comp['dataset'] == ds]

        # Prepare aggregated data
        # Accuracy
        acc_data = ds_data.groupby('defense')['acc'].mean().reset_index()
        acc_data['acc'] = acc_data['acc'] * 100

        # Rounds
        round_data = ds_data.groupby('defense')['rounds'].mean().reset_index()

        # Adv Selection
        adv_data = ds_data.groupby('defense')['adv_selection_rate'].mean().reset_index()
        adv_data['adv_selection_rate'] = adv_data['adv_selection_rate'] * 100

        # Create 1 row, 3 columns figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        order = ['skymask', 'skymask_small']

        # --- Plot A: Accuracy ---
        sns.barplot(data=acc_data, x='defense', y='acc', order=order, palette=palette, ax=axes[0], edgecolor='black')
        style_axis(axes[0], "Accuracy", "", "Accuracy (%)")
        axes[0].set_ylim(0, 105)

        # --- Plot B: Rounds (Cost) ---
        sns.barplot(data=round_data, x='defense', y='rounds', order=order, palette=palette, ax=axes[1], edgecolor='black')
        style_axis(axes[1], "Communication Cost", "", "Rounds")

        # --- Plot C: Adversary Selection ---
        sns.barplot(data=adv_data, x='defense', y='adv_selection_rate', order=order, palette=palette, ax=axes[2], edgecolor='black')
        style_axis(axes[2], "Adversary Selection", "", "Selection Rate (%)")
        axes[2].set_ylim(0, 105)

        # Common X-Axis Labels
        for ax in axes:
            ax.set_xticklabels([defense_labels[t.get_text()] for t in ax.get_xticklabels()])

            # Annotate
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f'{h:.1f}', (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 5), textcoords='offset points')

        plt.suptitle(f"SkyMask Versions Comparison - {ds}", fontsize=22, fontweight='bold', y=1.05)
        plt.tight_layout()

        outfile = output_dir / f"compare_{ds}_skymask_vs_small.pdf"
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        print(f"  Saved comparison for {ds} -> {outfile}")
        plt.close()

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect Data
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # 2. Print Numeric Report (Text)
    print_text_comparison(df)

    # 3. Generate Visual Comparison (PDFs)
    plot_direct_comparison(df, output_dir)

    print(f"\nDone. Check the '{FIGURE_OUTPUT_DIR}' folder for graphs.")

if __name__ == "__main__":
    main()