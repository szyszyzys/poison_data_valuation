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
FIGURE_OUTPUT_DIR = "./step2.5_figures"

# Set your minimum acceptable accuracy threshold (e.g., 0.70 for 70%)
REASONABLE_ACC_THRESHOLD = 0.70

# Set your minimum acceptable Benign Selection Rate (e.g., 0.50 for 50%)
REASONABLE_BSR_THRESHOLD = 0.50


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    (NEW) Parses the HP suffix folder name (e.g., 'opt_Adam_lr_0.001_epochs_2')
    """
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    else:
        print(f"Warning: Could not parse HP suffix '{hp_folder_name}'")
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step2.5_find_hps_martfl_image_CIFAR10')"""
    try:
        parts = scenario_name.split('_')
        return {
            "scenario": scenario_name,
            "defense": parts[3],
            "modality": parts[4],
            "dataset": parts[5],
        }
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads key data from final_metrics.json and marketplace_report.json
    """
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            if adv_sellers:
                run_data['adv_selection_rate'] = pd.Series([s['selection_rate'] for s in adv_sellers]).mean()
            if ben_sellers:
                run_data['benign_selection_rate'] = pd.Series([s['selection_rate'] for s in ben_sellers]).mean()

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    # Look for 'step2.5_find_hps_*' directories
    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step2.5_find_hps_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts:
                    continue

                hp_folder_name = relative_parts[0]

                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics,
                        "hp_suffix": hp_folder_name
                    })
            except Exception as e:
                print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_hp_sensitivity(df: pd.DataFrame, scenario: str, output_dir: Path):
    """
    (NEW) Generates heatmap plots for HP sensitivity (Objective 1 & 3).
    """
    scenario_df = df[df['scenario'] == scenario].copy()
    if scenario_df.empty:
        return

    print(f"\n--- Visualizing Sensitivity: {scenario} ---")

    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    # Filter out metrics that weren't successfully loaded (e.g., if marketplace report was missing)
    metrics_to_plot = [m for m in metrics_to_plot if m in scenario_df.columns]

    for metric in metrics_to_plot:
        # Check if there's any data for this metric
        if scenario_df[metric].isnull().all():
            print(f"Skipping heatmap for {metric} (no data).")
            continue

        g = sns.catplot(
            data=scenario_df,
            x='learning_rate',
            y='local_epochs',
            col='optimizer',
            kind='heatmap',
            height=4,
            aspect=1.2,
            # Create a pivot table for the heatmap values
            # Using median is more robust to a single bad seed than mean
            pivot_kws={'values': metric, 'aggfunc': 'median'},
            annot=True,  # Show the values in the cells
            fmt=".3f"  # Format to 3 decimal places
        )

        g.fig.suptitle(f'HP Sensitivity for: {metric}\n(Scenario: {scenario})', y=1.05)
        g.set_axis_labels("Learning Rate", "Local Epochs")

        plot_file = output_dir / f"plot_{scenario}_{metric.upper()}_heatmap.png"
        g.fig.savefig(plot_file)
        print(f"Saved plot: {plot_file}")
        plt.clf()


def plot_best_defense_comparison(df_best_by_defense: pd.DataFrame, dataset: str, output_dir: Path):
    """
    (NEW) Generates a bar plot comparing the *best* run from each defense (Objective 2).
    """
    dataset_df = df_best_by_defense[df_best_by_defense['dataset'] == dataset].copy()
    if dataset_df.empty:
        return

    print(f"\n--- Visualizing Best-of Comparison: {dataset} ---")

    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    metrics_to_plot = [m for m in metrics_to_plot if m in dataset_df.columns]

    plot_df = dataset_df.melt(
        id_vars=['defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x='defense',
        y='Value',
        hue='Metric'
    )
    plt.title(f'Best-of-Defense Performance Comparison (Default HPs)\nDataset: {dataset}')
    plt.ylabel('Rate')
    plt.xlabel('Defense')
    plt.legend(title='Metric')
    plt.tight_layout()
    plot_file = output_dir / f"plot_{dataset}_defense_comparison.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        return

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Analysis for Objective 1 & 3: Easiness and Selection Patterns ---
    print("\n" + "=" * 80)
    print("      Objective 1 & 3: Finding Best HPs & Assessing Sensitivity")
    print("=" * 80)

    # 1. Apply filters
    reasonable_acc_df = df[df['acc'] >= REASONABLE_ACC_THRESHOLD].copy()

    if 'benign_selection_rate' in reasonable_acc_df.columns:
        reasonable_final_df = reasonable_acc_df[
            reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD
            ].copy()
    else:
        print("\n!WARNING: 'benign_selection_rate' not found. Skipping Fairness filter.")
        reasonable_final_df = reasonable_acc_df.copy()

    # 2. Create sort metric
    reasonable_final_df['sort_metric'] = np.where(
        # For this step, we assume all attacks are backdoor (as per the generator)
        # If you add labelflip, this logic would need to check 'attack' column
        reasonable_final_df['acc'] > 0,  # Placeholder for 'is_backdoor'
        reasonable_final_df['asr'],  # Low is good
        1.0 - reasonable_final_df['acc']  # Low is good (high acc)
    )

    # 3. Sort to find the *best* HP set for each scenario
    sort_columns = ['scenario', 'sort_metric']
    if 'adv_selection_rate' in reasonable_final_df.columns:
        sort_columns.append('adv_selection_rate')

    df_sorted = reasonable_final_df.sort_values(
        by=sort_columns,
        ascending=[True, True, True]
    )

    # This df contains the single best run for each scenario
    df_best_by_defense = df_sorted.drop_duplicates(subset='scenario', keep='first')

    print(
        f"\n--- Best Training HP for each Defense/Dataset (acc >= {REASONABLE_ACC_THRESHOLD}, benign_select >= {REASONABLE_BSR_THRESHOLD}) ---")
    print(f"--- Sorted by: 1. Low ASR, 2. Low Adv. Selection ---")

    cols_to_show = [
        'scenario',
        'hp_suffix',
        'acc', 'asr',
        'adv_selection_rate', 'benign_selection_rate',
        'rounds'
    ]
    cols_present = [c for c in df_best_by_defense.columns if c in cols_to_show]
    print(df_best_by_defense[cols_present].to_string(index=False, float_format="%.4f"))

    # --- Analysis for Objective 2: Best Performance Comparison ---
    print("\n" + "=" * 80)
    print("           Objective 2: Best-of-Defense Comparison (Plots)")
    print("=" * 80)

    for dataset in df_best_by_defense['dataset'].unique():
        plot_best_defense_comparison(df_best_by_defense, dataset, output_dir)

    # --- Analysis for Objective 1 & 3: Sensitivity (Plots) ---
    print("\n" + "=" * 80)
    print("           Objective 1 & 3: HP Sensitivity Heatmaps (Plots)")
    print("=" * 80)

    # We use the *original* full dataframe (df) for plotting sensitivity
    for scenario in df['scenario'].unique():
        plot_hp_sensitivity(df, scenario, output_dir)

    print("\nAnalysis complete. Check 'step2.5_figures' folder for plots.")


if __name__ == "__main__":
    main()