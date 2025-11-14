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

# Define the relative 'usability' threshold
RELATIVE_ACC_THRESHOLD = 0.90


# ---------------------


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'opt_Adam_lr_0.001_epochs_2')
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
    """
    Parses the base scenario name (e.g., 'step2.5_find_hps_martfl_image_CIFAR10')
    """
    try:
        # Use the regex parser to avoid errors with underscores
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask)_(image|text|tabular)_(.+)'
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
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads key data from final_metrics.json
    """
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        return run_data
    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

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
                    })
            except Exception as e:
                print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # --- THIS IS THE KEY NEW LOGIC ---
    print("\nCalculating relative 'Platform Usable' thresholds...")

    # 1. Find the "gold standard" (max acc) for each dataset
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()

    # 2. Create a new column in the df for this max_acc
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)

    # 3. Define the usability threshold for *each row* relative to its dataset's max
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD

    # 4. Define "Usable" from the platform's (accuracy-only) perspective
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    print("Done calculating thresholds.")

    return df


def plot_platform_usability_metrics(df: pd.DataFrame, output_dir: Path):
    """
    (REWRITTEN)
    Plots the 3 key platform metrics in *separate, more readable files*.
    """
    print("\n--- Plotting Platform Usability Metrics ---")

    if df.empty:
        print("No data to plot.")
        return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # 1. Usability Rate
    print("  Plotting: 1. Usability Rate")
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100  # Convert to %

    g = sns.catplot(
        data=df_usability, kind='bar', x='defense', y='Value',
        col='dataset', col_wrap=3, height=4, aspect=1.1,
        order=defense_order, palette='viridis'
    )
    g.fig.suptitle(f"1. Usability Rate (What % of HPs achieved >{RELATIVE_ACC_THRESHOLD * 100}% of max accuracy?)",
                   y=1.05)
    g.set_axis_labels("Defense", "Usable HP Combinations (%)")
    g.set_titles(col_template="{col_name}")
    g.map_dataframe(annotate_bars)
    plot_file = output_dir / "plot_platform_1_USABILITY_RATE.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf();
    plt.close('all')

    # 2. Average Performance (of *usable* runs)
    print("  Plotting: 2. Average Usable Accuracy")
    df_perf = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['acc'].mean().reset_index()
    df_perf['Value'] = df_perf['acc'] * 100  # Convert to %

    g = sns.catplot(
        data=df_perf, kind='bar', x='defense', y='Value',
        col='dataset', col_wrap=3, height=4, aspect=1.1,
        order=defense_order, palette='Greens_d'
    )
    g.fig.suptitle(f"2. Avg. Usable Accuracy (Mean accuracy of *only* the usable runs)", y=1.05)
    g.set_axis_labels("Defense", "Average Accuracy (%)")
    g.set_titles(col_template="{col_name}")
    g.map_dataframe(annotate_bars)
    plot_file = output_dir / "plot_platform_2_AVG_ACCURACY.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf();
    plt.close('all')

    # 3. Stability (Std Dev of *all* runs)
    print("  Plotting: 3. Accuracy Instability")
    df_stability = df.groupby(['defense', 'dataset'])['acc'].std().reset_index()
    df_stability['Value'] = df_stability['acc'] * 100  # Convert to %

    g = sns.catplot(
        data=df_stability, kind='bar', x='defense', y='Value',
        col='dataset', col_wrap=3, height=4, aspect=1.1,
        order=defense_order, palette='Reds_d'
    )
    g.fig.suptitle(f"3. Accuracy Instability (Std. Dev. of acc across *all* HP sweeps) - Lower is Better", y=1.05)
    g.set_axis_labels("Defense", "Accuracy Std. Dev. (pts)")
    g.set_titles(col_template="{col_name}")
    g.map_dataframe(annotate_bars)
    plot_file = output_dir / "plot_platform_3_INSTABILITY.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf();
    plt.close('all')

    # --- Save the analysis CSV ---
    print("  Saving summary CSV...")
    csv_output_path = output_dir / "step2.5_platform_metrics_summary.csv"
    try:
        # Combine all 3 metrics into one DataFrame
        df_usability['Metric'] = '1. Usability Rate (%)'
        df_perf['Metric'] = '2. Avg. Usable Accuracy (%)'
        df_stability['Metric'] = '3. Accuracy Instability (Std Dev)'
        df_final = pd.concat([df_usability, df_perf, df_stability], ignore_index=True)
        # Pivot for a wide, readable CSV
        df_pivot = df_final.pivot_table(index=['dataset', 'defense'], columns='Metric', values='Value')
        df_pivot.to_csv(csv_output_path, float_format="%.2f")
        print(f"\n✅ Successfully saved platform metrics summary to: {csv_output_path}\n")
    except Exception as e:
        print(f"Could not save CSV: {e}")
    # -----------------------------


def annotate_bars(data, **kwargs):
    """Helper function to add text labels to bars."""
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=9)


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # --- Call the new plotter ---
    plot_platform_usability_metrics(df, output_dir)
    # ----------------------------

    print("\nAnalysis complete. Check 'step2.5_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()