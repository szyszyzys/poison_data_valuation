import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def plot_platform_usability_metrics_separate_pdfs(df: pd.DataFrame, output_dir: Path):
    """
    (NEW VERSION)
    Plots the 3 key platform metrics in *separate PDF files*, one for
    each metric and each dataset, for easier import into reports.

    (MODIFIED): Only shows 'skymask' for CIFAR10 and CIFAR100 datasets.
    """
    print("\n--- Plotting Platform Usability Metrics (Separate PDFs) ---")

    if df.empty:
        print("No data to plot.")
        return

    # --- THIS LINE IS REMOVED ---
    # We no longer need a single, static defense_order here.
    # defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # --- Data aggregation is the same ---
    # 1. Usability Rate
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100
    df_usability['Metric'] = '1. Usability Rate (%) (Higher is Better)'

    # 2. Average Performance (of *usable* runs)
    df_perf = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['acc'].mean().reset_index()
    df_perf['Value'] = df_perf['acc'] * 100
    df_perf['Metric'] = '2. Avg. Usable Accuracy (%) (Higher is Better)'

    # 3. Stability (Std Dev of *all* runs)
    df_stability = df.groupby(['defense', 'dataset'])['acc'].std().reset_index()
    df_stability['Value'] = df_stability['acc'] * 100
    df_stability['Metric'] = '3. Accuracy Instability (Std Dev) (Lower is Better)'

    # Combine all 3 metrics into one DataFrame
    df_final = pd.concat([df_usability, df_perf, df_stability], ignore_index=True)

    # --- Save the analysis CSV (same as before) ---
    csv_output_path = output_dir / "step2.5_platform_metrics_summary.csv"
    try:
        # Pivot for a wide, readable CSV
        df_pivot = df_final.pivot_table(index=['dataset', 'defense'], columns='Metric', values='Value')
        df_pivot.to_csv(csv_output_path, float_format="%.2f")
        print(f"\n✅ Successfully saved platform metrics summary to: {csv_output_path}\n")
    except Exception as e:
        print(f"Could not save CSV: {e}")
    # -----------------------------

    # --- NEW PLOTTING LOOP: Loop over each dataset AND each metric ---

    all_datasets = df_final['dataset'].unique()
    all_metrics = df_final['Metric'].unique()

    plot_count = 0
    for dataset in all_datasets:

        # --- NEW LOGIC: Define the order dynamically per dataset ---
        if dataset in ['CIFAR10', 'CIFAR100']:
            current_defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
        else:
            current_defense_order = ['fedavg', 'fltrust', 'martfl']
        # -----------------------------------------------------------

        for metric in all_metrics:

            # Filter data for this specific plot
            plot_df = df_final[
                (df_final['dataset'] == dataset) &
                (df_final['Metric'] == metric)
            ]

            if plot_df.empty:
                print(f"  Skipping {dataset} / {metric} (no data)")
                continue

            # Create a simple "safe" name for the file
            # e.g., "1. Usability Rate..." -> "1_Usability"
            safe_metric_name = metric.split(' ')[0].replace('.', '') + "_" + metric.split(' ')[1].replace('.', '')

            print(f"  Plotting: {dataset} - {safe_metric_name}")

            # 1. Create a new, single figure for this plot
            plt.figure(figsize=(7, 5)) # Good size for a single PDF plot

            # 2. Use sns.barplot on the new figure's axis
            ax = sns.barplot(
                data=plot_df,
                x='defense',
                y='Value',
                order=current_defense_order, # --- MODIFIED: Use dynamic order ---
                palette='viridis'
            )

            # 3. Set titles and labels
            ax.set_title(f"{metric}\nDataset: {dataset}", fontsize=14)
            ax.set_xlabel("Defense", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)

            # Set y-axis to start at 0
            ax.set_ylim(bottom=0)

            # 4. Add annotations
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 5),
                            textcoords='offset points',
                            fontsize=9)

            # 5. Define the PDF filename
            plot_file = output_dir / f"plot_{dataset}_{safe_metric_name}.pdf"

            # 6. Save as PDF
            plt.savefig(plot_file, bbox_inches='tight', format='pdf')
            plt.clf()
            plt.close('all') # Free up memory
            plot_count += 1

    print(f"\nDone. Generated {plot_count} separate PDF plots.")

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

    # --- THIS IS THE MODIFIED LINE ---
    # It now filters out any folder ending in '_no_clip'
    scenario_folders = [
        f for f in base_path.glob("step2.5_find_hps_*")
        if f.is_dir() and not f.name.endswith("_no_clip")
    ]
    # -----------------------------------

    if not scenario_folders:
        print(f"Error: No 'step2.5_find_hps_*' directories (excluding '_no_clip') found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories to process.")

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
    Plots the 3 key platform metrics in *separate, more readable files*,
    one for each dataset.
    """
    print("\n--- Plotting Platform Usability Metrics (Split by Dataset) ---")

    if df.empty:
        print("No data to plot.")
        return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Calculate the three metrics
    # 1. Usability Rate
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100  # Convert to %
    df_usability['Metric'] = '1. Usability Rate (%) (Higher is Better)'

    # 2. Average Performance (of *usable* runs)
    df_perf = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['acc'].mean().reset_index()
    df_perf['Value'] = df_perf['acc'] * 100  # Convert to %
    df_perf['Metric'] = '2. Avg. Usable Accuracy (%) (Higher is Better)'

    # 3. Stability (Std Dev of *all* runs)
    df_stability = df.groupby(['defense', 'dataset'])['acc'].std().reset_index()
    df_stability['Value'] = df_stability['acc'] * 100  # Convert to %
    df_stability['Metric'] = '3. Accuracy Instability (Std Dev) (Lower is Better)'

    # Combine all 3 metrics into one DataFrame
    df_final = pd.concat([df_usability, df_perf, df_stability], ignore_index=True)

    # --- Save the analysis CSV ---
    csv_output_path = output_dir / "step2.5_platform_metrics_summary.csv"
    try:
        # Pivot for a wide, readable CSV
        df_pivot = df_final.pivot_table(index=['dataset', 'defense'], columns='Metric', values='Value')
        df_pivot.to_csv(csv_output_path, float_format="%.2f")
        print(f"\n✅ Successfully saved platform metrics summary to: {csv_output_path}\n")
    except Exception as e:
        print(f"Could not save CSV: {e}")
    # -----------------------------

    # --- NEW PLOTTING LOOP ---
    # Loop over each dataset and create a separate figure
    for dataset in df_final['dataset'].unique():
        print(f"  Plotting all 3 metrics for: {dataset}")

        dataset_df = df_final[df_final['dataset'] == dataset]

        g = sns.catplot(
            data=dataset_df,
            kind='bar',
            x='defense',
            y='Value',
            col='Metric',  # Create 3 columns: Usability, Avg Acc, Instability
            order=defense_order,
            palette='viridis',
            height=4,
            aspect=1.1,
            sharey=False  # Each metric has its own scale
        )

        g.fig.suptitle(f"Platform-Centric Usability for {dataset} (vs. {RELATIVE_ACC_THRESHOLD * 100}% of Max Acc)",
                       y=1.05)
        g.set_axis_labels("Defense", "Value")
        g.set_titles(col_template="{col_name}")

        # Add annotations
        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 5),
                            textcoords='offset points',
                            fontsize=9)

        # Save with the dataset name in the file
        plot_file = output_dir / f"plot_platform_metrics_{dataset}.png"
        plt.savefig(plot_file, bbox_inches='tight')
        plt.clf();
        plt.close('all')

    print("Done plotting all datasets.")


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

    # --- Call the new PDF-generating plotter ---
    plot_platform_usability_metrics_separate_pdfs(df, output_dir)
    # ----------------------------------------

    print("\nAnalysis complete. Check 'step2.5_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()
