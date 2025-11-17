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
FIGURE_OUTPUT_DIR = "./figures/step2.5_figures"

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
    (UPDATED - Learning from step12)
    Loads metrics from final_metrics.json (acc, rounds)
    AND calculates *both* benign and adversary selection rates
    from marketplace_report.json.
    """
    run_data = {}
    try:
        # 1. Load basic metrics (acc, rounds)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        # 2. --- NEW LOGIC (from your step12 example) ---
        report_file = metrics_file.parent / "marketplace_report.json"

        if not report_file.exists():
            # Set to NaN to handle cases where report wasn't generated
            run_data['benign_selection_rate'] = np.nan
            run_data['adv_selection_rate'] = np.nan
            return run_data

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {}).values()

        # Check for both types, just like in your step12 script
        adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
        ben_sellers = [s for s in sellers if s.get('type') == 'benign']

        # Calculate mean for adv sellers (will be 0.0 if list is empty)
        run_data['adv_selection_rate'] = np.mean(
            [s['selection_rate'] for s in adv_sellers]
        ) if adv_sellers else 0.0

        # Calculate mean for benign sellers (will be NaN if list is empty)
        run_data['benign_selection_rate'] = np.mean(
            [s['selection_rate'] for s in ben_sellers]
        ) if ben_sellers else np.nan

        # --- END NEW LOGIC ---

        return run_data
    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    (UPDATED)
    Walks the results directory, aggregates all run data,
    AND filters out any folders ending in '_nolocalclip'.
    """
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    # --- THIS IS THE NEW FILTER LOGIC ---
    scenario_folders = [
        f for f in base_path.glob("step2.5_find_hps_*")
        if f.is_dir() and not f.name.endswith("_nolocalclip")
    ]
    # --- END NEW FILTER LOGIC ---

    if not scenario_folders:
        print(f"Error: No 'step2.5_find_hps_*' directories (excluding '_nolocalclip') found in {base_path}.")
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

                # --- This call now returns both selection rates ---
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

    print("\nCalculating relative 'Platform Usable' thresholds...")
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])
    print("Done calculating thresholds.")

    return df


def plot_platform_usability_with_selection(df: pd.DataFrame, output_dir: Path):
    """
    (UPDATED)
    Plots the 5 key platform metrics.
    - Metrics 1-4 are separate PDFs.
    - Metric 5 is a new GROUPED bar chart for *both* selection rates.
    """
    print("\n--- Plotting Platform Usability & Selection Metrics (Separate PDFs) ---")

    if df.empty:
        print("No data to plot.")
        return

    # Check that all necessary columns are present
    required_cols = ['acc', 'rounds', 'platform_usable', 'benign_selection_rate', 'adv_selection_rate']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing required data columns (acc, rounds, benign_selection_rate, or adv_selection_rate).")
        print("Please check 'load_run_data' and your .json files.")
        return

    # === METRIC CALCULATIONS ===

    # --- Metrics 1-4: Calculated as before ---
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100
    df_usability['Metric'] = '1. Usability Rate (%) (Higher is Better)'

    df_perf = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['acc'].mean().reset_index()
    df_perf['Value'] = df_perf['acc'] * 100
    df_perf['Metric'] = '2. Avg. Usable Accuracy (%) (Higher is Better)'

    df_speed = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['rounds'].mean().reset_index()
    df_speed['Value'] = df_speed['rounds']
    df_speed['Metric'] = '3. Avg. Usable Rounds (Lower is Better)'

    df_speed_stability = df.groupby(['defense', 'dataset'])['rounds'].std().reset_index()
    df_speed_stability['Value'] = df_speed_stability['rounds']
    df_speed_stability['Metric'] = '4. Rounds Instability (Std Dev) (Lower is Better)'

    # Combine metrics 1-4 for plotting
    df_metrics_1_4 = pd.concat([
        df_usability,
        df_perf,
        df_speed,
        df_speed_stability
    ], ignore_index=True)

    # --- Metric 5: Selection Rates (for grouped bar chart) ---
    df_selection_raw = df.groupby(['defense', 'dataset'])[
        ['benign_selection_rate', 'adv_selection_rate']].mean().reset_index()

    # Rename for melting
    df_selection_raw = df_selection_raw.rename(columns={
        'benign_selection_rate': 'Avg. Benign Selection Rate',
        'adv_selection_rate': 'Avg. Adversary Selection Rate'
    })

    # Melt to long format for seaborn `hue`
    df_selection_melted = df_selection_raw.melt(
        id_vars=['defense', 'dataset'],
        value_vars=['Avg. Benign Selection Rate', 'Avg. Adversary Selection Rate'],
        var_name='Seller Type',
        value_name='Selection Rate'
    )
    # Convert to percentage
    df_selection_melted['Selection Rate'] *= 100

    # === CSV SAVING ===
    # We'll save all 6 metrics in one pivoted CSV
    # Re-calculate selection metrics for the CSV
    df_benign_csv = df_selection_raw[['defense', 'dataset', 'Avg. Benign Selection Rate']].copy()
    df_benign_csv['Value'] = df_benign_csv['Avg. Benign Selection Rate'] * 100
    df_benign_csv['Metric'] = '5. Avg. Benign Selection Rate (%)'

    df_adv_csv = df_selection_raw[['defense', 'dataset', 'Avg. Adversary Selection Rate']].copy()
    df_adv_csv['Value'] = df_adv_csv['Avg. Adversary Selection Rate'] * 100
    df_adv_csv['Metric'] = '6. Avg. Adversary Selection Rate (%)'

    df_final_csv = pd.concat([
        df_metrics_1_4,
        df_benign_csv[['defense', 'dataset', 'Value', 'Metric']],
        df_adv_csv[['defense', 'dataset', 'Value', 'Metric']]
    ], ignore_index=True)

    csv_output_path = output_dir / "step2.5_platform_metrics_with_selection_summary.csv"
    try:
        df_pivot = df_final_csv.pivot_table(index=['dataset', 'defense'], columns='Metric', values='Value')
        df_pivot.to_csv(csv_output_path, float_format="%.2f")
        print(f"\n✅ Successfully saved platform metrics summary to: {csv_output_path}\n")
    except Exception as e:
        print(f"Could not save CSV: {e}")
    # -----------------------------

    # === PLOTTING LOOP ===
    all_datasets = df_final_csv['dataset'].unique()
    plot_count = 0

    for dataset in all_datasets:
        # Use the same dynamic defense order
        if dataset in ['CIFAR10', 'CIFAR100']:
            current_defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
        else:
            current_defense_order = ['fedavg', 'fltrust', 'martfl']

        # --- 1. Plot Metrics 1-4 (Separate PDFs) ---
        for metric in df_metrics_1_4['Metric'].unique():
            plot_df = df_metrics_1_4[
                (df_metrics_1_4['dataset'] == dataset) &
                (df_metrics_1_4['Metric'] == metric)
                ]
            plot_df = plot_df[plot_df['defense'].isin(current_defense_order)]

            if plot_df.empty:
                print(f"  Skipping {dataset} / {metric} (no data)")
                continue

            # Create a clean filename
            safe_parts = [re.sub(r'[^\w]', '', part) for part in metric.split(' ')[:3]]
            safe_metric_name = '_'.join(safe_parts)

            print(f"  Plotting: {dataset} - {safe_metric_name}")
            plt.figure(figsize=(7, 5))
            ax = sns.barplot(
                data=plot_df,
                x='defense',
                y='Value',
                order=current_defense_order,
                palette='viridis'
            )
            ax.set_title(f"{metric}\nDataset: {dataset}", fontsize=14)
            ax.set_xlabel("Defense", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_ylim(bottom=0)

            # Add annotations
            for p in ax.patches:
                format_str = '%.1f'
                if 'Rounds' in metric and p.get_height() < 100:
                    format_str = '%.2f'
                ax.annotate(format_str % p.get_height(),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 5), textcoords='offset points', fontsize=9)

            plot_file = output_dir / f"plot_{dataset}_{safe_metric_name}.pdf"
            plt.savefig(plot_file, bbox_inches='tight', format='pdf')
            plt.clf();
            plt.close('all');
            plot_count += 1

        # --- 2. Plot Metric 5 (Grouped Bar Chart) ---
        plot_df_selection = df_selection_melted[
            (df_selection_melted['dataset'] == dataset) &
            (df_selection_melted['defense'].isin(current_defense_order))
            ]

        if plot_df_selection.empty:
            print(f"  Skipping {dataset} / Selection Rates (no data)")
            continue

        safe_metric_name = "5_Selection_Rates"
        print(f"  Plotting: {dataset} - {safe_metric_name}")

        plt.figure(figsize=(9, 6))
        ax = sns.barplot(
            data=plot_df_selection,
            x='defense',
            y='Selection Rate',
            hue='Seller Type',  # This creates the grouped bars
            order=current_defense_order,
            palette={'Avg. Benign Selection Rate': 'g', 'Avg. Adversary Selection Rate': 'r'}
        )
        ax.set_title(f"Avg. Selection Rates\nDataset: {dataset}", fontsize=14)
        ax.set_xlabel("Defense", fontsize=12)
        ax.set_ylabel("Selection Rate (%)", fontsize=12)
        ax.set_ylim(bottom=0, top=105)  # Set top to 105 to see 100% bars
        ax.legend(title='Seller Type')

        # Add annotations
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5), textcoords='offset points', fontsize=9)

        plot_file = output_dir / f"plot_{dataset}_{safe_metric_name}.pdf"
        plt.savefig(plot_file, bbox_inches='tight', format='pdf')
        plt.clf();
        plt.close('all');
        plot_count += 1

    print(f"\nDone. Generated {plot_count} total PDF plots.")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # This will use the new load_run_data and get both selection rates
    # It will also filter out '_nolocalclip' folders
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # --- Call the NEW plotter ---
    plot_platform_usability_with_selection(df, output_dir)
    # ----------------------------

    print("\nAnalysis complete. Check 'step2.5_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()