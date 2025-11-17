import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step5_figures"


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'adv_0.1_poison_0.5')
    """
    hps = {}
    pattern = r'adv_([0-9\.]+)_poison_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['adv_rate'] = float(match.group(1))
        hps['poison_rate'] = float(match.group(2))
    else:
        print(f"Warning: Could not parse HP suffix '{hp_folder_name}'")
    return hps


def parse_scenario_name(scenario_name: str) -> Optional[Dict[str, str]]:
    """
    (FIXED) Parses the base scenario name...
    """
    try:
        # --- THIS REGEX MUST MATCH ALL YOUR FOLDERS ---
        pattern = r'step5_atk_sens_(adv|poison)_(fedavg|fltrust|martfl|skymask)_(backdoor|labelflip)_(image|text|tabular)$'
        match = re.search(pattern, scenario_name)

        if match:
            modality = match.group(4)

            # --- THIS IS THE FIX ---
            # Add mappings for all your modalities
            if modality == 'image':
                dataset_name = 'CIFAR100'
            elif modality == 'text':
                dataset_name = 'TREC'  # Or whatever your text dataset is
            elif modality == 'tabular':
                dataset_name = 'Texas100'  # Or whatever your tabular dataset is
            else:
                dataset_name = 'unknown'  # This should now be impossible

            return {
                "scenario": scenario_name,
                "sweep_type": match.group(1),
                "defense": match.group(2),
                "attack": match.group(3),
                "modality": modality,
                "dataset": dataset_name,  # This will now be correct
            }
        else:
            # If this runs, you'll see a warning.
            print(f"Warning: Could not parse scenario name '{scenario_name}'. Ignoring folder.")
            return None

    except Exception as e:
        print(f"Warning: Error parsing scenario name '{scenario_name}': {e}. Ignoring folder.")
        return None


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

            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
            run_data['benign_selection_rate'] = np.mean(
                [s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step5_atk_sens_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step5_atk_sens_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    parsed_count = 0
    metrics_found_count = 0

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        if run_scenario is None:
            continue

        print(f"\nProcessing Scenario: {scenario_name}")
        print(f"  -> Parsed as: {run_scenario}")

        parsed_count += 1

        files_in_scenario = list(scenario_path.rglob("final_metrics.json"))
        if not files_in_scenario:
            print(f"  -> !! WARNING: No 'final_metrics.json' files found in {scenario_name}")
            continue

        for metrics_file in files_in_scenario:
            try:
                metrics_found_count += 1
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts:
                    print(f"  -> Skipping metric file with no relative path: {metrics_file}")
                    continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)

                if not run_hps:
                    print(f"  -> Skipping metric file due to HP parse fail: {hp_folder_name}")
                    continue

                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics,
                        "hp_suffix": hp_folder_name
                    })
                else:
                    print(f"  -> Skipping metric file, load_run_data failed: {metrics_file}")

            except Exception as e:
                print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    print(f"\n--- DEBUG SUMMARY ---")
    print(f"Successfully parsed {parsed_count} scenario folders.")
    print(f"Found a total of {metrics_found_count} 'final_metrics.json' files.")
    print(f"Aggregated {len(all_runs)} total runs into the DataFrame.")
    print(f"--- END DEBUG ---")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_sensitivity_lines(df: pd.DataFrame, x_metric: str, attack_type: str, dataset: str, output_dir: Path):
    """
    (REWRITTEN)
    Generates an *individual PDF plot for each metric*.
    """
    print(f"\n--- Plotting Robustness vs. '{x_metric}' (for {attack_type} on {dataset}) ---")

    if attack_type == 'backdoor':
        metrics_to_plot = ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate', 'rounds']
    elif attack_type == 'labelflip':
        metrics_to_plot = ['acc', 'benign_selection_rate', 'adv_selection_rate', 'rounds']
    else:
        print(f"Skipping plot for unknown attack type: {attack_type}")
        return

    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    if not metrics_to_plot:
        print(f"No metrics found to plot for {dataset}/{attack} vs {x_metric}.")
        return

    # Melt the dataframe *once*
    plot_df = df.melt(
        id_vars=[x_metric, 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    rate_metrics = ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']
    plot_df['Value'] = plot_df.apply(
        lambda row: row['Value'] * 100 if row['Metric'] in rate_metrics else row['Value'],
        axis=1
    )

    # --- NEW LOOP: Create one plot per metric ---
    for metric in metrics_to_plot:

        # Create a clean metric name for title and filename
        metric_name_clean = f'{metric} (%)' if metric in rate_metrics else metric

        print(f"  -> Plotting metric: {metric_name_clean}")

        # Filter the melted dataframe for this metric only
        metric_df = plot_df[plot_df['Metric'] == metric].copy()

        if metric_df.empty:
            print(f"    ...No data for metric '{metric}'. Skipping plot.")
            continue

        # Create a new figure for this plot
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=metric_df,
            x=x_metric,
            y='Value',
            hue='defense',
            style='defense',
            markers=True,
            dashes=False
        )

        title_x = x_metric.replace("_", " ").title()
        title_dataset = dataset.upper()
        title_attack = attack_type.title()

        ax.set_title(f'{metric_name_clean} vs. {title_x}\n({title_attack} on {title_dataset})')
        ax.set_xlabel(title_x)
        ax.set_ylabel(metric_name_clean)

        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set y-axis for rates
        if metric in rate_metrics:
            ax.set_ylim(-5, 105)  # From -5 to 105 to see 0 and 100 clearly

        # Create a safe filename for the dataset
        safe_dataset_name = re.sub(r'[^\w]', '', dataset)

        # Create the new, specific filename
        plot_file_name = f"plot_robustness_{attack_type}_{safe_dataset_name}_vs_{x_metric}_{metric}.pdf"
        plot_file = output_dir / plot_file_name

        plt.savefig(plot_file, bbox_inches='tight', format='pdf')
        print(f"    ...Saved plot: {plot_file_name}")

        # Close the plot to free memory
        plt.clf()
        plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # Loop over dataset and attack
    for dataset in df['dataset'].unique():
        if dataset in ['unknown', 'parse_failed']:
            print(f"Skipping plots for '{dataset}' group (due to parsing errors).")
            continue

        for attack in df['attack'].unique():
            if pd.isna(attack):
                continue

            scenario_df = df[(df['dataset'] == dataset) & (df['attack'] == attack)].copy()

            if scenario_df.empty:
                continue

            # Plot 1: Sweep vs. Adversary Rate
            adv_rate_df = scenario_df[scenario_df['sweep_type'] == 'adv']
            if not adv_rate_df.empty:
                plot_sensitivity_lines(adv_rate_df, 'adv_rate', attack, dataset, output_dir)
            else:
                print(f"No data found for {dataset}/{attack} vs. adv_rate sweep.")

            # Plot 2: Sweep vs. Poison Rate
            poison_rate_df = scenario_df[scenario_df['sweep_type'] == 'poison']
            if not poison_rate_df.empty:
                plot_sensitivity_lines(poison_rate_df, 'poison_rate', attack, dataset, output_dir)
            else:
                print(f"No data found for {dataset}/{attack} vs. poison_rate sweep.")

    print("\nAnalysis complete. Check 'step5_figures' folder for plots.")


if __name__ == "__main__":
    main()