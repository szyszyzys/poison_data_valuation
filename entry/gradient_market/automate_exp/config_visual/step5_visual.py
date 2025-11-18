import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm # Added tqdm for progress

# --- Configuration ---
#
# !!! IMPORTANT !!!
# Make sure this path is correct. Use the full absolute path.
#
BASE_RESULTS_DIR = "/scratch/zzs5287/poison_data_valuation/results"
#
# (The figure output dir is not needed for this script)
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
                dataset_name = 'TREC'
            elif modality == 'tabular':
                dataset_name = 'Texas100'
            else:
                dataset_name = 'unknown'

            return {
                "scenario": scenario_name,
                "sweep_type": match.group(1),
                "defense": match.group(2),
                "attack": match.group(3),
                "modality": modality,
                "dataset": dataset_name,
            }
        else:
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
        # These keys match your final_metrics.json
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

    for scenario_path in tqdm(scenario_folders, desc="Processing Scenarios"):
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        if run_scenario is None:
            continue

        files_in_scenario = list(scenario_path.rglob("final_metrics.json"))
        if not files_in_scenario:
            continue

        for metrics_file in files_in_scenario:
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts:
                    continue

                hp_folder_name = relative_parts[0] # This correctly gets 'adv_0.3_poison_0.5'
                run_hps = parse_hp_suffix(hp_folder_name)

                if not run_hps:
                    continue

                run_metrics = load_run_data(metrics_file)

                # --- ADDED FOR DEBUGGING ---
                # This adds the source file path to your data
                try:
                    relative_metric_path = metrics_file.relative_to(base_path)
                    run_metrics['source_file'] = str(relative_metric_path)
                except ValueError:
                    run_metrics['source_file'] = str(metrics_file)
                # --- END ADD ---

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


def main():
    """
    Main function to collect data, filter for martfl/backdoor,
    and save to a CSV file.
    """
    output_csv_file = "./martfl_backdoor_debug.csv"
    print(f"Starting data collection from: {BASE_RESULTS_DIR}")
    print(f"Debug CSV will be saved to: {output_csv_file}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("\n--- ERROR ---")
        print("No data was loaded. Exiting.")
        print(f"Please check your BASE_RESULTS_DIR path: {BASE_RESULTS_DIR}")
        return

    print(f"\nTotal runs aggregated: {len(df)}")

    # --- Apply Your Filters ---
    print("Filtering for defense='martfl' and attack='backdoor'...")
    filtered_df = df[
        (df['defense'] == 'martfl') &
        (df['attack'] == 'backdoor')
    ].copy()

    if filtered_df.empty:
        print("\n--- ERROR ---")
        print("No data found matching 'martfl' AND 'backdoor'.")
        print("This is likely the source of your plotting error.")
        print("\n--- Debug Info ---")
        print(f"Available defenses found in all data: {df['defense'].unique()}")
        print(f"Available attacks found in all data: {df['attack'].unique()}")
        return

    print(f"Found {len(filtered_df)} runs matching 'martfl' and 'backdoor'.")

    # --- Select and Order Columns for CSV ---
    all_cols = [
        'defense', 'attack', 'dataset', 'modality', 'sweep_type',
        'adv_rate', 'poison_rate',
        'acc', 'asr', 'benign_selection_rate', 'adv_selection_rate', 'rounds',
        'source_file', 'hp_suffix', 'scenario'
    ]

    # Only keep columns that were successfully found in the dataframe
    export_cols = [col for col in all_cols if col in filtered_df.columns]

    final_df = filtered_df[export_cols].sort_values(by=['sweep_type', 'adv_rate', 'poison_rate', 'source_file'])

    # --- Save to CSV ---
    try:
        final_df.to_csv(output_csv_file, index=False)
        print(f"\n✅ Successfully saved debug data to {output_csv_file}")
        print("\nYou can now open this CSV in Excel or Pandas to inspect the values.")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Failed to save CSV file: {e}")


if __name__ == "__main__":
    main()