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
FIGURE_OUTPUT_DIR = "./figures/skymask_comparison"
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

def parse_scenario_name_universal(scenario_name: str) -> Dict[str, str]:
    """
    UNIVERSAL PARSER: Captures any defense name.
    It looks for text between 'step2.5_find_hps_' and the modality.
    """
    try:
        # Regex explanation:
        # step2\.5_find_hps_  -> Prefix
        # (?P<defense>.+?)    -> Capture Defense (Non-greedy, takes anything until the next underscore)
        # _                   -> Separator
        # (?P<modality>image|text|tabular) -> Modality
        # _                   -> Separator
        # (?P<dataset>.+)     -> Dataset
        pattern = r'step2\.5_find_hps_(?P<defense>.+?)_(?P<modality>image|text|tabular)_(?P<dataset>.+)'

        match = re.search(pattern, scenario_name)

        if match:
            return match.groupdict() # Returns {'defense': '...', 'modality': '...', 'dataset': '...'}
        else:
            return {} # Return empty if no match
    except Exception as e:
        return {}

def collect_all_results_debug(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"\n--- SCANNING DIRECTORY: {base_path.resolve()} ---")

    # 1. Check if directory exists
    if not base_path.exists():
        print(f"ERROR: The directory {base_path} does not exist!")
        return pd.DataFrame()

    # 2. List all candidates
    # We remove the '_nolocalclip' filter temporarily to see if that's the cause
    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]

    print(f"Found {len(scenario_folders)} potential folders.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name

        # DEBUG: Check exclusion
        if scenario_name.endswith("_nolocalclip"):
            print(f"SKIPPING (explicit filter): {scenario_name}")
            continue

        # Try parsing
        run_scenario = parse_scenario_name_universal(scenario_name)

        if not run_scenario:
            print(f"SKIPPING (regex mismatch): {scenario_name}")
            continue

        # Check if we found skymask_small
        defense_name = run_scenario['defense']

        # Look for metrics files
        metrics_files = list(scenario_path.rglob("final_metrics.json"))

        if not metrics_files:
            print(f"SKIPPING (no json files):  {scenario_name} (Defense: {defense_name})")
            continue

        print(f"LOADING:                   {scenario_name} -> Found {len(metrics_files)} runs.")

        for metrics_file in metrics_files:
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)

                # Load Metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Load Marketplace Report (optional)
                report_file = metrics_file.parent / "marketplace_report.json"
                adv_rate, ben_rate = 0.0, np.nan
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        rep = json.load(f)
                    sellers = rep.get('seller_summaries', {}).values()
                    adv_s = [s for s in sellers if s.get('type') == 'adversary']
                    ben_s = [s for s in sellers if s.get('type') == 'benign']
                    if adv_s: adv_rate = np.mean([s['selection_rate'] for s in adv_s])
                    if ben_s: ben_rate = np.mean([s['selection_rate'] for s in ben_s])

                all_runs.append({
                    **run_scenario,
                    **run_hps,
                    'acc': metrics.get('acc', 0),
                    'rounds': metrics.get('completed_rounds', 0),
                    'adv_selection_rate': adv_rate,
                    'benign_selection_rate': ben_rate
                })
            except Exception as e:
                # print(f"Error reading file: {e}")
                continue

    df = pd.DataFrame(all_runs)
    if df.empty:
        print("\nCRITICAL: Resulting DataFrame is empty.")
        return df

    print("\n--- DATA SUMMARY ---")
    print(f"Total Rows: {len(df)}")
    print("Defenses found:", df['defense'].unique())
    print("Datasets found:", df['dataset'].unique())

    return df

def plot_direct_comparison(df: pd.DataFrame, output_dir: Path):
    # Filter for loose matches of "skymask"
    # This allows 'skymask', 'skymask_small', 'skymask-small', etc.
    df_comp = df[df['defense'].str.contains("skymask", case=False)].copy()

    if df_comp.empty:
        print("\nNo 'skymask' related data found to plot.")
        return

    print(f"\nPlotting defenses: {df_comp['defense'].unique()}")

    sns.set_theme(style="whitegrid", context="talk")
    datasets = df_comp['dataset'].unique()

    for ds in datasets:
        ds_data = df_comp[df_comp['dataset'] == ds]

        # Dynamic Palette
        unique_defenses = sorted(ds_data['defense'].unique())
        palette = sns.color_palette("Blues", n_colors=len(unique_defenses))

        # ACCURACY
        plt.figure(figsize=(10, 6))
        sns.barplot(data=ds_data, x='defense', y='acc', order=unique_defenses, palette=palette, edgecolor='black')
        plt.title(f"Accuracy Comparison - {ds}")
        plt.tight_layout()
        plt.savefig(output_dir / f"debug_compare_{ds}_acc.pdf")
        plt.close()

        print(f"Saved plot for {ds}")

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Use the DEBUG collector
    df = collect_all_results_debug(BASE_RESULTS_DIR)

    if not df.empty:
        plot_direct_comparison(df, output_dir)

if __name__ == "__main__":
    main()