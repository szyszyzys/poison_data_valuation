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

# This is now the *only* metric for usability
PLATFORM_USABLE_ACC_THRESHOLD = 0.70


# ---------------------


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    (FIXED) Parses the HP suffix folder name (e.g., 'opt_Adam_lr_0.001_epochs_2')
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
    (FIXED) Parses the base scenario name using regex to handle
    underscores in dataset and model names.
    e.g., 'step2.5_find_hps_martfl_image_CIFAR100'
    """
    try:
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3),
            }
        raise ValueError("Pattern not matched")
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
    # Define "Usable" from the platform's (accuracy-only) perspective
    df['platform_usable'] = (df['acc'] >= PLATFORM_USABLE_ACC_THRESHOLD)

    return df


def plot_platform_usability_rate(df: pd.DataFrame, output_dir: Path):
    """
    Plots the 'Platform Usability Rate' for each defense.
    This answers: "How easy is it to get a high-accuracy model?"
    """
    print("\n--- Plotting Defense 'Platform Usability Rate' (Accuracy Only) ---")

    if 'platform_usable' not in df.columns:
        print("Error: 'platform_usable' column not found. Cannot plot usability.")
        return

    # Calculate the Usability Rate: (Number of Usable Runs) / (Total Runs)
    usability_df = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    usability_df['Platform Usability Rate'] = usability_df['platform_usable'] * 100  # Convert to %

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    g = sns.catplot(
        data=usability_df,
        kind='bar',
        x='defense',
        y='Platform Usability Rate',
        col='dataset',
        col_wrap=3,
        height=4,
        aspect=1.2,
        order=defense_order,
        palette='viridis'
    )

    g.fig.suptitle(f"Defense 'Platform Usability Rate' (HP Sweeps > {PLATFORM_USABLE_ACC_THRESHOLD * 100}% Accuracy)",
                   y=1.03)
    g.set_axis_labels("Defense", "Usable HP Combinations (%)")
    g.set_titles(col_template="{col_name}")

    # Add annotations
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=9)

    plot_file = output_dir / "plot_platform_USABILITY_RATE.png"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved Platform Usability Rate plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # --- Save the analysis CSV ---
    csv_output_path = output_dir / "step2.5_full_results_summary.csv"
    df.to_csv(csv_output_path, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_path}\n")
    # -----------------------------

    # --- Call the new plotter ---
    plot_platform_usability_rate(df, output_dir)
    # ----------------------------

    print("\nAnalysis complete. Check 'step2.5_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()