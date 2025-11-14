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
BASE_RESULTS_DIR = "./results"  # <-- Make sure this points to your new results
FIGURE_OUTPUT_DIR = "./step2.5_figures"

# --- Define what "usable" means ---
# We use a LOW bar here. We just want the model to learn *something*
# and not be 100% backdoored.
USABLE_ACC_THRESHOLD = 0.50  # Must be > 50% accurate
USABLE_ASR_THRESHOLD = 0.95  # Must be < 95% ASR


# ----------------------------------


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses 'step2.5_find_hps_fedavg_image_CIFAR10'"""
    try:
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario_base": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3),
            }
        raise ValueError("Pattern not matched")
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario_base": scenario_name}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses 'opt_Adam_lr_0.001_epochs_2' into a dict."""
    hps = {}
    try:
        opt_match = re.search(r'opt_([A-Za-z]+)', hp_folder_name)
        lr_match = re.search(r'lr_([0-9\.]+)', hp_folder_name)
        epochs_match = re.search(r'epochs_([0-9]+)', hp_folder_name)

        if opt_match: hps['optimizer'] = opt_match.group(1)
        if lr_match: hps['lr'] = float(lr_match.group(1))
        if epochs_match: hps['epochs'] = int(epochs_match.group(1))

    except Exception as e:
        print(f"Error parsing HPs from '{hp_folder_name}': {e}")
    return hps


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads key metrics from output files."""
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
    """Walks the Step 2.5 results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = list(base_path.glob("step2.5_find_hps_*"))
    if not scenario_folders:
        print(f"Error: No 'step2.5_find_hps_*' directories found in {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        if not scenario_path.is_dir(): continue

        base_scenario_info = parse_scenario_name(scenario_path.name)

        for hp_path in scenario_path.iterdir():
            if not hp_path.is_dir(): continue

            hp_info = parse_hp_suffix(hp_path.name)

            for metrics_file in hp_path.rglob("final_metrics.json"):
                try:
                    run_metrics = load_run_data(metrics_file)
                    if run_metrics:
                        all_runs.append({
                            **base_scenario_info,
                            **hp_info,
                            **run_metrics
                        })
                except Exception as e:
                    print(f"Error processing file {metrics_file}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # --- This is the key logic ---
    df['is_usable'] = (df['acc'] >= USABLE_ACC_THRESHOLD) & \
                      (df['asr'] <= USABLE_ASR_THRESHOLD)

    # Handle the 'no attack' case (ASR=0)
    df.loc[df['asr'] == 0.0, 'is_usable'] = (df['acc'] >= USABLE_ACC_THRESHOLD)

    df['defense_score'] = df['acc'] - df['asr']

    return df


def plot_usability_rate(df: pd.DataFrame, output_dir: Path):
    """
    Plots the 'Usability Rate' for each defense.
    This answers: "How easy is it to get a working model?"
    """
    print("\n--- Plotting Defense Usability Rate ---")

    if 'is_usable' not in df.columns:
        print("Error: 'is_usable' column not found. Cannot plot usability.")
        return

    # Calculate the Usability Rate: (Number of Usable Runs) / (Total Runs)
    # We group by defense and dataset to see the whole picture
    usability_df = df.groupby(['defense', 'dataset'])['is_usable'].mean().reset_index()
    usability_df['Usability Rate'] = usability_df['is_usable'] * 100  # Convert to %

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    g = sns.catplot(
        data=usability_df,
        kind='bar',
        x='defense',
        y='Usability Rate',
        col='dataset',
        col_wrap=3,
        height=4,
        aspect=1.2,
        order=defense_order,
        palette='viridis'
    )

    g.fig.suptitle(
        f"Defense 'Usability Rate' (HP Sweeps > {USABLE_ACC_THRESHOLD * 100}% Acc & < {USABLE_ASR_THRESHOLD * 100}% ASR)",
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

    plot_file = output_dir / "plot_defense_USABILITY_RATE.png"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved Usability Rate plot: {plot_file}")
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
    plot_usability_rate(df, output_dir)
    # ----------------------------

    print("\nAnalysis complete. Check 'step2.5_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()