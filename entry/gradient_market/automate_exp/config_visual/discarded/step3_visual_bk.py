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
# IMPORTANT: Point this to the output directory from your script
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step3_figures_v2"


# ---------------------


def parse_base_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses 'step3_tune_skymask_backdoor_image_CIFAR100_cnn_new'"""
    try:
        pattern = r'step3_tune_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(image|text|tabular)_(.+)_new'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "defense": match.group(1),
                "attack_type": match.group(2),
                "modality": match.group(3),
                "dataset_model": match.group(4),
                "dataset": match.group(4).split('_')[0]
            }
        raise ValueError("Pattern not matched")
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def parse_hps_from_folder(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses 'aggregation.skymask.mask_epochs_20_...' into a dict.
    """
    hps = {}
    try:
        # This regex finds all key-value pairs
        pattern = r'([a-zA-Z0-9\._]+)_([a-zA-Z0-9\._]+)'
        matches = re.findall(pattern, hp_folder_name)

        # We re-build the key-value pairs
        i = 0
        while i < len(matches):
            key_part = matches[i][0]
            val_part = matches[i][1]

            # Handle keys split by the regex
            if key_part == 'aggregation.skymask.mask':
                key = f"{key_part}_{val_part}"  # e.g., 'aggregation.skymask.mask_epochs'
                val = matches[i + 1][0] if i + 1 < len(matches) else 'ERROR'
                if val_part == 'lr':  # Fix for 'lr_0.001'
                    key = 'aggregation.skymask.mask_lr'
                    val = f"0.{matches[i][1].split('.')[1]}"
                i += 1  # Move to the next full pair
            elif key_part == 'aggregation.martfl.max':
                key = 'aggregation.martfl.max_k'
                val = val_part
            elif key_part == 'aggregation.clip':
                key = 'aggregation.clip_norm'
                val = val_part
            else:
                key = key_part
                val = val_part

            # Clean up key name and value
            key = key.split('.')[-1]  # 'aggregation.skymask.mask_epochs' -> 'mask_epochs'
            if val == 'None':
                val = 'None'
            elif '.' in val:
                val = float(val)
            else:
                try:
                    val = int(val)
                except:
                    val = val

            hps[key] = val
            i += 1

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
    """Walks the Step 3 results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step3_tune_*_new") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step3_tune_*_new' directories found in {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        base_scenario_info = parse_base_scenario_name(scenario_path.name)

        for hp_path in scenario_path.iterdir():
            if not hp_path.is_dir(): continue
            hp_info = parse_hps_from_folder(hp_path.name)

            for metrics_file in hp_path.rglob("final_metrics.json"):
                run_metrics = load_run_data(metrics_file)
                if run_metrics:
                    all_runs.append({
                        **base_scenario_info,
                        **hp_info,
                        **run_metrics
                    })

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    # Create the 'defense_score' = acc - asr
    df['defense_score'] = df['acc'] - df['asr']
    return df


def plot_skymask_failure_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Specifically answers: 'Did SkyMask fail in all setups?'
    """
    print("\n--- Plotting SkyMask Failure Analysis ---")
    df_skymask = df[df['defense'] == 'skymask'].copy()
    if df_skymask.empty:
        print("No SkyMask data found to analyze.")
        return

    # 'Failure' = 100% benign selection AND 100% adversary selection
    df_skymask['filter_failure'] = (df_skymask['benign_selection_rate'] >= 0.999) & \
                                   (df_skymask['adv_selection_rate'] >= 0.999)

    # Aggregate by all HPs, datasets, and attacks
    agg_df = df_skymask.groupby([
        'dataset', 'attack_type',
        'mask_epochs', 'mask_lr', 'mask_threshold'
    ]).agg(
        failure_rate=('filter_failure', 'mean'),
        defense_score=('defense_score', 'mean')
    ).reset_index()

    # Plot 1: Heatmap of Failure Rate
    g = sns.catplot(
        data=agg_df,
        x='mask_lr',
        y='mask_threshold',
        col='dataset',
        row='attack_type',
        hue='mask_epochs',
        kind='bar',
        palette='Reds',
        height=4, aspect=1.2,
        estimator=np.mean,  # Use estimator to get mean failure rate
        errorbar=None,
        legend_out=True
    )
    # This plot doesn't work, let's use a faceted bar chart
    plt.close('all')  # Close the bad plot

    g = sns.catplot(
        data=agg_df,
        x='mask_epochs',
        y='failure_rate',
        hue='mask_threshold',
        col='mask_lr',
        row='dataset',
        kind='bar',
        palette='viridis',
        height=3,
        aspect=1
    )
    g.fig.suptitle("SkyMask Filter Failure Rate (1.0 = 100% Failure)", y=1.03)
    g.set_axis_labels("Mask Epochs", "Failure Rate (1.0 = 100%)")
    g.set_titles(col_template="LR={col_name}", row_template="{row_name}")
    g.add_legend(title='Mask Threshold')

    plot_file = output_dir / "plot_skymask_FAILURE_ANALYSIS.png"
    plt.savefig(plot_file)
    print(f"Saved SkyMask failure plot: {plot_file}")
    plt.clf()


def plot_tuning_tradeoffs(df: pd.DataFrame, output_dir: Path):
    """
    Generates plots to find the best HPs for each defense.
    """
    print("\n--- Plotting Defense Tuning Trade-offs ---")

    # Plot FLTrust Tuning
    df_fltrust = df[df['defense'] == 'fltrust'].copy()
    if not df_fltrust.empty:
        df_fltrust['clip_norm'] = df_fltrust['clip_norm'].fillna('None')
        g = sns.catplot(
            data=df_fltrust,
            x='clip_norm',
            y='defense_score',
            col='dataset',
            row='attack_type',
            kind='bar',
            height=3, aspect=1.2,
            order=['3.0', '5.0', '10.0', 'None']
        )
        g.fig.suptitle("FLTrust Tuning: Defense Score (Acc - ASR)", y=1.03)
        g.set_axis_labels("Clip Norm", "Defense Score (Higher is Better)")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        plot_file = output_dir / "plot_tuning_fltrust.png"
        plt.savefig(plot_file)
        print(f"Saved FLTrust tuning plot: {plot_file}")
        plt.clf()

    # Plot MartFL Tuning
    df_martfl = df[df['defense'] == 'martfl'].copy()
    if not df_martfl.empty:
        df_martfl['clip_norm'] = df_martfl['clip_norm'].fillna('None')
        g = sns.catplot(
            data=df_martfl,
            x='clip_norm',
            y='defense_score',
            hue='max_k',
            col='dataset',
            row='attack_type',
            kind='bar',
            height=3, aspect=1.2
        )
        g.fig.suptitle("MartFL Tuning: Defense Score (Acc - ASR)", y=1.03)
        g.set_axis_labels("Clip Norm", "Defense Score (Higher is Better)")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.add_legend(title='Max K')
        plot_file = output_dir / "plot_tuning_martfl.png"
        plt.savefig(plot_file)
        print(f"Saved MartFL tuning plot: {plot_file}")
        plt.clf()

    # Plot SkyMask Tuning
    df_skymask = df[df['defense'] == 'skymask'].copy()
    if not df_skymask.empty:
        g = sns.catplot(
            data=df_skymask,
            x='mask_lr',
            y='defense_score',
            hue='mask_threshold',
            col='dataset',
            row='mask_epochs',
            kind='bar',
            height=3, aspect=1.2
        )
        g.fig.suptitle("SkyMask Tuning: Defense Score (Acc - ASR)", y=1.03)
        g.set_axis_labels("Mask LR", "Defense Score (Higher is Better)")
        g.set_titles(col_template="{col_name}", row_template="Epochs={row_name}")
        g.add_legend(title='Mask Threshold')
        plot_file = output_dir / "plot_tuning_skymask.png"
        plt.savefig(plot_file)
        print(f"Saved SkyMask tuning plot: {plot_file}")
        plt.clf()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # Call the plotters
    plot_skymask_failure_analysis(df, output_dir)
    plot_tuning_tradeoffs(df, output_dir)

    print("\nAnalysis complete. Check 'step3_figures' folder for plots.")


if __name__ == "__main__":
    main()