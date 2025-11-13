import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./step3_figures"

# Set your minimum acceptable accuracy threshold (e.g., 0.70 for 70%)
REASONABLE_ACC_THRESHOLD = 0.70

# Set your minimum acceptable Benign Selection Rate (e.g., 0.50 for 50%)
REASONABLE_BSR_THRESHOLD = 0.50


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    (FIXED) Parses the HP suffix folder name using regex.
    This is robust to underscores in parameter names.
    """
    hps = {}

    # Define the regex patterns for each HP we expect, based on the generator script
    patterns = {
        'martfl.max_k': r'aggregation\.martfl\.max_k_([0-9]+)',
        'clip_norm': r'aggregation\.clip_norm_([0-9\.]+|None)',
        'skymask.mask_epochs': r'aggregation\.skymask\.mask_epochs_([0-9]+)',
        'skymask.mask_lr': r'aggregation\.skymask\.mask_lr_([0-9\.]+)',
        'skymask.mask_threshold': r'aggregation\.skymask\.mask_threshold_([0-9\.]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, hp_folder_name)
        if match:
            value_str = match.group(1)
            if value_str == "None":
                hps[key] = None
            else:
                # Try to cast to float (which covers int)
                try:
                    hps[key] = float(value_str)
                    # If it's a whole number, store as int for cleaner grouping
                    if hps[key] == int(hps[key]):
                        hps[key] = int(hps[key])
                except ValueError:
                    hps[key] = value_str  # Fallback for any other string
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    (FIXED) Parses the base scenario name using regex to handle
    underscores in dataset and model names.
    e.g., 'step3_tune_martfl_labelflip_tabular_Texas100_mlp_texas100_baseline_new'
    """
    try:
        # Regex breakdown:
        # 1. (fedavg|martfl|fltrust|skymask): Captures the defense
        # 2. ([a-z]+): Captures the attack type (e.g., labelflip)
        # 3. (image|text|tabular): Captures the modality
        # 4. (.+?): Non-greedy match for the dataset (e.g., Texas100)
        # 5. (.+): Greedy match for the model name (e.g., mlp_texas100_baseline)
        pattern = r'step3_tune_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(image|text|tabular)_(.+?)_(.+)_(new|old)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "attack": match.group(2),
                "modality": match.group(3),
                "dataset": match.group(4),
                "model": match.group(5),
            }
        else:
            # Fallback for old names if needed, or just raise error
            raise ValueError(f"Pattern not matched for: {scenario_name}")

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
        run_data['loss'] = metrics.get('loss', 0)
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

    scenario_folders = [f for f in base_path.glob("step3_tune_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step3_tune_*' directories found directly inside {base_path}.")
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

    # Convert to DataFrame and fill missing HP columns with NaN
    # This ensures 'clip_norm' exists for all, even if a defense didn't tune it
    df = pd.DataFrame(all_runs)
    return df


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str, output_dir: Path):
    """Generates plots for a specific scenario to compare HP settings."""
    scenario_df = df[df['scenario'] == scenario].copy()

    if scenario_df.empty:
        print(f"No data found for scenario '{scenario}'")
        return

    print(f"\n--- Visualizing Scenario: {scenario} ---")

    # --- 1. Plot for Objective 1: Best Performance Trade-off ---

    # Handle potentially missing clip_norm (though parser fix should prevent this)
    if 'clip_norm' not in scenario_df.columns:
        scenario_df['clip_norm'] = np.nan

    scenario_df['clip_norm_sort'] = scenario_df['clip_norm'].fillna(np.inf)

    other_hp_cols = [col for col in scenario_df.columns if col.startswith(defense.lower())]
    sort_cols = [c for c in other_hp_cols if c != 'clip_norm'] + ['clip_norm_sort']
    sort_cols = [c for c in sort_cols if c in scenario_df.columns]

    if sort_cols:
        scenario_df = scenario_df.sort_values(by=sort_cols)

    metrics_to_plot = ['acc', 'asr']
    if 'adv_selection_rate' in scenario_df.columns:
        metrics_to_plot.append('adv_selection_rate')
    if 'benign_selection_rate' in scenario_df.columns:
        metrics_to_plot.append('benign_selection_rate')

    plot_df = scenario_df.melt(
        id_vars=['hp_suffix'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(16, 7))
    sns.barplot(
        data=plot_df,
        x='hp_suffix',
        y='Value',
        hue='Metric',
        order=scenario_df['hp_suffix'].unique()
    )
    plt.title(f'Performance & Selection vs. HPs for {defense} (Scenario: {scenario})')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination (Sorted by Value)')
    plt.xticks(rotation=20, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plot_file = output_dir / f"plot_{scenario}_performance.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()

    # --- 2. Plot for Objective 2: Stableness (Parameter Sensitivity) ---
    if 'clip_norm' in scenario_df.columns and not scenario_df['clip_norm'].isnull().all():

        unique_clips = sorted(scenario_df['clip_norm'].fillna(np.inf).unique())
        clip_order_labels = [str(c) if c != np.inf else "None (auto)" for c in unique_clips]

        scenario_df['clip_norm_str'] = scenario_df['clip_norm'].fillna("None (auto)")

        other_hps = [col for col in scenario_df.columns if col.startswith(defense.lower()) and col != 'clip_norm']

        id_vars = ['clip_norm_str'] + other_hps
        if not all(c in scenario_df.columns for c in id_vars):
            print(f"Skipping sensitivity plot for {defense}: missing required columns.")
            return

        melted_sensitivity = scenario_df.melt(
            id_vars=id_vars,
            value_vars=['acc', 'asr'],
            var_name='Metric',
            value_name='Value'
        )

        # Determine the column for faceting (e.g., 'martfl.max_k')
        facet_col = None
        if other_hps:
            # Check for a column that actually has more than one value
            for col in other_hps:
                if col in melted_sensitivity.columns and melted_sensitivity[col].nunique() > 1:
                    facet_col = col
                    break

        g = sns.catplot(
            data=melted_sensitivity,
            x='clip_norm_str',
            y='Value',
            hue='Metric',
            col=facet_col,  # Use the determined facet column
            kind='point',
            palette={'acc': 'blue', 'asr': 'red'},
            order=clip_order_labels,
            height=5,
            aspect=1.2
        )

        g.fig.suptitle(f'HP Stability: Sensitivity to "clip_norm" for {defense}', y=1.03)
        g.set_axis_labels('Clip Norm', 'Rate')
        plot_file = output_dir / f"plot_{scenario}_stability.png"
        g.fig.savefig(plot_file)
        print(f"Saved plot: {plot_file}")
        plt.clf()
    else:
        print(f"Skipping 'clip_norm' stability plot for {defense} (no clip_norm data found).")


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

    # --- Analysis for Objective 1: Best Performance with Filtering ---
    print("\n" + "=" * 80)
    print(f" Objective 1: Finding Best HPs (Filtered by acc >= {REASONABLE_ACC_THRESHOLD})")
    print("=" * 80)

    # 1. Apply accuracy filter
    reasonable_acc_df = df[df['acc'] >= REASONABLE_ACC_THRESHOLD].copy()

    if reasonable_acc_df.empty:
        print(f"\n!WARNING: No runs met the accuracy threshold of {REASONABLE_ACC_THRESHOLD}.")
        print("  Falling back to all runs for analysis.")
        reasonable_acc_df = df.copy()

    # 2. Apply Benign Selection Rate (Fairness) filter
    if 'benign_selection_rate' in reasonable_acc_df.columns:
        reasonable_final_df = reasonable_acc_df[
            reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD
            ].copy()

        if reasonable_final_df.empty:
            print(
                f"\n!WARNING: No runs passed the BSR threshold of {REASONABLE_BSR_THRESHOLD} (after passing accuracy).")
            print("  Falling back to only accuracy-filtered runs.")
            reasonable_final_df = reasonable_acc_df.copy()
    else:
        print("\n!WARNING: 'benign_selection_rate' not found. Skipping Fairness filter.")
        reasonable_final_df = reasonable_acc_df.copy()

    # 3. Create a unified sort metric
    reasonable_final_df['sort_metric'] = np.where(
        reasonable_final_df['attack'] == 'backdoor',
        reasonable_final_df['asr'],  # Low is good
        1.0 - reasonable_final_df['acc']  # Low is good (high acc)
    )

    # 4. Sort the *reasonable* runs:
    sort_columns = ['scenario', 'sort_metric']
    sort_ascending = [True, True]

    if 'adv_selection_rate' in reasonable_final_df.columns:
        sort_columns.append('adv_selection_rate')
        sort_ascending.append(True)  # We want LOW adv_selection_rate

    df_sorted = reasonable_final_df.sort_values(
        by=sort_columns,
        ascending=sort_ascending
    )

    print(
        f"\n--- Best HP Combinations (acc >= {REASONABLE_ACC_THRESHOLD}, benign_select_rate >= {REASONABLE_BSR_THRESHOLD}) ---")
    print(f"--- Sorted by: 1. Best Defense (Low ASR or High ACC), 2. Low Adv. Selection ---")

    grouped = df_sorted.groupby('scenario')

    for name, group in grouped:
        print(f"\n--- SCENARIO: {name} ---")
        cols_to_show = [
            'hp_suffix',
            'acc',  # Utility
            'asr',  # Defense Outcome (Backdoor)
            'adv_selection_rate',  # Defense Outcome (Mechanism)
            'benign_selection_rate',  # Defense Outcome (Fairness)
            'rounds',  # Stability/Efficiency
        ]
        cols_present = [c for c in group.columns if c in cols_to_show]
        print(group[cols_present].to_string(index=False, float_format="%.4f"))

    # --- Analysis for Objective 2: Stableness (Visualization) ---
    print("\n" + "=" * 80)
    print("           Objective 2: Assessing Defense Stability (Plots)")
    print("=" * 80)

    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        plot_defense_comparison(df, scenario, defense, output_dir)

    print("\nAnalysis complete. Check 'step3_figures' folder for plots.")


if __name__ == "__main__":
    main()
