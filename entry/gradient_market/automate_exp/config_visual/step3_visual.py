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
FIGURE_OUTPUT_DIR = "./step3_figures"

# Set your minimum acceptable accuracy threshold (e.g., 0.70 for 70%)
REASONABLE_ACC_THRESHOLD = 0.70

# Set your minimum acceptable Benign Selection Rate (e.g., 0.50 for 50%)
REASONABLE_BSR_THRESHOLD = 0.50


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    (FIXED) Parses the HP suffix folder name using regex.
    This is robust to underscores in parameter names and scientific notation.
    """
    hps = {}

    # Define the regex patterns for each HP we expect, based on the generator script
    patterns = {
        'martfl.max_k': r'aggregation\.martfl\.max_k_([0-9]+)',
        'clip_norm': r'aggregation\.clip_norm_([0-9\.]+|None)',
        'mask_epochs': r'aggregation\.skymask\.mask_epochs_([0-9]+)',
        'mask_lr': r'aggregation\.skymask\.mask_lr_([0-9e\.\+]+)',  # Handles 0.01 and 1e7
        'mask_threshold': r'aggregation\.skymask\.mask_threshold_([0-9\.]+)',
        'mask_clip': r'aggregation\.skymask\.mask_clip_([0-9e\.\-]+)'  # Handles 10.0 and 1e-7
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, hp_folder_name)
        if match:
            value_str = match.group(1)

            # Clean up key name
            clean_key = key.split('.')[-1]  # e.g., 'skymask.mask_epochs' -> 'mask_epochs'

            if value_str == "None":
                hps[clean_key] = 'None'  # Use string 'None' for grouping
            else:
                # Try to cast to float (which covers int and sci-notation)
                try:
                    val_float = float(value_str)
                    # If it's a whole number, store as int for cleaner grouping
                    if val_float == int(val_float):
                        hps[clean_key] = int(val_float)
                    else:
                        hps[clean_key] = val_float
                except ValueError:
                    hps[clean_key] = value_str  # Fallback
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    (FIXED) Parses the base scenario name using regex to handle
    underscores in dataset and model names.
    e.g., 'step3_tune_martfl_labelflip_tabular_Texas100_mlp_texas100_baseline_new'
    """
    try:
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

            # Handle 0-attacker case (adv_selection_rate will be NaN)
            if not adv_sellers and ben_sellers:
                run_data['adv_selection_rate'] = 0.0  # No adversaries, so 0% selected

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    # Find the new folders (and old ones, just in case)
    scenario_folders = list(base_path.glob("step3_tune_*"))
    if not scenario_folders:
        print(f"Error: No 'step3_tune_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        if not scenario_path.is_dir(): continue
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        # Look inside the HP-specific folders
        for hp_path in scenario_path.iterdir():
            if not hp_path.is_dir(): continue

            hp_folder_name = hp_path.name
            run_hps = parse_hp_suffix(hp_folder_name)

            # Find all seed runs inside
            for metrics_file in hp_path.rglob("final_metrics.json"):
                try:
                    run_metrics = load_run_data(metrics_file)

                    if run_metrics:
                        all_runs.append({
                            **run_scenario,
                            **run_hps,
                            **run_metrics,
                            "hp_suffix": hp_folder_name
                        })
                except Exception as e:
                    print(f"Error processing file {metrics_file}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_runs)

    # --- Create the unified 'defense_score' ---
    # We want HIGH acc and LOW asr
    # We will use this to find the "best" HP combo
    df['defense_score'] = df['acc'] - df['asr']

    return df


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str, output_dir: Path):
    """Generates plots for a specific scenario to compare HP settings."""
    scenario_df = df[df['scenario'] == scenario].copy()

    if scenario_df.empty:
        print(f"No data found for scenario '{scenario}'")
        return

    print(f"\n--- Visualizing Scenario: {scenario} ---")

    # --- 1. Plot for Objective 1: Best Performance Trade-off ---

    # Get all HP columns for this defense
    hp_cols = list(df.filter(regex=f"^{defense}").columns)
    if 'clip_norm' in df.columns and 'clip_norm' not in hp_cols:
        hp_cols.append('clip_norm')
    hp_cols = [c for c in hp_cols if c in scenario_df.columns and scenario_df[c].nunique() > 1]

    # Create a unique, readable x-axis label from the HPs
    if not hp_cols:
        scenario_df['hp_label'] = 'default'
        hp_cols = ['hp_label']
    else:
        # Create a label like "lr:0.01_clip:10.0_..."
        scenario_df['hp_label'] = scenario_df[hp_cols].apply(
            lambda row: '_'.join([f"{col.split('.')[-1]}:{row[col]}" for col in hp_cols]),
            axis=1
        )

    # Sort by the defense_score to make plot readable
    scenario_df = scenario_df.sort_values(by='defense_score', ascending=False)

    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    metrics_to_plot = [m for m in metrics_to_plot if m in scenario_df.columns]

    plot_df = scenario_df.melt(
        id_vars=['hp_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(max(16, 0.5 * scenario_df['hp_label'].nunique()), 7))
    sns.barplot(
        data=plot_df,
        x='hp_label',
        y='Value',
        hue='Metric',
        order=scenario_df['hp_label'].unique()  # Use the sorted order
    )
    plt.title(f'Performance & Selection vs. HPs for {defense.upper()} (Scenario: {scenario})')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination (Sorted by Defense Score)')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.legend(title='Metric')
    plt.tight_layout()
    plot_file = output_dir / f"plot_{scenario}_performance.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')

    # --- 2. Plot for Objective 2: Stableness (Parameter Sensitivity) ---
    # This plots a grid to show how HPs interact

    # This logic is now dynamic
    hp_cols_for_plot = [c for c in hp_cols if c in scenario_df.columns and scenario_df[c].nunique() > 1]
    if len(hp_cols_for_plot) >= 2:
        # We can make a 2D plot
        x_hp = hp_cols_for_plot[0]
        y_hp = hp_cols_for_plot[1]

        # Use remaining HPs for columns/rows
        col_hp = hp_cols_for_plot[2] if len(hp_cols_for_plot) > 2 else None
        row_hp = hp_cols_for_plot[3] if len(hp_cols_for_plot) > 3 else None

        # Plot defense_score, acc, asr, and filter failure
        scenario_df['filter_failure'] = (scenario_df['benign_selection_rate'] >= 0.99) & \
                                        (scenario_df['adv_selection_rate'] >= 0.99)

        metrics_to_plot = ['defense_score', 'acc', 'asr', 'filter_failure']

        plot_df_melted = scenario_df.melt(
            id_vars=[x_hp, y_hp, col_hp, row_hp],
            value_vars=metrics_to_plot,
            var_name='Metric',
            value_name='Value'
        )

        g = sns.catplot(
            data=plot_df_melted,
            x=x_hp,
            y='Value',
            hue=y_hp,
            col=col_hp,
            row='Metric',
            kind='bar',
            palette='viridis',
            height=3,
            aspect=1.2,
            sharey=False,
            legend_out=True
        )
        g.fig.suptitle(f'HP Stability Analysis for {defense.upper()} ({scenario})', y=1.03)
        g.set_axis_labels(x_hp, 'Metric Value')
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        plot_file = output_dir / f"plot_{scenario}_stability_grid.png"
        g.fig.savefig(plot_file)
        print(f"Saved stability grid plot: {plot_file}")
        plt.clf()
        plt.close('all')

    else:
        print(f"Skipping stability grid plot for {defense} (not enough HPs to compare).")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # --- Save the analysis CSV you requested ---
    csv_output_path = output_dir / "step3_full_results_summary.csv"
    df.to_csv(csv_output_path, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_path}\n")
    # -------------------------------------------

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

    # 3. Create a unified sort metric (we want to MINIMIZE this)
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
            'hp_label',  # Use the new human-readable label
            'acc',  # Utility
            'asr',  # Defense Outcome (Backdoor)
            'adv_selection_rate',  # Defense Outcome (Mechanism)
            'benign_selection_rate',  # Defense Outcome (Fairness)
            'rounds',  # Stability/Efficiency
            'defense_score'  # The score we used to sort
        ]
        # Find HPs for this group
        hp_cols = list(df.filter(regex=f"^{group['defense'].iloc[0]}").columns)
        if 'clip_norm' in df.columns: hp_cols.append('clip_norm')
        hp_cols = [c for c in hp_cols if c in group.columns]

        cols_to_show = hp_cols + cols_to_show
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