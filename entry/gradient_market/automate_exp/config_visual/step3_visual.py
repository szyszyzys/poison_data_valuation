import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---

# !IMPORTANT: Set this to your main results directory
BASE_RESULTS_DIR = "./results_new"  # Path from your generator script

# !NEW: Set your minimum acceptable accuracy threshold here.
# Runs below this accuracy will be considered 'unreasonable' and
# will be filtered out before selecting the "best" defense.
# [cite_start]Based on your CIFAR10 run[cite: 1], 0.70 (70%) seems like a reasonable start.
REASONABLE_ACC_THRESHOLD = 0.70


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name into a dictionary.
    This logic MUST mirror your generator's 'hp_parts' logic.
    """
    hps = {}
    parts = hp_folder_name.split('_')
    i = 0
    while i < len(parts):
        # This is a simple parser. It looks for known prefixes.
        # Example: 'aggregation.martfl.max_k' becomes 'martfl.max_k'
        if "max_k" in parts[i]:
            hps['martfl.max_k'] = int(parts[i + 1])
            i += 2
        elif "clip_norm" in parts[i]:
            hps['clip_norm'] = None if parts[i + 1] == "None" else float(parts[i + 1])
            i += 2
        elif "mask_epochs" in parts[i]:
            hps['skymask.mask_epochs'] = int(parts[i + 1])
            i += 2
        elif "mask_lr" in parts[i]:
            hps['skymask.mask_lr'] = float(parts[i + 1])
            i += 2
        elif "mask_threshold" in parts[i]:
            hps['skymask.mask_threshold'] = float(parts[i + 1])
            i += 2
        else:
            i += 1  # Move to the next part if unrecognized
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses the base scenario name (e.g., 'step3_tune_martfl_backdoor_image_CIFAR10_cifar10_cnn_new')
    """
    try:
        parts = scenario_name.split('_')
        return {
            "scenario": scenario_name,
            "defense": parts[2],
            "attack": parts[3],
            "modality": parts[4],
            "dataset": parts[5],
            "model": parts[6],
        }
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads key data from one run's final_metrics.json and marketplace_report.json
    """
    run_data = {}
    try:
        # [cite_start]Load final metrics [cite: 1]
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['loss'] = metrics.get('loss', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        # [cite_start]Load marketplace report [cite: 2]
        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            if adv_sellers:
                # [cite_start]Get avg selection rate for adversaries [cite: 2]
                run_data['adv_selection_rate'] = pd.Series([s['selection_rate'] for s in adv_sellers]).mean()
            if ben_sellers:
                # [cite_start]Get avg selection rate for benign sellers [cite: 2]
                run_data['ben_selection_rate'] = pd.Series([s['selection_rate'] for s in ben_sellers]).mean()

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks the results directory and aggregates all run data.
    """
    all_runs = []
    base_path = Path(base_dir)

    print(f"Searching for results in {base_path.resolve()}...")

    for metrics_file in base_path.rglob("final_metrics.json"):
        try:
            hp_folder_name = metrics_file.parent.name
            scenario_name = metrics_file.parent.parent.name

            run_hps = parse_hp_suffix(hp_folder_name)
            run_scenario = parse_scenario_name(scenario_name)
            run_metrics = load_run_data(metrics_file)

            if run_metrics:
                all_runs.append({
                    **run_scenario,
                    **run_hps,
                    **run_metrics,
                    "hp_suffix": hp_folder_name
                })
        except Exception as e:
            print(f"Error processing path {metrics_file}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were found. Did you set BASE_RESULTS_DIR correctly?")
        return pd.DataFrame()

    return pd.DataFrame(all_runs)


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str):
    """
    Generates plots for a specific scenario to compare HP settings.
    """
    scenario_df = df[df['scenario'] == scenario].copy()

    if scenario_df.empty:
        print(f"No data found for scenario '{scenario}'")
        return

    print(f"\n--- Visualizing Scenario: {scenario} ---")

    # === 1. Plot for Objective 1: Best Performance Trade-off ===
    # Melt for grouped bar plot

    # Check if 'adv_selection_rate' column exists
    metrics_to_plot = ['acc', 'asr']
    if 'adv_selection_rate' in scenario_df.columns:
        metrics_to_plot.append('adv_selection_rate')

    plot_df = scenario_df.melt(
        id_vars=['hp_suffix'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(16, 6))
    sns.barplot(
        data=plot_df,
        x='hp_suffix',
        y='Value',
        hue='Metric'
    )
    plt.title(f'Performance vs. HPs for {defense} (Scenario: {scenario})')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination')
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plot_file = f"plot_{scenario}_performance.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()

    # === 2. Plot for Objective 2: Stableness (Parameter Sensitivity) ===
    # This example focuses on 'clip_norm'. You can adapt this for other HPs.
    if 'clip_norm' in scenario_df.columns:
        # Convert 'None' to a numeric value (e.g., 0) for plotting, or handle as category
        scenario_df['clip_norm'] = scenario_df['clip_norm'].fillna('None (auto)')

        # Determine other HPs to create facets (e.g., 'martfl.max_k')
        other_hps = [col for col in scenario_df.columns if col.startswith(defense.lower()) and col != 'clip_norm']

        id_vars = ['clip_norm'] + other_hps
        if not all(c in scenario_df.columns for c in id_vars):
            print(f"Skipping sensitivity plot for {defense}: missing required columns.")
            return

        melted_sensitivity = scenario_df.melt(
            id_vars=id_vars,
            value_vars=['acc', 'asr'],
            var_name='Metric',
            value_name='Value'
        )

        # Use catplot for combined line plots
        g = sns.catplot(
            data=melted_sensitivity,
            x='clip_norm',
            y='Value',
            hue='Metric',
            col=other_hps[0] if other_hps else None,  # Facet by other HPs
            kind='point',
            palette={'acc': 'blue', 'asr': 'red'},
            height=5,
            aspect=1.2
        )

        g.fig.suptitle(f'HP Stability: Sensitivity to "clip_norm" for {defense}', y=1.03)
        g.set_axis_labels('Clip Norm', 'Rate')
        plot_file = f"plot_{scenario}_stability.png"
        g.fig.savefig(plot_file)
        print(f"Saved plot: {plot_file}")
        plt.clf()


def main():
    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        return

    # Display all columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Analysis for Objective 1 (Refined): Best Performance with Filtering ---
    print("\n" + "=" * 80)
    print(f" Objective 1: Finding Best HPs (Filtered by acc >= {REASONABLE_ACC_THRESHOLD})")
    print("=" * 80)

    # 1. Apply the user's filter
    reasonable_df = df[df['acc'] >= REASONABLE_ACC_THRESHOLD].copy()

    if reasonable_df.empty:
        print(f"\n!WARNING: No runs met the accuracy threshold of {REASONABLE_ACC_THRESHOLD}.")
        print("  Consider lowering the threshold to see any results.")
        print("  Displaying all runs instead, sorted by ASR:")
        reasonable_df = df.copy()  # Fallback to all runs

    # 2. Sort the *reasonable* runs:
    #    Primary goal: Lowest ASR
    #    Secondary goal: Lowest Adversary Selection Rate (if available)
    sort_columns = ['scenario', 'asr']
    sort_ascending = [True, True]

    if 'adv_selection_rate' in reasonable_df.columns:
        sort_columns.append('adv_selection_rate')
        sort_ascending.append(True)

    df_sorted = reasonable_df.sort_values(
        by=sort_columns,
        ascending=sort_ascending
    )

    print(f"\n--- Best HP Combinations (acc >= {REASONABLE_ACC_THRESHOLD}) ---")
    print(f"--- Sorted by: 1. ASR (low){', 2. Adv Selection (low)' if 'adv_selection_rate' in df.columns else ''} ---")

    grouped = df_sorted.groupby('scenario')

    for name, group in grouped:
        print(f"\n--- SCENARIO: {name} ---")
        # Define columns to show
        cols_to_show = [
            'hp_suffix',
            'acc',  # The utility
            'asr',  # The primary defense metric
            'adv_selection_rate',  # The secondary defense metric
        ]
        # Handle missing adv_selection_rate if marketplace_report failed
        cols_present = [c for c in cols_to_show if c in group.columns]
        print(group[cols_present].to_string(index=False))

    # --- Analysis for Objective 2: Stableness (Visualization) ---
    print("\n" + "=" * 80)
    print("           Objective 2: Assessing Defense Stability (Plots)")
    print("=" * 80)

    # We use the *original* full dataframe (df) for plotting stability,
    # as it's crucial to see the HPs that *caused* the accuracy to drop.
    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        plot_defense_comparison(df, scenario, defense)

    print("\nAnalysis complete. Check for 'plot_*.png' files.")


if __name__ == "__main__":
    main()