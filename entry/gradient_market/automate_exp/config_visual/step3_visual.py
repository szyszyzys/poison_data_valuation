import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"

# Set your minimum acceptable accuracy threshold (e.g., 0.70 for 70%)
REASONABLE_ACC_THRESHOLD = 0.70

# Set your maximum acceptable False Positive Rate (Benign Filter Rate)
# (e.g., 0.20 means you don't filter more than 20% of good clients)
REASONABLE_FP_THRESHOLD = 0.20


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses the HP suffix folder name into a dictionary."""
    hps = {}
    parts = hp_folder_name.split('_')
    i = 0
    while i < len(parts):
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
            i += 1
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name."""
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
    Loads key data from final_metrics.json and marketplace_report.json
    """
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['loss'] = metrics.get('loss', 0)
        # !NEW: Load the completed rounds
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            if adv_sellers:
                run_data['adv_outlier_rate'] = pd.Series([s['outlier_rate'] for s in adv_sellers]).mean()
            if ben_sellers:
                run_data['benign_outlier_rate'] = pd.Series([s['outlier_rate'] for s in ben_sellers]).mean()

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

    return pd.DataFrame(all_runs)


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str):
    """Generates plots for a specific scenario to compare HP settings."""
    scenario_df = df[df['scenario'] == scenario].copy()

    if scenario_df.empty:
        print(f"No data found for scenario '{scenario}'")
        return

    print(f"\n--- Visualizing Scenario: {scenario} ---")

    # === 1. Plot for Objective 1: Best Performance Trade-off ===
    metrics_to_plot = ['acc', 'asr']
    if 'adv_outlier_rate' in scenario_df.columns:
        metrics_to_plot.append('adv_outlier_rate')
    if 'benign_outlier_rate' in scenario_df.columns:
        metrics_to_plot.append('benign_outlier_rate')

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
        hue='Metric'
    )
    plt.title(f'Performance & Filtering vs. HPs for {defense} (Scenario: {scenario})')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination')
    plt.xticks(rotation=20, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plot_file = f"plot_{scenario}_performance.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()

    # === 2. Plot for Objective 2: Stableness (Parameter Sensitivity) ===
    if 'clip_norm' in scenario_df.columns:
        scenario_df['clip_norm'] = scenario_df['clip_norm'].fillna('None (auto)')

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

        g = sns.catplot(
            data=melted_sensitivity,
            x='clip_norm',
            y='Value',
            hue='Metric',
            col=other_hps[0] if other_hps else None,
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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Analysis for Objective 1 (Refined): Best Performance with Filtering ---
    print("\n" + "=" * 80)
    print(f" Objective 1: Finding Best HPs (Filtered by acc >= {REASONABLE_ACC_THRESHOLD})")
    print("=" * 80)

    # 1. Apply the user's accuracy filter
    reasonable_acc_df = df[df['acc'] >= REASONABLE_ACC_THRESHOLD].copy()

    if reasonable_acc_df.empty:
        print(f"\n!WARNING: No runs met the accuracy threshold of {REASONABLE_ACC_THRESHOLD}.")
        print("  Falling back to all runs for analysis.")
        reasonable_acc_df = df.copy()

    # 2. Apply the user's False Positive filter
    if 'benign_outlier_rate' in reasonable_acc_df.columns:
        reasonable_final_df = reasonable_acc_df[
            reasonable_acc_df['benign_outlier_rate'] <= REASONABLE_FP_THRESHOLD
            ].copy()

        if reasonable_final_df.empty:
            print(f"\n!WARNING: No runs passed the FP threshold of {REASONABLE_FP_THRESHOLD} (after passing accuracy).")
            print("  Falling back to only accuracy-filtered runs.")
            reasonable_final_df = reasonable_acc_df.copy()
    else:
        print("\n!WARNING: 'benign_outlier_rate' not found. Skipping False Positive filter.")
        reasonable_final_df = reasonable_acc_df.copy()

    # 3. Sort the *reasonable* runs:
    sort_columns = ['scenario', 'asr']
    sort_ascending = [True, True]

    if 'adv_outlier_rate' in reasonable_final_df.columns:
        sort_columns.append('adv_outlier_rate')
        sort_ascending.append(False)  # We want this HIGH, so ascending=False

    df_sorted = reasonable_final_df.sort_values(
        by=sort_columns,
        ascending=sort_ascending
    )

    print(
        f"\n--- Best HP Combinations (acc >= {REASONABLE_ACC_THRESHOLD}, benign_filter <= {REASONABLE_FP_THRESHOLD}) ---")
    print(f"--- Sorted by: 1. ASR (low){', 2. Adv Filter (high)' if 'adv_outlier_rate' in df.columns else ''} ---")

    grouped = df_sorted.groupby('scenario')

    for name, group in grouped:
        print(f"\n--- SCENARIO: {name} ---")
        # !NEW: Added 'rounds' to the list
        cols_to_show = [
            'hp_suffix',
            'acc',  # Utility
            'asr',  # Defense Outcome
            'adv_outlier_rate',  # Defense Mechanism (Filter Adversaries)
            'benign_outlier_rate',  # Defense Mechanism (False Positives)
            'rounds',  # Stability/Efficiency
        ]
        cols_present = [c for c in cols_to_show if c in group.columns]
        print(group[cols_present].to_string(index=False))

    # --- Analysis for Objective 2: Stableness (Visualization) ---
    print("\n" + "=" * 80)
    print("           Objective 2: Assessing Defense Stability (Plots)")
    print("=" * 80)

    # We use the *original* full dataframe (df) for plotting stability
    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        plot_defense_comparison(df, scenario, defense)

    print("\nAnalysis complete. Check for 'plot_*.png' files.")


if __name__ == "__main__":
    main()