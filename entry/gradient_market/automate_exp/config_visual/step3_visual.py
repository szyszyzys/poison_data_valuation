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
        # Load final metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['loss'] = metrics.get('loss', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        # Load marketplace report
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
    plot_df = scenario_df.melt(
        id_vars=['hp_suffix'],
        value_vars=['acc', 'asr', 'adv_selection_rate'],
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

    # --- Analysis for Objective 1: Best Performance ---
    print("\n" + "=" * 80)
    print("           Objective 1: Finding Best HP per Scenario")
    print("=" * 80)

    # We want to MINIMIZE ASR and MAXIMIZE ACC
    # A simple score: acc * (1 - asr). Higher is better.
    df['tradeoff_score'] = df['acc'] * (1 - df['asr'])

    # Sort to find the best HPs
    df_sorted = df.sort_values(by=['scenario', 'tradeoff_score'], ascending=[True, False])

    print("\n--- Best HP Combination per Scenario (sorted by tradeoff_score = acc * (1-asr)) ---")

    # Group by scenario and show the top-performing HPs
    grouped = df_sorted.groupby('scenario')

    for name, group in grouped:
        print(f"\n--- SCENARIO: {name} ---")
        print(group[['hp_suffix', 'acc', 'asr', 'adv_selection_rate', 'tradeoff_score']].to_string(index=False))

    # --- Analysis for Objective 2: Stableness (Visualization) ---
    print("\n" + "=" * 80)
    print("           Objective 2: Assessing Defense Stability (Plots)")
    print("=" * 80)

    # Generate plots for each unique scenario
    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        plot_defense_comparison(df_sorted, scenario, defense)

    print("\nAnalysis complete. Check for 'plot_*.png' files.")


if __name__ == "__main__":
    main()