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
FIGURE_OUTPUT_DIR = "./step6_figures"


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'adv_0.3_poison_0.5_blend_alpha_0.1')
    """
    hps = {}

    # adv_rate and poison_rate are always present
    adv_match = re.search(r'adv_([0-9\.]+)', hp_folder_name)
    poison_match = re.search(r'poison_([0-9\.]+)', hp_folder_name)
    if adv_match:
        hps['adv_rate'] = float(adv_match.group(1))
    if poison_match:
        hps['poison_rate'] = float(poison_match.group(1))

    # blend_alpha is optional
    blend_match = re.search(r'blend_alpha_([0-9\.]+)', hp_folder_name)
    if blend_match:
        hps['blend_alpha'] = float(blend_match.group(1))

    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step6_adv_sybil_oracle_blend_martfl')"""
    try:
        # This regex captures the strategy name (which could have underscores)
        # and the defense name (which is the last part)
        match = re.search(r'step6_adv_sybil_(.+)_(fedavg|martfl|fltrust|skymask)', scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "strategy": match.group(1),  # e.g., 'baseline_no_sybil', 'oracle_blend'
                "defense": match.group(2),  # e.g., 'martfl'
            }
        else:
            raise ValueError("Pattern not matched")
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

    # Look for 'step6_adv_sybil_*' directories
    scenario_folders = [f for f in base_path.glob("step6_adv_sybil_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step6_adv_sybil_*' directories found directly inside {base_path}.")
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

    df = pd.DataFrame(all_runs)
    return df


def plot_sybil_comparison(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    """
    (UPDATED)
    Generates the grouped bar chart comparing Sybil strategies for a single defense.
    Assumes 'defense_df' is already filtered for the correct defense
    and has the 'strategy_label' column.
    """
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- Logical Sorting for the X-axis ---
    # Define the desired order
    def get_sort_key(label):
        if label == 'baseline_no_sybil':
            return (0, 0.0)
        if label.startswith('oracle_blend'):
            try:
                alpha = float(label.split('_')[-1])
                return (2, alpha)
            except:
                return (2, np.inf)  # Failsafe
        if label == 'mimic':
            return (1, 0.0)
        # We no longer need the 'systematic_probe' key as it's filtered out
        return (4, 0.0)  # Other strategies

    # Get unique labels and sort them
    unique_labels = defense_df['strategy_label'].unique()
    sorted_labels = sorted(unique_labels, key=get_sort_key)
    # --- End Sorting ---

    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    metrics_to_plot = [m for m in metrics_to_plot if m in defense_df.columns]

    # Melt the data for plotting.
    # The 'defense_df' is already pre-aggregated (mean over seeds).
    plot_df = defense_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(16, 7))
    # --- Use `dodge=True` (default) for grouped bars ---
    # Note: `sns.barplot` will show mean + confidence interval if multiple seeds are present.
    # Since we pre-aggregated, it will just show the mean value.
    sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels  # Apply the logical sort order
    )
    plt.title(f'Effectiveness of Sybil Attacks against {defense.upper()} Defense')
    plt.ylabel('Rate')
    plt.xlabel('Sybil Attack Strategy')
    plt.xticks(rotation=20, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()

    plot_file = output_dir / f"plot_sybil_effectiveness_{defense}.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # --- FIX 1: Filter out 'systematic_probe' ---
    df = df[df['strategy'] != 'systematic_probe'].copy()
    print(f"\nFiltered out 'systematic_probe'. Remaining runs: {len(df)}")

    # --- FIX 2: Create the strategy_label column once ---
    df['strategy_label'] = df['strategy']
    blend_rows = df['blend_alpha'].notna()
    df.loc[blend_rows, 'strategy_label'] = df.loc[blend_rows].apply(
        lambda row: f"oracle_blend_{row['blend_alpha']}", axis=1
    )

    # --- FIX 3: Generate the aggregated CSV ---
    print("\n--- Generating Aggregated Summary CSV ---")
    metrics_to_agg = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate', 'rounds']
    metrics_to_agg = [m for m in metrics_to_agg if m in df.columns]

    # Aggregate by defense and the new strategy label
    # This averages over all seeds for each strategy
    df_agg = df.groupby(['defense', 'strategy_label'])[metrics_to_agg].mean()
    df_agg = df_agg.reset_index()

    # Save to CSV
    csv_path = output_dir / "step6_sybil_summary.csv"
    df_agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✅ Saved aggregated summary to: {csv_path}")
    # --- END OF FIXES ---

    # Get all unique defenses
    defenses = df_agg['defense'].unique()

    for defense in defenses:
        # Pass the pre-aggregated and pre-labeled dataframe chunk
        plot_sybil_comparison(df_agg[df_agg['defense'] == defense].copy(), defense, output_dir)

    print("\nAnalysis complete. Check 'step6_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()