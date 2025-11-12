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
FIGURE_OUTPUT_DIR = "./step5_figures"


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'adv_0.1_poison_0.5')
    """
    hps = {}
    pattern = r'adv_([0-9\.]+)_poison_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['adv_rate'] = float(match.group(1))
        hps['poison_rate'] = float(match.group(2))
    else:
        print(f"Warning: Could not parse HP suffix '{hp_folder_name}'")
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step5_atk_sens_adv_martfl_backdoor_image')"""
    try:
        parts = scenario_name.split('_')
        return {
            "scenario": scenario_name,
            "sweep_type": parts[3],  # 'adv' or 'poison'
            "defense": parts[4],
            "attack": parts[5],
            "modality": parts[6],
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

    scenario_folders = [f for f in base_path.glob("step5_atk_sens_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step5_atk_sens_*' directories found directly inside {base_path}.")
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


def plot_sensitivity_lines(df: pd.DataFrame, x_metric: str, attack_type: str, output_dir: Path):
    """
    Generates the multi-panel line plots for robustness.
    """
    print(f"\n--- Plotting Robustness vs. '{x_metric}' (for {attack_type} attack) ---")

    # --- THIS IS THE UPDATED METRIC LIST ---
    if attack_type == 'backdoor':
        # For backdoor, we care about acc, asr, fairness, and stability
        metrics_to_plot = ['acc', 'asr', 'benign_selection_rate', 'rounds']
    elif attack_type == 'labelflip':
        # For labelflip, 'asr' is not used. 'acc' *is* the defense metric.
        metrics_to_plot = ['acc', 'benign_selection_rate', 'rounds']
    else:
        print(f"Skipping plot for unknown attack type: {attack_type}")
        return
    # --- END OF UPDATE ---

    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    if not metrics_to_plot:
        print("No metrics found to plot.")
        return

    plot_df = df.melt(
        id_vars=[x_metric, 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    # Calculate grid size, ensuring we have max 2 columns
    n_metrics = len(metrics_to_plot)
    col_wrap = 2 if n_metrics > 1 else 1

    g = sns.relplot(
        data=plot_df,
        x=x_metric,
        y='Value',
        hue='defense',
        style='defense',
        col='Metric',
        kind='line',
        col_wrap=col_wrap,  # Use dynamic col_wrap
        height=4,
        aspect=1.2,
        facet_kws={'sharey': False},
        markers=True,
        dashes=False
    )

    g.fig.suptitle(f'Defense Robustness vs. {x_metric.replace("_", " ").title()} ({attack_type.title()} Attack)',
                   y=1.05)
    g.set_axis_labels(x_metric.replace("_", " ").title(), "Value")
    g.set_titles(col_template="{col_name}")

    plot_file = output_dir / f"plot_robustness_{attack_type.upper()}_vs_{x_metric}.png"
    g.fig.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        return

    attack_types = df['attack'].unique()

    for attack in attack_types:
        attack_df = df[df['attack'] == attack].copy()

        # Plot 1: Sweep vs. Adversary Rate
        adv_rate_df = attack_df[attack_df['sweep_type'] == 'adv']
        if not adv_rate_df.empty:
            plot_sensitivity_lines(adv_rate_df, 'adv_rate', attack, output_dir)
        else:
            print(f"No data found for {attack} vs. adv_rate sweep.")

        # Plot 2: Sweep vs. Poison Rate
        poison_rate_df = attack_df[attack_df['sweep_type'] == 'poison']
        if not poison_rate_df.empty:
            plot_sensitivity_lines(poison_rate_df, 'poison_rate', attack, output_dir)
        else:
            print(f"No data found for {attack} vs. poison_rate sweep.")

    print("\nAnalysis complete. Check 'step5_figures' folder for plots.")


if __name__ == "__main__":
    main()