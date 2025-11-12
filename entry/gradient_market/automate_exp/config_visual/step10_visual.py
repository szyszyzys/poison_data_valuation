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
FIGURE_OUTPUT_DIR = "./step10_figures"


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'n_sellers_10')
    """
    hps = {}
    pattern = r'n_sellers_([0-9]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['n_sellers'] = int(match.group(1))
    else:
        print(f"Warning: Could not parse HP suffix '{hp_folder_name}'")
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step10_scalability_martfl_cifar100')"""
    try:
        pattern = r'step10_scalability_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "dataset": match.group(2),
                "attack": "backdoor"  # Inferred from your generator script
            }
        else:
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

    scenario_folders = [f for f in base_path.glob("step10_scalability_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step10_scalability_*' directories found directly inside {base_path}.")
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
                        **run_metrics
                    })
            except Exception as e:
                print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_scalability_lines(df: pd.DataFrame, attack_type: str, output_dir: Path):
    """
    Generates the 2x2 multi-panel line plots for scalability.
    """
    print(f"\n--- Plotting Scalability for {attack_type} attack ---")

    metrics_to_plot = [
        'acc', 'asr',
        'benign_selection_rate', 'adv_selection_rate'
    ]
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    if not metrics_to_plot:
        print("No metrics found to plot.")
        return

    plot_df = df.melt(
        id_vars=['n_sellers', 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    # Rename for clearer labels
    plot_df['Metric'] = plot_df['Metric'].replace({
        'acc': '1. Model Accuracy (Utility)',
        'asr': '2. Attack Success Rate (Defense)',
        'adv_selection_rate': '3. Attacker Selection (%) (Mechanism)',
        'benign_selection_rate': '4. Benign Selection (%) (Fairness)',
    })
    plot_df = plot_df.sort_values(by='Metric')

    g = sns.relplot(
        data=plot_df,
        x='n_sellers',
        y='Value',
        hue='defense',
        style='defense',
        col='Metric',
        kind='line',
        col_wrap=2,  # 2x2 grid
        height=4,
        aspect=1.2,
        facet_kws={'sharey': False},
        markers=True,
        dashes=False
    )

    g.fig.suptitle(f'Defense Scalability vs. Marketplace Size ({attack_type.title()} Attack)', y=1.05)
    g.set_axis_labels("Number of Sellers (n_sellers)", "Value")
    g.set_titles(col_template="{col_name}")

    plot_file = output_dir / f"plot_scalability_{attack_type.upper()}.png"
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

    # Plot for each attack type found (likely just 'backdoor')
    for attack in df['attack'].unique():
        attack_df = df[df['attack'] == attack].copy()
        plot_scalability_lines(attack_df, attack, output_dir)

    print("\nAnalysis complete. Check 'step10_figures' folder for plots.")


if __name__ == "__main__":
    main()