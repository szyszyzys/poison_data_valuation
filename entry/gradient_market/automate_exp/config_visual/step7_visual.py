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
FIGURE_OUTPUT_DIR = "./step7_figures"


# --- End Configuration ---


def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses the base scenario name
    e.g., 'step7_adaptive_gradient_inversion_gradient_manipulation_martfl_CIFAR100'
    """
    try:
        # Regex to capture the parts
        pattern = r'step7_adaptive_([a-z_]+)_([a-z_]+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            threat_model = match.group(1)
            adaptive_mode = match.group(2)
            defense = match.group(3)
            dataset = match.group(4)

            # Create a clean label for plotting
            if adaptive_mode == 'gradient_manipulation':
                mode_label = 'grad'
            elif adaptive_mode == 'data_poisoning':
                mode_label = 'data'
            else:
                mode_label = adaptive_mode

            return {
                "scenario": scenario_name,
                "threat_model": threat_model,
                "adaptive_mode": adaptive_mode,
                "defense": defense,
                "dataset": dataset,
                "strategy_label": f"{threat_model} ({mode_label})"
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

    scenario_folders = [f for f in base_path.glob("step7_adaptive_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step7_adaptive_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        metrics_files = list(scenario_path.rglob("final_metrics.json"))

        if not metrics_files:
            print(f"Warning: No 'final_metrics.json' found in {scenario_path}")
            continue

        run_metrics_list = [load_run_data(mf) for mf in metrics_files]

        run_metrics_df = pd.DataFrame(run_metrics_list)
        run_metrics = run_metrics_df.mean().to_dict()  # Average over all seeds

        if run_metrics:
            all_runs.append({
                **run_scenario,
                **run_metrics
            })
        else:
            print(f"Warning: No valid metrics loaded for {scenario_name}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_adaptive_comparison(df: pd.DataFrame, defense: str, output_dir: Path):
    """
    Generates the grouped bar chart comparing adaptive strategies for a single defense.
    """
    defense_df = df[df['defense'] == defense].copy()
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Adaptive Attack Effectiveness for: {defense} ---")

    # --- Logical Sorting for the X-axis ---
    def get_sort_key(label):
        if 'black_box (data)' in label: return (0, 0)
        if 'black_box (grad)' in label: return (1, 0)
        if 'gradient_inversion' in label: return (2, 0)
        if 'oracle' in label: return (3, 0)
        return (4, 0)  # Fallback

    unique_labels = defense_df['strategy_label'].unique()
    sorted_labels = sorted(unique_labels, key=get_sort_key)
    # --- End Sorting ---

    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate', 'rounds']
    metrics_to_plot = [m for m in metrics_to_plot if m in defense_df.columns]

    plot_df = defense_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(16, 7))
    sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels  # Apply the logical sort order
    )
    plt.title(f'Effectiveness of Adaptive Attacks against {defense.upper()} Defense')
    plt.ylabel('Value')
    plt.xlabel('Adaptive Attack Strategy')
    plt.xticks(rotation=10, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()

    plot_file = output_dir / f"plot_adaptive_effectiveness_{defense}.png"
    plt.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        return

    defenses = df['defense'].unique()

    for defense in defenses:
        plot_adaptive_comparison(df, defense, output_dir)

    print("\nAnalysis complete. Check 'step7_figures' folder for plots.")


if __name__ == "__main__":
    main()