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


# ---------------------


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

            # --- THIS IS THE FIX ---
            # This dictionary was missing, causing the NameError
            threat_model_map = {
                'black_box': '1. Black-Box',
                'gradient_inversion': '2. Grad-Inversion',
                'oracle': '3. Oracle'
            }
            # --- END FIX ---

            # Use .get() for a safe fallback
            threat_label = threat_model_map.get(threat_model, threat_model)

            return {
                "scenario": scenario_name,
                "threat_model": threat_model,
                "adaptive_mode": adaptive_mode,
                "defense": defense,
                "dataset": dataset,
                "strategy_label": f"{threat_label} ({mode_label})"
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

        # If parsing failed, run_scenario won't have 'defense' key
        # This check prevents the KeyError later
        if 'defense' not in run_scenario:
            print(f"  Skipping {scenario_name}, parsing failed.")
            continue

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
    (IMPROVED PLOT)
    Generates a faceted bar chart comparing adaptive strategies for a single defense.
    This version is much cleaner and easier to read.
    """
    defense_df = df[df['defense'] == defense].copy()
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Adaptive Attack Effectiveness for: {defense} ---")

    # Metrics to show as separate plots
    metrics_to_plot = {
        'acc': 'Accuracy (Higher is Better)',
        'asr': 'ASR (Lower is Better)',
        'adv_selection_rate': 'Attacker Selection Rate (Higher is Better)',
        'benign_selection_rate': 'Benign Selection Rate (Higher is Better)'
    }

    # Filter to only metrics we have data for
    metrics_present = [m for m in metrics_to_plot.keys() if m in defense_df.columns]

    plot_df = defense_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_present,
        var_name='Metric',
        value_name='Value'
    )

    # Apply the pretty names
    plot_df['Metric'] = plot_df['Metric'].map(metrics_to_plot)
    # Get the order for the columns from the map
    metric_order = [metrics_to_plot[m] for m in metrics_present]

    # Use the 'strategy_label' which is already sorted 1, 2, 3
    sorted_labels = sorted(plot_df['strategy_label'].unique())

    g = sns.catplot(
        data=plot_df,
        kind='bar',
        x='strategy_label',
        y='Value',
        col='Metric',  # <-- Create a column for each metric
        col_wrap=2,  # <-- Arrange in a 2x2 grid
        order=sorted_labels,
        col_order=metric_order,  # <-- Keep the metrics in a consistent 1,2,3,4 order
        height=4,
        aspect=1.2,
        sharex=True,  # All plots share the same x-axis
        sharey=False  # Each metric has its own y-axis
    )

    g.fig.suptitle(f'Effectiveness of Adaptive Attacks against {defense.upper()} Defense', y=1.03)
    g.set_axis_labels('Adaptive Attack Strategy', 'Value')
    g.set_titles(col_template="{col_name}")

    # Add annotations
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle

    plot_file = output_dir / f"plot_adaptive_effectiveness_{defense}.png"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results found. Exiting.")
        return

    # --- THIS IS THE NEW LINE ---
    csv_output_file = output_dir / "step7_adaptive_results_summary.csv"
    df.to_csv(csv_output_file, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_file}\n")
    # ----------------------------

    # This check prevents the KeyError if the 'defense' column is missing
    if 'defense' not in df.columns:
        print("Error: 'defense' column not found in data. Check parsing.")
        return

    defenses = df['defense'].unique()

    for defense in defenses:
        if pd.notna(defense):
            plot_adaptive_comparison(df, defense, output_dir)

    print("\nAnalysis complete. Check 'step7_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()