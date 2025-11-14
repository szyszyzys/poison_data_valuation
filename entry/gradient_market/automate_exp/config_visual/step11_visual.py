import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./step11_figures"
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]  # Must match your generator


# ---------------------


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step11_heterogeneity_martfl_CIFAR100')"""
    try:
        pattern = r'step11_heterogeneity_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario_base": scenario_name,
                "defense": match.group(1),
                "dataset": match.group(2)
            }
        else:
            raise ValueError(f"Pattern not matched for: {scenario_name}")

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario_base": scenario_name}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses 'alpha_1.0' or 'alpha_0.5'"""
    hps = {}
    pattern = r'alpha_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['dirichlet_alpha'] = float(match.group(1))
    else:
        print(f"Warning: Could not parse HP suffix '{hp_folder_name}'")
    return hps


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

            run_data['adv_selection_rate'] = np.mean(
                [s['selection_rate'] for s in adv_sellers]) if adv_sellers else np.nan
            run_data['benign_selection_rate'] = np.mean(
                [s['selection_rate'] for s in ben_sellers]) if ben_sellers else np.nan

            if not adv_sellers and ben_sellers:
                run_data['adv_selection_rate'] = 0.0

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step11_heterogeneity_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step11_heterogeneity_*' directories found directly inside {base_path}.")
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


def plot_heterogeneity_impact(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates the 2x2 multi-panel line plots for heterogeneity impact.
    Uses a log scale for the x-axis.
    """
    print(f"\n--- Plotting Heterogeneity Impact for {dataset} ---")

    plot_df = df[df['dataset'] == dataset].copy()
    if plot_df.empty:
        print("No data found to plot.")
        return

    metrics_to_plot = [
        'acc', 'asr',
        'benign_selection_rate', 'adv_selection_rate'
    ]
    metrics_to_plot = [m for m in metrics_to_plot if m in plot_df.columns]

    if not metrics_to_plot:
        print("No metrics found to plot.")
        return

    plot_df = plot_df.melt(
        id_vars=['dirichlet_alpha', 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    # Rename for clearer labels
    plot_df['Metric'] = plot_df['Metric'].replace({
        'acc': '1. Model Accuracy (Utility)',
        'asr': '2. Attack Success Rate (Robustness)',
        'benign_selection_rate': '3. Benign Selection Rate (Fairness)',
        'adv_selection_rate': '4. Attacker Selection Rate (Security)',
    })
    plot_df = plot_df.sort_values(by='Metric')

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    g = sns.relplot(
        data=plot_df,
        x='dirichlet_alpha',
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
        dashes=False,
        hue_order=defense_order,
        style_order=defense_order
    )

    g.fig.suptitle(f'Defense Performance vs. Market-Wide Heterogeneity ({dataset})', y=1.05)
    g.set_axis_labels("Dirichlet Alpha (Lower is More Heterogeneous)", "Value")
    g.set_titles(col_template="{col_name}")

    # --- This is the key part for readability ---
    # Set the x-axis to a log scale and format ticks
    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
        ax.xaxis.set_major_formatter(FixedFormatter([str(a) for a in ALPHAS_IN_TEST]))

    # Reverse the x-axis so "more heterogeneous" is on the right
    g.set(xlim=(max(ALPHAS_IN_TEST) * 1.5, min(ALPHAS_IN_TEST) * 0.8))
    # --- End key part ---

    plot_file = output_dir / f"plot_heterogeneity_impact_{dataset}.png"
    g.fig.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # --- Save the analysis CSV ---
    csv_output_file = output_dir / "step11_heterogeneity_summary.csv"
    df.to_csv(csv_output_file, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_file}\n")
    # -----------------------------

    # Plot for each dataset found
    for dataset in df['dataset'].unique():
        if pd.notna(dataset):
            plot_heterogeneity_impact(df, dataset, output_dir)

    print("\nAnalysis complete. Check 'step11_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()