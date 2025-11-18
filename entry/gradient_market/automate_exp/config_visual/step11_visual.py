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

# --- Configuration (Must be updated to match your new result structure) ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step11_figures"
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]  # Must match your generator

# NEW: Mapping of the result directory structure to the bias type
BIAS_FOLDER_MAPPING = {
    "step11_market_wide": "Market-Wide Bias",
    "step11_buyer_only": "Buyer-Only Bias",
    "step11_seller_only": "Seller-Only Bias"
}


# ---------------------

# --- Helper Functions (parse_scenario_name, parse_hp_suffix, load_run_data remain the same) ---

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step11_heterogeneity_martfl_CIFAR100')"""
    try:
        # NOTE: Using the pattern that matches your original generated scenario name
        pattern = r'step11_heterogeneity_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario_base": scenario_name,
                "defense": match.group(1),
                "dataset": match.group(2)
            }
        else:
            # Fallback for unexpected naming conventions
            return {"scenario_base": scenario_name, "defense": "unknown", "dataset": "unknown"}

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
    """Loads key data from final_metrics.json and marketplace_report.json"""
    run_data = {}
    # ... (Implementation remains exactly the same as your original) ...
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


# --- Core Aggregation Function (MODIFIED) ---

def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the NEW, mapped results directories and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()} based on BIAS_FOLDER_MAPPING...")

    # Iterate over the three new, manually created root folders
    for folder_prefix, bias_label in BIAS_FOLDER_MAPPING.items():
        root_path = base_path / folder_prefix

        if not root_path.is_dir():
            print(f"Warning: Root directory '{root_path}' not found. Skipping {bias_label}.")
            continue

        # Search inside the sub-folders (which contain the original scenario names)
        scenario_folders = [f for f in root_path.glob("step11_heterogeneity_*") if f.is_dir()]
        print(f"Found {len(scenario_folders)} scenarios under {bias_label}.")

        for scenario_path in scenario_folders:
            scenario_name = scenario_path.name
            run_scenario = parse_scenario_name(scenario_name)

            for metrics_file in scenario_path.rglob("final_metrics.json"):
                try:
                    # The relative path parts will now be: [hp_folder_name, config_name]
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
                            "bias_source": bias_label  # ADDED: The bias type based on the root folder
                        })
                except Exception as e:
                    print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


# --- Plotting Function (MODIFIED for 2x4 layout) ---

def plot_heterogeneity_impact(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates the multi-panel line plots for heterogeneity impact, separated by bias source.
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

    # Rename for clearer labels
    plot_df['Metric'] = plot_df['Metric'].replace({
        'acc': '1. Model Accuracy (Utility)',
        'asr': '2. Attack Success Rate (Robustness)',
        'benign_selection_rate': '3. Benign Selection Rate (Fairness)',
        'adv_selection_rate': '4. Attacker Selection Rate (Security)',
    })
    plot_df = plot_df.sort_values(by='Metric')

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Order the bias sources for plotting rows
    bias_order = [v for k, v in BIAS_FOLDER_MAPPING.items()]

    g = sns.relplot(
        data=plot_df,
        x='dirichlet_alpha',
        y='Value',
        hue='defense',
        style='defense',
        col='Metric',
        row='bias_source',
        row_order=bias_order,
        kind='line',
        height=4,
        aspect=1.0,
        facet_kws={'sharey': False, 'margin_titles': True},
        markers=True,
        dashes=False,
        hue_order=defense_order,
        style_order=defense_order
    )

    g.fig.suptitle(f'Defense Performance vs. Bias Type/Strength ({dataset})', y=1.02)
    g.set_axis_labels("Dirichlet Alpha (Lower is More Biased)", "Value")

    # Set titles for rows (bias source) and columns (metric)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    # Set the x-axis to a log scale and format ticks
    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
        ax.xaxis.set_major_formatter(FixedFormatter([str(a) for a in ALPHAS_IN_TEST]))

        # Reverse the x-axis so "more heterogeneous" is on the right
        ax.set_xlim(max(ALPHAS_IN_TEST) * 1.5, min(ALPHAS_IN_TEST) * 0.8)

    plot_file = output_dir / f"plot_heterogeneity_by_source_{dataset}.png"
    g.fig.savefig(plot_file, bbox_inches='tight')  # Use bbox_inches to ensure titles/labels are saved
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
    csv_output_file = output_dir / "step11_heterogeneity_summary_by_source.csv"
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