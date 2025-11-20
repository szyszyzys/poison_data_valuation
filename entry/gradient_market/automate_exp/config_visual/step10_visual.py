import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
from matplotlib.ticker import MaxNLocator

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step10_figures"


# --- Styling Helper ---
def set_plot_style():
    """Sets a consistent professional style for all plots."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 9


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step10_scalability_martfl_CIFAR100')"""
    try:
        pattern = r'step10_scalability_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario_base": scenario_name,
                "defense": match.group(1),
                "dataset": match.group(2)
            }
        else:
            # Fallback or ignore
            return {}

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """ParsES 'n_sellers_10'"""
    hps = {}
    pattern = r'n_sellers_([0-9]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['n_sellers'] = int(match.group(1))
    else:
        pass
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

    scenario_folders = [f for f in base_path.glob("step10_scalability_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step10_scalability_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)
        if not run_scenario: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

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
                print(f"Error processing file {metrics_file}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_scalability_composite_row(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Generates a SINGLE wide figure (1x4) for Scalability Analysis.
    Plots:
      1. Accuracy vs N Sellers
      2. ASR vs N Sellers
      3. Benign Selection vs N Sellers
      4. Adv Selection vs N Sellers
    """
    print(f"\n--- Plotting Composite Scalability Row: {dataset} ---")

    subset = df[df['dataset'] == dataset].copy()
    if subset.empty:
        print("  -> No data found for this dataset.")
        return

    # Convert rates to percentages
    for col in ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']:
        if col in subset.columns:
            subset[col] = subset[col] * 100

    set_plot_style()

    # Initialize Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5), constrained_layout=True)

    defense_order = sorted(subset['defense'].unique())

    # Define the 4 metrics to plot in order
    metrics_map = [
        ('acc', 'Accuracy (%)'),
        ('asr', 'ASR (%)'),
        ('benign_selection_rate', 'Benign Select (%)'),
        ('adv_selection_rate', 'Adv. Select (%)')
    ]

    for i, (metric, label) in enumerate(metrics_map):
        ax = axes[i]
        if metric in subset.columns:
            sns.lineplot(
                ax=ax,
                data=subset,
                x='n_sellers',
                y=metric,
                hue='defense',
                style='defense',
                hue_order=defense_order,
                style_order=defense_order,
                markers=True,
                dashes=False,
                linewidth=2.5,
                markersize=9
            )
            ax.set_title(f"{label.replace(' (%)', '')}", fontweight='bold')
            ax.set_xlabel("Number of Sellers")
            ax.set_ylabel(label)

            # Force integer ticks on X-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Remove individual legends
            ax.get_legend().remove()
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    # --- Global Legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    # Capitalize labels nicely
    labels = [
        l.capitalize().replace("Fedavg", "FedAvg").replace("Fltrust", "FLTrust").replace("Skymask", "SkyMask").replace(
            "Martfl", "MARTFL") for l in labels]

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(defense_order), frameon=True, title="Defense Methods", fontsize=14)

    # Main Title
    fig.suptitle(f"Scalability Analysis: {dataset} (Fixed 30% Attack Rate)", fontsize=18, fontweight='bold', y=1.08)

    # Save
    filename = output_dir / f"plot_scalability_composite_{dataset}.pdf"
    plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f"  -> Saved plot to: {filename}")
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    # Plot for each dataset found
    for dataset in df['dataset'].unique():
        if pd.notna(dataset):
            plot_scalability_composite_row(df, dataset, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()