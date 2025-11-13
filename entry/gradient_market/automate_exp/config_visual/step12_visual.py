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
FIGURE_OUTPUT_DIR = "./step12_figures"


# --- End Configuration ---


def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses the base scenario name
    e.g., 'step12_main_summary_martfl_image_CIFAR10_cnn'
    """
    try:
        pattern = r'step12_main_summary_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(.+)_([a-z0-9_]+)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3),
                "model": match.group(4),
            }
        else:
            raise ValueError(f"Pattern not matched for: {scenario_name}")

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_final_metrics(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads only the final, global metrics from final_metrics.json
    and averages from marketplace_report.json
    """
    try:
        metrics = {}
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Load marketplace averages for summary
        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']
            metrics['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
            metrics['benign_selection_rate'] = np.mean(
                [s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0

        return metrics

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results_for_heatmap(base_dir: str) -> pd.DataFrame:
    """Collects just the final_metrics.json for the summary heatmap."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for 'final_metrics.json' for heatmap in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step12_main_summary_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step12_main_summary_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)

        # Find the average metrics over all seeds (e.g., inside run_1_seed_X)
        metrics_files = list(scenario_path.rglob("final_metrics.json"))
        if not metrics_files:
            continue

        run_metrics_list = [load_final_metrics(mf) for mf in metrics_files]
        run_metrics_df = pd.DataFrame(run_metrics_list)
        run_metrics_agg = run_metrics_df.mean().to_dict()  # Average over all seeds

        if run_metrics_agg:
            all_runs.append({
                **run_scenario,
                **run_metrics_agg
            })

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    return pd.DataFrame(all_runs)


def collect_all_seller_data_for_valuation(base_dir: str) -> pd.DataFrame:
    """
    Collects the detailed 'seller_metrics.csv' from all runs
    to analyze valuation metrics.
    """
    all_seller_rows = []
    base_path = Path(base_dir)
    print(f"\nSearching for 'seller_metrics.csv' for valuation in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step12_main_summary_*") if f.is_dir()]
    if not scenario_folders:
        print("No 'step12_main_summary_*' directories found.")
        return pd.DataFrame()

    for scenario_path in scenario_folders:
        run_scenario_base = parse_scenario_name(scenario_path.name)

        seller_metric_files = list(scenario_path.rglob("seller_metrics.csv"))
        if not seller_metric_files:
            continue

        for i, metrics_csv in enumerate(seller_metric_files):
            try:
                # --- THIS IS THE FIX ---
                # Skip bad lines instead of crashing
                seller_df = pd.read_csv(metrics_csv, on_bad_lines='skip')
                # ---------------------

                # Add scenario info
                seller_df['defense'] = run_scenario_base.get('defense')
                seller_df['dataset'] = run_scenario_base.get('dataset')
                seller_df['seed'] = i
                seller_df['seller_type'] = seller_df['seller_id'].apply(
                    lambda x: 'adversary' if x.startswith('adv') else 'benign'
                )
                all_seller_rows.append(seller_df)
            except Exception as e:
                # This will catch other errors, but the 'on_bad_lines'
                # will handle the tokenizing error.
                print(f"Error reading {metrics_csv}: {e}")

    if not all_seller_rows:
        print("Error: No 'seller_metrics.csv' files were successfully processed.")
        return pd.DataFrame()

    return pd.concat(all_seller_rows, ignore_index=True)


def plot_main_summary_heatmaps(df: pd.DataFrame, output_dir: Path):
    """
    Generates the main summary heatmaps (Figure 1).
    """
    print("\n--- Plotting Main Summary Heatmaps (Fig 1) ---")

    metrics_to_plot = {
        'acc': 'Greens',
        'asr': 'Reds',
        'adv_selection_rate': 'Reds',
        'benign_selection_rate': 'Greens'
    }

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    dataset_order = sorted(df['dataset'].unique())

    for metric, colormap in metrics_to_plot.items():
        if metric not in df.columns:
            print(f"Skipping heatmap for '{metric}': Column not found.")
            continue

        pivot_df = pd.pivot_table(
            df, values=metric, index='dataset', columns='defense'
        )
        pivot_df = pivot_df.reindex(index=dataset_order, columns=defense_order).dropna(how='all', axis=0).dropna(
            how='all', axis=1)

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            pivot_df,
            annot=True, fmt=".3f", cmap=colormap,
            linewidths=.5, cbar_kws={'label': metric}
        )
        plt.title(f'Main Summary: {metric.replace("_", " ").title()}')
        plt.xlabel('Defense')
        plt.ylabel('Dataset')
        plt.tight_layout()

        plot_file = output_dir / f"plot_main_summary_heatmap_{metric}.png"
        plt.savefig(plot_file)
        print(f"Saved plot: {plot_file}")
        plt.clf()


def plot_valuation_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Generates the valuation analysis box plots (Figure 2).
    """
    print("\n--- Plotting Valuation Analysis Plots (Fig 2) ---")

    # Dynamically find all valuation columns from the generator script
    val_cols = [
        'sim_to_oracle', 'sim_to_buyer',
        'influence_score', 'loo_score', 'kernelshap_score'
    ]
    # Filter to only columns that actually exist in the loaded data
    val_cols = [col for col in val_cols if col in df.columns]

    if not val_cols:
        print("No valuation columns found in seller_metrics.csv. Skipping plots.")
        return

    # Create plots for each defense/dataset combo
    for (defense, dataset), group_df in df.groupby(['defense', 'dataset']):
        print(f"  - Plotting valuation for {defense} on {dataset}...")

        plot_df = group_df.melt(
            id_vars=['seller_type', 'selected', 'round'],
            value_vars=val_cols,
            var_name='Valuation Method',
            value_name='Score'
        )
        plot_df = plot_df.dropna(subset=['Score'])

        if plot_df.empty:
            print(f"    ...No valuation data found for {defense}/{dataset}. Skipping.")
            continue

        fig, axes = plt.subplots(
            nrows=len(val_cols), ncols=2,
            figsize=(12, 4 * len(val_cols)),
            squeeze=False
        )
        fig.suptitle(f'Valuation Analysis for {defense.upper()} on {dataset.upper()}', y=1.03)

        for i, val_method in enumerate(val_cols):
            method_df = plot_df[plot_df['Valuation Method'] == val_method]
            if method_df.empty:
                axes[i, 0].set_title(f"{val_method}: No Data")
                axes[i, 1].set_title(f"{val_method}: No Data")
                continue

            # Plot 1: Valuation vs. Seller Type
            sns.boxplot(
                data=method_df, x='seller_type', y='Score',
                ax=axes[i, 0], palette={'benign': 'g', 'adversary': 'r'},
                order=['benign', 'adversary']
            )
            axes[i, 0].set_title(f'{val_method} vs. Seller Type')
            axes[i, 0].set_xlabel('Seller Type')

            # Plot 2: Valuation vs. Selection
            sns.boxplot(
                data=method_df, x='selected', y='Score',
                ax=axes[i, 1], order=[True, False]
            )
            axes[i, 1].set_title(f'{val_method} vs. Selection Decision')
            axes[i, 1].set_xlabel('Was Selected?')

        plt.tight_layout()
        plot_file = output_dir / f"plot_valuation_analysis_{defense}_{dataset}.png"
        plt.savefig(plot_file)
        print(f"Saved plot: {plot_file}")
        plt.clf()
        plt.close(fig)  # Close the figure


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # --- Figure 1 ---
    df_heatmap = collect_all_results_for_heatmap(BASE_RESULTS_DIR)
    if not df_heatmap.empty:
        plot_main_summary_heatmaps(df_heatmap, output_dir)
    else:
        print("Skipping summary heatmaps, no final_metrics data found.")

    # --- Figure 2 ---
    df_seller = collect_all_seller_data_for_valuation(BASE_RESULTS_DIR)
    if not df_seller.empty:
        plot_valuation_analysis(df_seller, output_dir)
    else:
        print("Skipping valuation analysis, no seller_metrics data found.")

    print("\nAnalysis complete. Check 'step12_figures' folder for plots.")


if __name__ == "__main__":
    main()
