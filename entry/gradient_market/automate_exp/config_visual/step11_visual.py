# FILE: step11_visual.py

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
FIGURE_OUTPUT_DIR = "./figures/step11_figures"
# Alphas in descending order (Uniform -> Highly Heterogeneous)
ALPHAS_IN_TEST = [100.0, 1.0, 0.5, 0.1]
# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses: 'step11_seller_only_martfl_CIFAR100'
    """
    try:
        pattern = r'step11_(market_wide|buyer_only|seller_only)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            raw_bias = match.group(1)
            # Convert 'seller_only' -> 'Seller-Only Bias' to match plotting expectations
            bias_formatted = raw_bias.replace('_', '-').title() + " Bias"

            return {
                "scenario_base": scenario_name,
                "bias_source": bias_formatted,
                "defense": match.group(2),
                "dataset": match.group(3)
            }
        else:
            # Fallback for unknown patterns
            return {
                "scenario_base": scenario_name,
                "bias_source": "Unknown",
                "defense": "unknown",
                "dataset": "unknown"
            }

    except Exception as e:
        print(f"Error parsing scenario name '{scenario_name}': {e}")
        return {"scenario_base": scenario_name}


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses 'alpha_1.0' -> 1.0"""
    hps = {}
    pattern = r'alpha_([0-9\.]+)'
    match = re.search(pattern, hp_folder_name)
    if match:
        hps['dirichlet_alpha'] = float(match.group(1))
    return hps


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads metrics from final_metrics.json and marketplace_report.json"""
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Normalize accuracy if it's 0-100
        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        run_data['acc'] = acc
        run_data['asr'] = metrics.get('asr', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            sellers = list(report.get('seller_summaries', {}).values())
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            # Handle edge cases where no sellers of a type exist
            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0
        else:
            run_data['adv_selection_rate'] = 0.0
            run_data['benign_selection_rate'] = 0.0

        return run_data

    except Exception as e:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step11_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                # Extract alpha from parent folder name
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                if 'dirichlet_alpha' not in run_hps: continue

                metrics = load_run_data(metrics_file)
                if metrics:
                    all_runs.append({**run_scenario, **run_hps, **metrics})
            except Exception:
                continue

    return pd.DataFrame(all_runs)


def plot_heterogeneity_impact(df: pd.DataFrame, dataset: str, output_dir: Path):
    print(f"\n--- Plotting Heterogeneity Impact for {dataset} ---")

    plot_df = df[df['dataset'] == dataset].copy()
    if plot_df.empty: return

    metrics_map = {
        'acc': '1. Model Accuracy (Utility)',
        'asr': '2. Attack Success Rate (Robustness)',
        'benign_selection_rate': '3. Benign Selection (Fairness)',
        'adv_selection_rate': '4. Attacker Selection (Security)',
    }

    # Filter for available metrics
    avail_metrics = [m for m in metrics_map.keys() if m in plot_df.columns]

    plot_df = plot_df.melt(
        id_vars=['dirichlet_alpha', 'defense', 'bias_source'],
        value_vars=avail_metrics,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Metric'] = plot_df['Metric'].map(metrics_map)
    plot_df = plot_df.sort_values(by='Metric')

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    # Bias Order (Matches the parser output)
    bias_order = ['Market-Wide Bias', 'Buyer-Only Bias', 'Seller-Only Bias']

    # Filter bias_order to only what exists in data (prevents empty rows)
    existing_biases = plot_df['bias_source'].unique()
    bias_order = [b for b in bias_order if b in existing_biases]

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
        height=3.5,
        aspect=1.2,
        facet_kws={'sharey': False, 'margin_titles': True},
        markers=True,
        dashes=False,
        hue_order=defense_order,
        style_order=defense_order,
        errorbar=('ci', 95) # Show confidence intervals
    )

    g.fig.suptitle(f'Impact of Heterogeneity Source & Strength ({dataset})', y=1.02)

    # Formatting Axes
    for ax in g.axes.flat:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(ALPHAS_IN_TEST))
        ax.xaxis.set_major_formatter(FixedFormatter([str(a) for a in ALPHAS_IN_TEST]))

        # Reverse Axis: 100 (Uniform) -> 0.1 (Heterogeneous)
        ax.set_xlim(max(ALPHAS_IN_TEST) * 1.5, min(ALPHAS_IN_TEST) * 0.8)
        ax.grid(True, which='major', linestyle='--', alpha=0.5)

    # Save as PDF (Best for papers) and PNG
    g.fig.savefig(output_dir / f"plot_heterogeneity_source_{dataset}.pdf", bbox_inches='tight')
    print(f"Saved plot: plot_heterogeneity_source_{dataset}.pdf")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No data found.")
        return

    # Save Summary
    df.to_csv(output_dir / "step11_summary.csv", index=False)

    for dataset in df['dataset'].unique():
        plot_heterogeneity_impact(df, dataset, output_dir)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()