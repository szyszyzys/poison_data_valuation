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
FIGURE_OUTPUT_DIR = "./figures/step8_figures"


# --- End Configuration ---


def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses the base scenario name
    e.g., 'step8_buyer_attack_dos_fedavg_CIFAR100'
    """
    try:
        pattern = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)

        if match:
            attack_tag = match.group(1)
            defense = match.group(2)
            dataset = match.group(3)

            return {
                "scenario": scenario_name,
                "attack": attack_tag,
                "defense": defense,
                "dataset": dataset,
            }
        else:
            raise ValueError(f"Pattern not matched for: {scenario_name}")

    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    """
    (REWRITTEN)
    Loads key data and returns a LIST of per-seller records.
    Each record shares the same 'acc' and 'rounds' but has a unique
    'seller_id' and 'selection_rate'.
    """
    run_records = []

    # 1. Load global metrics
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        base_metrics['acc'] = metrics.get('acc', 0)
        base_metrics['rounds'] = metrics.get('completed_rounds', 0)
    except Exception as e:
        print(f"Error loading final_metrics.json: {e}")
        return []  # Skip this run

    # 2. Load per-seller data
    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists():
            # No seller data, just return the global metrics
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})
        if not sellers:
            return [base_metrics]

        for seller_id, seller_data in sellers.items():
            # We only care about the distribution of BENIGN sellers
            if seller_data.get('type') == 'benign':
                record = base_metrics.copy()
                record['seller_id'] = seller_id
                record['selection_rate'] = seller_data.get('selection_rate', 0.0)
                run_records.append(record)

        if not run_records:  # e.g., if there were 0 benign sellers
            return [base_metrics]

        return run_records

    except Exception as e:
        print(f"Error loading marketplace_report.json: {e}")
        return [base_metrics]  # Return global data even if report fails


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    (REWRITTEN)
    Walks the results directory and aggregates all run data.
    Each experiment run will now produce MULTIPLE rows, one for each
    benign seller in that run.
    """
    all_seller_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step8_buyer_attack_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step8_buyer_attack_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario_base = parse_scenario_name(scenario_name)

        metrics_files = list(scenario_path.rglob("final_metrics.json"))

        if not metrics_files:
            print(f"Warning: No 'final_metrics.json' found in {scenario_path}")
            continue

        for metrics_file in metrics_files:  # Loop over seeds
            # load_run_data now returns a list of seller records
            per_seller_records = load_run_data(metrics_file)

            for seller_record in per_seller_records:
                all_seller_runs.append({
                    **run_scenario_base,
                    **seller_record
                })

    if not all_seller_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_seller_runs)
    return df


def plot_buyer_attack_distribution(df: pd.DataFrame, output_dir: Path):
    """
    (NEW) Generates a box plot grid showing the DISTRIBUTION of
    benign seller selection rates. This highlights outliers.
    """
    print("\n--- Plotting Benign Seller Selection Rate Distribution (Fig 1) ---")

    if 'selection_rate' not in df.columns:
        print("Skipping: 'selection_rate' column not found.")
        return

    # Define the logical order for columns and rows
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    attack_order = [
        'dos', 'erosion', 'starvation',
        'class_exclusion_neg', 'class_exclusion_pos',
        'orthogonal_pivot_legacy',
        'oscillating_binary', 'oscillating_random', 'oscillating_drift'
    ]

    # Filter to only known attacks/defenses
    plot_df = df[df['defense'].isin(defense_order) & df['attack'].isin(attack_order)].copy()

    g = sns.catplot(
        data=plot_df,
        x='defense',
        y='selection_rate',
        col='attack',
        kind='box',  # Use a box plot to show distribution and outliers
        order=defense_order,
        col_order=attack_order,
        col_wrap=3,  # 3 attacks per row
        height=4,
        aspect=1.0,
        sharey=True
    )

    g.fig.suptitle('Buyer Attack Impact on Benign Seller Selection Rate (Distribution)', y=1.03)
    g.set_axis_labels('Seller-Side Defense', 'Benign Selection Rate')
    g.set_titles(col_template="{col_name}")

    plot_file = output_dir / "plot_buyer_attack_SELECTION_RATES.png"
    g.fig.savefig(plot_file)
    print(f"Saved plot: {plot_file}")
    plt.clf()


def plot_buyer_attack_performance(df: pd.DataFrame, output_dir: Path):
    """
    (NEW) Generates a bar plot grid showing the impact on
    global metrics like 'acc' and 'rounds'.
    """
    print("\n--- Plotting Model Performance & Stability (Fig 2) ---")

    metrics_to_plot = ['acc', 'rounds']
    if not all(m in df.columns for m in metrics_to_plot):
        print("Skipping: 'acc' or 'rounds' columns not found.")
        return

    # De-duplicate the dataframe to get one value per run for global metrics
    plot_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds'])

    # Define the logical order
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    attack_order = [
        'dos', 'erosion', 'starvation',
        'class_exclusion_neg', 'class_exclusion_pos',
        'orthogonal_pivot_legacy',
        'oscillating_binary', 'oscillating_random', 'oscillating_drift'
    ]

    plot_df = plot_df[plot_df['defense'].isin(defense_order) & plot_df['attack'].isin(attack_order)].copy()

    # Melt to long format
    plot_df_long = plot_df.melt(
        id_vars=['attack', 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    g = sns.catplot(
        data=plot_df_long,
        x='defense',
        y='Value',
        col='attack',
        row='Metric',  # New rows for acc vs. rounds
        kind='bar',
        order=defense_order,
        col_order=attack_order,
        height=3.5,
        aspect=1.2,
        sharey=False  # acc and rounds have different scales
    )

    g.fig.suptitle('Buyer Attack Impact on Model Performance & Stability', y=1.03)
    g.set_axis_labels('Seller-Side Defense', 'Value')
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    plot_file = output_dir / "plot_buyer_attack_PERFORMANCE.png"
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

    # Call the two new plotting functions
    plot_buyer_attack_distribution(df, output_dir)
    plot_buyer_attack_performance(df, output_dir)

    print("\nAnalysis complete. Check 'step8_figures' folder for plots.")


if __name__ == "__main__":
    main()