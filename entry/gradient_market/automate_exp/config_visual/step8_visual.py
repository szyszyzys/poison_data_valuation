import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
    Loads key data and returns a LIST of per-seller records.
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
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})
        if not sellers:
            return [base_metrics]

        for seller_id, seller_data in sellers.items():
            if seller_data.get('type') == 'benign':
                record = base_metrics.copy()
                record['seller_id'] = seller_id
                record['selection_rate'] = seller_data.get('selection_rate', 0.0)
                run_records.append(record)

        if not run_records:
            return [base_metrics]

        return run_records

    except Exception as e:
        print(f"Error loading marketplace_report.json: {e}")
        return [base_metrics]


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks the results directory and aggregates all run data.
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


def load_step2_5_baseline_summary(step2_5_csv_path: Path) -> Tuple[pd.DataFrame, Dict[Tuple, float]]:
    """
    (NEW)
    Loads the Step 2.5 summary CSV to create a performance baseline
    and a selection rate lookup.
    """
    print(f"\nLoading Step 2.5 Baseline from: {step2_5_csv_path}")
    df_perf_baseline = pd.DataFrame()
    baseline_sel_lookup = {}

    if not step2_5_csv_path.exists():
        print(f"  Warning: {step2_5_csv_path} not found. Cannot add baseline to plots.")
        return df_perf_baseline, baseline_sel_lookup

    try:
        df_step2_5 = pd.read_csv(step2_5_csv_path)

        # 1. Create the Performance Baseline DataFrame
        acc_col = '2. Avg. Usable Accuracy (%) (Higher is Better)'
        rounds_col = '3. Avg. Usable Rounds (Lower is Better)'

        # Check if necessary columns exist
        if acc_col in df_step2_5.columns and rounds_col in df_step2_5.columns:
            df_perf_baseline = df_step2_5[['defense', 'dataset', acc_col, rounds_col]].copy()
            df_perf_baseline = df_perf_baseline.rename(columns={
                acc_col: 'acc',
                rounds_col: 'rounds'
            })
            # Convert acc from 83.3 -> 0.833
            df_perf_baseline['acc'] /= 100.0
            # Add the 'attack' column to match the step8 data
            df_perf_baseline['attack'] = '0. Baseline (No Attack)'
            print(f"  Successfully loaded {len(df_perf_baseline)} performance baseline rows.")
        else:
            print(f"  Warning: Could not find '{acc_col}' or '{rounds_col}' in baseline CSV.")

        # 2. Create the Selection Rate Lookup Dictionary
        sel_col = '5. Avg. Benign Selection Rate (%)'
        if sel_col in df_step2_5.columns:
            # Convert from 50.0 -> 0.50
            df_step2_5[sel_col] /= 100.0
            baseline_sel_lookup = df_step2_5.set_index(['defense', 'dataset'])[sel_col].to_dict()
            print(f"  Successfully created baseline selection lookup for {len(baseline_sel_lookup)} defenses.")
        else:
            print(f"  Warning: Could not find '{sel_col}' in baseline CSV.")

    except Exception as e:
        print(f"  Error processing {step2_5_csv_path}: {e}")

    return df_perf_baseline, baseline_sel_lookup


def plot_buyer_attack_distribution(df: pd.DataFrame, baseline_sel_lookup: Dict, output_dir: Path):
    """
    (UPDATED)
    Generates an individual PDF box plot for each attack type.
    Adds a horizontal line for the 'no attack' baseline.
    """
    print("\n--- Plotting Benign Seller Selection Rate Distribution (Fig 1) ---")

    if 'selection_rate' not in df.columns:
        print("Skipping: 'selection_rate' column not found.")
        return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    attack_order = [
        'dos', 'erosion', 'starvation',
        'class_exclusion_neg', 'class_exclusion_pos',
        'orthogonal_pivot_legacy',
        'oscillating_binary', 'oscillating_random', 'oscillating_drift'
    ]

    plot_df = df[df['defense'].isin(defense_order) & df['attack'].isin(attack_order)].copy()

    # --- Loop to create individual plots ---
    for attack in attack_order:
        attack_df = plot_df[plot_df['attack'] == attack]
        if attack_df.empty:
            continue

        # Get the dataset for this attack (assumes one dataset per attack tag)
        dataset = attack_df['dataset'].iloc[0]

        print(f"  Plotting Selection Rate for: {attack}")
        plt.figure(figsize=(7, 5))
        ax = sns.boxplot(
            data=attack_df,
            x='defense',
            y='selection_rate',
            order=defense_order
        )

        # --- NEW: Add baseline horizontal lines ---
        for i, defense in enumerate(defense_order):
            baseline_val = baseline_sel_lookup.get((defense, dataset))
            if baseline_val is not None:
                ax.hlines(
                    y=baseline_val,
                    xmin=i-0.4, xmax=i+0.4,
                    color='red',
                    linestyle='--',
                    lw=2,
                    label='Baseline (No Attack)' if i == 0 else None
                )
        if any(baseline_sel_lookup.get((d, dataset)) is not None for d in defense_order):
             ax.legend()
        # --- END NEW ---

        ax.set_title(f'Benign Seller Selection Rate (Distribution)\nAttack: {attack}')
        ax.set_xlabel('Seller-Side Defense')
        ax.set_ylabel('Benign Selection Rate')
        ax.set_ylim(-0.05, 1.05) # Set Y-axis 0-1

        plot_file = output_dir / f"plot_buyer_attack_SELECTION_RATES_{attack}.pdf"
        plt.savefig(plot_file, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close('all')


def plot_buyer_attack_performance(df: pd.DataFrame, df_perf_baseline: pd.DataFrame, output_dir: Path):
    """
    (UPDATED)
    Generates an individual PDF (with 2 subplots) for each attack type.
    Now includes the '0. Baseline (No Attack)' data.
    """
    print("\n--- Plotting Model Performance & Stability (Fig 2) ---")

    metrics_to_plot = ['acc', 'rounds']
    if not all(m in df.columns for m in metrics_to_plot):
        print("Skipping: 'acc' or 'rounds' columns not found.")
        return

    # De-duplicate the dataframe to get one value per run for global metrics
    plot_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds', 'attack', 'defense'])

    # --- NEW: Combine with baseline ---
    # Filter baseline to only datasets in the main df
    datasets_in_step8 = plot_df['dataset'].unique()
    df_perf_baseline = df_perf_baseline[df_perf_baseline['dataset'].isin(datasets_in_step8)]

    # Combine the step8 summary and the step2.5 baseline
    plot_df = pd.concat([plot_df, df_perf_baseline], ignore_index=True)
    # --- END NEW ---

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    # --- NEW: Add baseline to attack order ---
    attack_order = [
        '0. Baseline (No Attack)', # <-- NEW
        'dos', 'erosion', 'starvation',
        'class_exclusion_neg', 'class_exclusion_pos',
        'orthogonal_pivot_legacy',
        'oscillating_binary', 'oscillating_random', 'oscillating_drift'
    ]

    plot_df = plot_df[plot_df['defense'].isin(defense_order) & plot_df['attack'].isin(attack_order)].copy()

    plot_df_long = plot_df.melt(
        id_vars=['attack', 'defense'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    # --- Loop to create individual plots ---
    for attack in attack_order:
        attack_df = plot_df_long[plot_df_long['attack'] == attack]
        if attack_df.empty:
            continue

        print(f"  Plotting Performance for: {attack}")

        g = sns.catplot(
            data=attack_df,
            x='defense',
            y='Value',
            row='Metric',  # This creates the acc/rounds stacking
            kind='bar',
            order=defense_order,
            height=3.5,
            aspect=1.5,
            sharey=False  # acc and rounds have different scales
        )

        g.fig.suptitle(f'Model Performance & Stability\nAttack: {attack}', y=1.05)
        g.set_axis_labels('Seller-Side Defense', 'Value')
        g.set_titles(row_template="{row_name}")

        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 5),
                            textcoords='offset points',
                            fontsize=9)

        plot_file = output_dir / f"plot_buyer_attack_PERFORMANCE_{attack}.pdf"
        g.fig.savefig(plot_file, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Load Step 8 Data
    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No 'step8' data was loaded. Exiting.")
        return

    # 2. (NEW) Load Step 2.5 Baseline Data
    step2_5_csv = Path(FIGURE_OUTPUT_DIR).parent / "step2.5_figures" / "step2.5_platform_metrics_with_selection_summary.csv"
    df_perf_baseline, baseline_sel_lookup = load_step2_5_baseline_summary(step2_5_csv)

    # 3. Call the two new plotting functions
    plot_buyer_attack_distribution(df, baseline_sel_lookup, output_dir)
    plot_buyer_attack_performance(df, df_perf_baseline, output_dir)

    print("\nAnalysis complete. Check 'step8_figures' folder for plots.")


if __name__ == "__main__":
    main()