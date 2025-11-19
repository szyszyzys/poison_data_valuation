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
TARGET_VICTIM_ID = "bn_5"  # The ID used in your config generation
# --- End Configuration ---


def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        pattern = r'step8_buyer_attack_(.+)_(fedavg|martfl|fltrust|skymask)_(.*)'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "attack": match.group(1),
                "defense": match.group(2),
                "dataset": match.group(3),
            }
        return {"scenario": scenario_name}
    except Exception as e:
        print(f"Warning parsing {scenario_name}: {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> List[Dict[str, Any]]:
    run_records = []
    base_metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        # Normalize Acc to 0.0-1.0 if it looks like 0-100
        acc = metrics.get('acc', 0)
        if acc > 1.0: acc /= 100.0

        base_metrics['acc'] = acc
        base_metrics['rounds'] = metrics.get('completed_rounds', 0)
    except Exception:
        return []

    report_file = metrics_file.parent / "marketplace_report.json"
    try:
        if not report_file.exists():
            return [base_metrics]

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {})
        for seller_id, seller_data in sellers.items():
            if seller_data.get('type') == 'benign':
                record = base_metrics.copy()
                record['seller_id'] = seller_id
                record['selection_rate'] = seller_data.get('selection_rate', 0.0)
                run_records.append(record)

        return run_records if run_records else [base_metrics]
    except Exception:
        return [base_metrics]


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_seller_runs = []
    base_path = Path(base_dir)
    scenario_folders = [f for f in base_path.glob("step8_buyer_attack_*") if f.is_dir()]

    print(f"Found {len(scenario_folders)} scenario directories.")

    for scenario_path in scenario_folders:
        run_scenario_base = parse_scenario_name(scenario_path.name)
        for metrics_file in scenario_path.rglob("final_metrics.json"):
            per_seller_records = load_run_data(metrics_file)
            for r in per_seller_records:
                all_seller_runs.append({**run_scenario_base, **r})

    return pd.DataFrame(all_seller_runs)


def load_step2_5_baseline_summary(step2_5_csv_path: Path) -> Tuple[pd.DataFrame, Dict[Tuple, float]]:
    print(f"\nLoading Step 2.5 Baseline from: {step2_5_csv_path}")
    df_perf = pd.DataFrame()
    sel_lookup = {}

    if not step2_5_csv_path.exists():
        print("  Warning: Baseline CSV not found.")
        return df_perf, sel_lookup

    try:
        df = pd.read_csv(step2_5_csv)

        # Performance
        acc_col = '2. Avg. Usable Accuracy (%) (Higher is Better)'
        rounds_col = '3. Avg. Usable Rounds (Lower is Better)'
        if acc_col in df.columns:
            df_perf = df[['defense', 'dataset', acc_col, rounds_col]].copy()
            df_perf = df_perf.rename(columns={acc_col: 'acc', rounds_col: 'rounds'})
            df_perf['acc'] /= 100.0
            df_perf['attack'] = '0. Baseline'

        # Selection
        sel_col = '5. Avg. Benign Selection Rate (%)'
        if sel_col in df.columns:
            df[sel_col] /= 100.0
            sel_lookup = df.set_index(['defense', 'dataset'])[sel_col].to_dict()

    except Exception as e:
        print(f"  Error loading baseline: {e}")

    return df_perf, sel_lookup


def plot_buyer_attack_distribution(df: pd.DataFrame, baseline_sel_lookup: Dict, output_dir: Path):
    """Standard boxplots for DoS, Erosion, Starvation, Oscillation."""
    print("\n--- Plotting General Selection Rates (Fig 1) ---")
    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # Exclude pivot from this generic plot, we will handle it separately
    generic_attacks = [a for a in df['attack'].unique() if 'pivot' not in a and '0. Baseline' not in a]

    for attack in generic_attacks:
        attack_df = df[df['attack'] == attack]
        if attack_df.empty: continue

        dataset = attack_df['dataset'].iloc[0]
        plt.figure(figsize=(7, 5))

        ax = sns.boxplot(data=attack_df, x='defense', y='selection_rate', order=defense_order)

        # Draw Baseline Lines
        for i, defense in enumerate(defense_order):
            baseline_val = baseline_sel_lookup.get((defense, dataset))
            if baseline_val is not None:
                ax.hlines(y=baseline_val, xmin=i-0.4, xmax=i+0.4, color='red', linestyle='--', lw=2)

        ax.set_title(f'Selection Rate Distribution\nAttack: {attack}')
        ax.set_ylim(-0.05, 1.05)
        plt.savefig(output_dir / f"plot_SELECTION_{attack}.pdf", bbox_inches='tight')
        plt.close()


def plot_targeted_attack_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    (NEW) Visualizes Targeted Attacks (Orthogonal Pivot).
    Splits the data into 'Victim' (bn_5) vs 'Other Benign'.
    """
    print("\n--- Plotting Targeted Attack Breakdown (Fig 1.5) ---")

    pivot_df = df[df['attack'].str.contains("pivot", case=False)].copy()
    if pivot_df.empty:
        return

    # Create a category: Victim vs Others
    pivot_df['Status'] = pivot_df['seller_id'].apply(
        lambda x: 'Victim (bn_5)' if x == TARGET_VICTIM_ID else 'Other Benign'
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=pivot_df,
        x='defense',
        y='selection_rate',
        hue='Status',
        errorbar='sd', # Show standard deviation
        palette={'Victim (bn_5)': 'red', 'Other Benign': 'gray'}
    )

    plt.title("Targeted Exclusion (Orthogonal Pivot)\nSuccess = Victim Bar is near 0.0")
    plt.ylabel("Selection Rate")
    plt.ylim(0, 1.05)
    plt.savefig(output_dir / "plot_SELECTION_targeted_pivot.pdf", bbox_inches='tight')
    plt.close()


def plot_buyer_attack_performance(df: pd.DataFrame, df_perf_baseline: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Performance Metrics (Fig 2) ---")

    # Prepare DataFrame
    plot_df = df.drop_duplicates(subset=['scenario', 'acc', 'rounds', 'attack', 'defense'])

    # Filter baseline to matching datasets
    datasets = plot_df['dataset'].unique()
    baseline_subset = df_perf_baseline[df_perf_baseline['dataset'].isin(datasets)]

    plot_df = pd.concat([plot_df, baseline_subset], ignore_index=True)

    metrics = ['acc', 'rounds']
    plot_df_long = plot_df.melt(id_vars=['attack', 'defense'], value_vars=metrics, var_name='Metric', value_name='Value')

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    attacks = sorted([a for a in plot_df['attack'].unique() if a != '0. Baseline'])
    attacks.insert(0, '0. Baseline') # Ensure baseline is first

    for attack in attacks:
        attack_df = plot_df_long[plot_df_long['attack'] == attack]
        if attack_df.empty: continue

        g = sns.catplot(
            data=attack_df, x='defense', y='Value', row='Metric',
            kind='bar', order=defense_order, height=3, aspect=2, sharey=False
        )
        g.fig.suptitle(f'Global Model Performance\nAttack: {attack}', y=1.02)
        g.savefig(output_dir / f"plot_PERFORMANCE_{attack}.pdf", bbox_inches='tight')
        plt.close()


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Step 8
    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty: return

    # 2. Load Baseline
    step2_5_csv = Path(FIGURE_OUTPUT_DIR).parent / "step2.5_figures" / "step2.5_platform_metrics_with_selection_summary.csv"
    df_perf_baseline, baseline_sel_lookup = load_step2_5_baseline_summary(step2_5_csv)

    # 3. Generate Plots
    plot_buyer_attack_distribution(df, baseline_sel_lookup, output_dir)
    plot_targeted_attack_breakdown(df, output_dir) # <--- NEW FUNCTION CALL
    plot_buyer_attack_performance(df, df_perf_baseline, output_dir)

    print(f"\nAnalysis complete. Plots saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()