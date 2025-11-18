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
FIGURE_OUTPUT_DIR = "./step7_full_analysis_figures_MARTFL_ONLY" # Changed output dir

# Regex to parse 'step7_adaptive_black_box_gradient_manipulation_martfl_CIFAR100'
SCENARIO_PATTERN = re.compile(
    r'step7_adaptive_([a-z_]+)_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)'
)
# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses the experiment scenario name from the folder name.
    """
    try:
        match = SCENARIO_PATTERN.search(scenario_name)
        if match:
            threat_model = match.group(1)
            adaptive_mode = match.group(2)
            defense = match.group(3)
            dataset = match.group(4)

            threat_model_map = {
                'black_box': '1. Black-Box',
                'gradient_inversion': '2. Grad-Inversion',
                'oracle': '3. Oracle'
            }
            threat_label = threat_model_map.get(threat_model, threat_model)

            return {
                "threat_model": threat_model,
                "adaptive_mode": adaptive_mode,
                "defense": defense,
                "dataset": dataset,
                "threat_label": threat_label
            }
        else:
            return {"defense": "unknown"}
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"defense": "unknown"}

def collect_all_results(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walks the entire results directory, parses all runs, and loads all data.
    """
    all_seller_dfs = []
    all_global_log_dfs = []
    all_summary_rows = []

    base_path = Path(base_dir)
    print(f"Searching for 'step7_adaptive_*' directories in {base_path.resolve()}...")

    scenario_folders = list(base_path.glob("step7_adaptive_*"))
    if not scenario_folders:
        print("Error: No 'step7_adaptive_*' directories found in ./results")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(scenario_folders)} 'step7_adaptive_*' base directories.")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)
        if "defense" not in scenario_params or scenario_params["defense"] == "unknown":
            continue

        marker_files = list(scenario_path.rglob('final_metrics.json'))
        if not marker_files:
            print(f"  No completed runs (no 'final_metrics.json') found in: {scenario_path.name}")
            continue

        print(f"  Found {len(marker_files)} completed runs in: {scenario_path.name}")
        for final_metrics_file in marker_files:
            run_dir = final_metrics_file.parent
            seed_id = f"{scenario_path.name}__{run_dir.name}"

            # Load Time-Series: seller_metrics.csv
            seller_file = run_dir / 'seller_metrics.csv'
            df_seller = pd.DataFrame()
            if seller_file.exists():
                try:
                    df_seller = pd.read_csv(seller_file, on_bad_lines='skip')

                    df_seller['seed_id'] = seed_id
                    df_seller = df_seller.assign(**scenario_params)
                    df_seller['seller_type'] = df_seller['seller_id'].apply(
                        lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign'
                    )
                    all_seller_dfs.append(df_seller)
                except Exception as e:
                    if 'EmptyDataError' in str(e):
                         print(f"    Warning: {seller_file} is empty or all lines were bad.")
                    else:
                         print(f"    Error loading {seller_file}: {e}")

            # Load Time-Series: training_log.csv (for global ACC only)
            log_file = run_dir / 'training_log.csv'
            if log_file.exists():
                try:
                    use_cols = ['round', 'val_acc']
                    df_log = pd.read_csv(log_file, usecols=lambda c: c in use_cols, on_bad_lines='skip')
                    if 'val_acc' in df_log.columns:
                        df_log['seed_id'] = seed_id
                        df_log = df_log.assign(**scenario_params)
                        all_global_log_dfs.append(df_log)
                except Exception as e:
                    print(f"    Error loading {log_file}: {e}")

            # Load Summary Data (without ASR)
            try:
                with open(final_metrics_file, 'r') as f:
                    final_metrics = json.load(f)

                adv_sel_rate = 0.0
                ben_sel_rate = 0.0
                # Use the *final* selection rate from the timeseries, not the mean
                if not df_seller.empty:
                    # Get the last round's data
                    df_seller_last_round = df_seller[df_seller['round'] == df_seller['round'].max()]
                    if not df_seller_last_round.empty:
                        adv_sel_rate = df_seller_last_round[df_seller_last_round['seller_type'] == 'Adversary']['selected'].mean()
                        ben_sel_rate = df_seller_last_round[df_seller_last_round['seller_type'] == 'Benign']['selected'].mean()

                summary_row = {
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': final_metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel_rate,
                    'ben_sel_rate': ben_sel_rate
                }
                all_summary_rows.append(summary_row)
            except Exception as e:
                print(f"    Error loading {final_metrics_file}: {e}")

    if not all_summary_rows:
        print("\nError: No valid 'step7' data was successfully loaded.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_seller_timeseries = (
        pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()
    )
    df_global_timeseries = (
        pd.concat(all_global_log_dfs, ignore_index=True) if all_global_log_dfs else pd.DataFrame()
    )
    df_summary = pd.DataFrame(all_summary_rows)

    return df_seller_timeseries, df_global_timeseries, df_summary


def load_step2_5_baseline(step2_5_csv_path: Path) -> pd.DataFrame:
    """
    (UPDATED)
    Loads the Step 2.5 summary CSV to extract the 'no attack' baseline.
    Fixes the bug where the 'acc' column name was wrong.
    """
    print(f"\nLoading Step 2.5 Baseline from: {step2_5_csv_path}")
    if not step2_5_csv_path.exists():
        print(f"  Warning: {step2_5_csv_path} not found. Cannot add baseline to plots.")
        return pd.DataFrame()

    try:
        df_step2_5 = pd.read_csv(step2_5_csv_path)

        # We only care about the Benign Selection Rate (and adv, if it exists)
        metrics_to_load = {
            '5. Avg. Benign Selection Rate (%)': 'ben_sel_rate',
            '6. Avg. Adversary Selection Rate (%)': 'adv_sel_rate'
        }

        # --- THIS IS THE FIX ---
        # The column name in the CSV has '(Higher is Better)' at the end.
        acc_metric_name = '2. Avg. Usable Accuracy (%) (Higher is Better)'
        if acc_metric_name not in df_step2_5.columns:
            # Fallback just in case
            acc_metric_name = '2. Avg. Usable Accuracy (%)'
            if acc_metric_name not in df_step2_5.columns:
                 print("  Warning: Could not find 'Avg. Usable Accuracy' column in Step 2.5 CSV.")
        # --- END FIX ---


        # Pivot to long format
        df_long = df_step2_5.melt(
            id_vars=['defense', 'dataset'],
            value_vars=[col for col in metrics_to_load.keys() if col in df_step2_5.columns],
            var_name='Metric_Name',
            value_name='Value'
        )

        df_long['Value'] /= 100.0 # Convert from 83.3 -> 0.833

        # Rename columns to match the summary_df
        df_long = df_long.rename(columns={
            'Metric_Name': 'Metric',
            'Value': 'Value_mean'
        })
        df_long['Metric'] = df_long['Metric'].map(metrics_to_load)

        # --- Add columns to match the step7 data structure ---
        df_long['threat_label'] = '0. Baseline (No Attack)'
        df_long['adaptive_mode'] = 'N/A'
        df_long['seed_id'] = 'step2_5_avg'

        # We need to un-melt it to match the structure of df_summary
        df_pivot = df_long.pivot_table(
            index=['defense', 'dataset', 'threat_label', 'adaptive_mode', 'seed_id'],
            columns='Metric',
            values='Value_mean'
        ).reset_index()

        # Add 'acc' - use the '2. Avg. Usable Accuracy (%)'
        if acc_metric_name in df_step2_5.columns:
            acc_df = df_step2_5[['defense', 'dataset', acc_metric_name]].copy()
            acc_df = acc_df.rename(columns={acc_metric_name: 'acc'})
            acc_df['acc'] /= 100.0
            df_pivot = df_pivot.merge(acc_df, on=['defense', 'dataset'], how='left')

        print(f"  Successfully loaded and processed {len(df_pivot)} baseline rows.")
        return df_pivot

    except Exception as e:
        print(f"  Error processing {step2_5_csv_path}: {e}")
        return pd.DataFrame()


# ========================================================================
# PLOTTING FUNCTIONS
# ========================================================================

def plot_selection_rate_curves(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 1: The Core Plot
    (UPDATED to remove vertical "attack start" line)
    """
    print("Generating Plot 1: Selection Rate Learning Curves (one per file)...")

    if df.empty:
        print("  Skipping Plot 1: The seller time-series DataFrame is empty.")
        return

    df_plot = df.copy()
    df_plot = df_plot.sort_values(by=['seed_id', 'seller_type', 'round'])
    group_cols = ['seed_id', 'seller_type', 'defense', 'threat_label', 'adaptive_mode', 'round']

    df_agg = df_plot.groupby(group_cols)['selected'].mean().reset_index()

    group_cols_rolling = ['seed_id', 'seller_type', 'defense', 'threat_label', 'adaptive_mode']
    df_agg['rolling_sel_rate'] = df_agg.groupby(group_cols_rolling)['selected'] \
                                     .transform(lambda x: x.rolling(3, min_periods=1).mean())

    defenses = df_agg['defense'].unique()
    threat_labels = df_agg['threat_label'].unique()

    for defense in defenses:
        for threat in threat_labels:
            df_facet = df_agg[
                (df_agg['defense'] == defense) & (df_agg['threat_label'] == threat)
            ]
            if df_facet.empty:
                continue

            threat_filename = threat.replace(' ', '').replace('.', '')

            # --- PLOT 1a: The Original Plot (Selection Rate) ---
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(
                data=df_facet,
                x='round',
                y='rolling_sel_rate',
                hue='seller_type',
                style='adaptive_mode',
                palette={'Adversary': 'red', 'Benign': 'blue'},
                lw=2.5,
                errorbar=('ci', 95) # Use confidence interval
            )

            # --- VERTICAL LINE REMOVED ---

            ax.set_title(f'Selection Rate: {defense.upper()} vs {threat}')
            ax.set_xlabel('Training Round')
            ax.set_ylabel('Selection Rate (3-round Avg)')
            ax.legend(title='Seller / Mode')
            ax.set_ylim(-0.05, 1.05) # Set Y-axis 0-1

            plot_file = output_dir / f"plot1_sel_rate_{defense}_{threat_filename}.png"
            plt.savefig(plot_file, bbox_inches='tight')
            plt.clf(); plt.close('all')

            # --- PLOT 1b: The NEW "Advantage" Plot ---
            df_pivot = df_facet.pivot_table(
                index=['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode'],
                columns='seller_type',
                values='rolling_sel_rate'
            ).reset_index()

            if 'Adversary' not in df_pivot.columns or 'Benign' not in df_pivot.columns:
                print(f"  Skipping 'Advantage' plot for {defense}/{threat}: missing Adv or Benign data.")
                continue

            df_pivot['Selection_Advantage'] = df_pivot['Adversary'] - df_pivot['Benign']

            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(
                data=df_pivot,
                x='round',
                y='Selection_Advantage',
                hue='adaptive_mode', # Show one line per adaptive mode
                style='adaptive_mode',
                lw=2.5,
                errorbar=('ci', 95)
            )

            # --- VERTICAL LINE REMOVED ---
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            ax.set_title(f"Attacker's Selection Advantage: {defense.upper()} vs {threat}")
            ax.set_xlabel('Training Round')
            ax.set_ylabel('Adv. Rate - Benign Rate (Higher is better for Attacker)')
            ax.legend(title='Adaptive Mode')

            plot_file = output_dir / f"plot1_sel_ADVANTAGE_{defense}_{threat_filename}.png"
            plt.savefig(plot_file, bbox_inches='tight')
            plt.clf(); plt.close('all')


def plot_global_performance_curves(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 2: Global Accuracy
    (UPDATED to remove vertical "attack start" line)
    """
    print("Generating Plot 2: Global Accuracy Curves (one per file)...")

    if df.empty:
        print("  Skipping Plot 2: The global time-series DataFrame is empty.")
        return

    defenses = df['defense'].unique()
    threat_labels = df['threat_label'].unique()

    for defense in defenses:
        for threat in threat_labels:
            df_facet = df[
                (df['defense'] == defense) & (df['threat_label'] == threat)
            ]
            if df_facet.empty:
                continue

            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(
                data=df_facet,
                x='round',
                y='val_acc',
                hue='adaptive_mode',
                style='adaptive_mode',
                palette='Greens_d',
                lw=2.5,
                errorbar=None
            )

            # --- VERTICAL LINE REMOVED ---

            ax.set_title(f'Global Accuracy: {defense.upper()} vs {threat}')
            ax.set_xlabel('Training Round')
            ax.set_ylabel('Global Accuracy')
            ax.legend(title='Adaptive Mode')

            threat_filename = threat.replace(' ', '').replace('.', '')
            plot_file = output_dir / f"plot2_global_acc_{defense}_{threat_filename}.png"
            plt.savefig(plot_file, bbox_inches='tight')
            plt.clf()
            plt.close('all')

def plot_final_summary(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 3: The "Final" Plot
    """
    print("Generating Plot 3: Final Effectiveness Summary (one per defense)...")

    if df.empty:
        print("  Skipping Plot 3: The summary DataFrame is empty.")
        return

    df_melted = df.melt(
        id_vars=['defense', 'threat_label', 'adaptive_mode', 'seed_id'],
        value_vars=['adv_sel_rate', 'ben_sel_rate', 'acc'],
        var_name='Metric',
        value_name='Value'
    )

    # Check if 'Value' column is all NaN, which can happen if columns are missing
    if df_melted['Value'].isnull().all():
        print("  Skipping Plot 3: Melted DataFrame contains no valid data (all NaNs).")
        return

    df_melted['Value'] *= 100 # Convert to percentage

    metric_map = {
        'adv_sel_rate': 'Adversary Selection Rate (%)',
        'ben_sel_rate': 'Benign Selection Rate (%)',
        'acc': 'Final Global Accuracy (%)'
    }
    df_melted['Metric'] = df_melted['Metric'].map(metric_map)

    x_order = [
        '0. Baseline (No Attack)',
        '1. Black-Box',
        '2. Grad-Inversion',
        '3. Oracle'
    ]
    x_order = [x for x in x_order if x in df_melted['threat_label'].unique()]

    for defense in df['defense'].unique():
        df_defense = df_melted[df_melted['defense'] == defense]
        if df_defense.empty:
            continue

        g = sns.catplot(
            data=df_defense,
            kind='bar',
            x='threat_label',
            y='Value',
            col='Metric',
            hue='adaptive_mode',
            order=x_order,
            dodge=True,
            height=4,
            aspect=1.0,
            margin_titles=True,
            sharey=False,
            errorbar=('ci', 95)
        )

        g.set_axis_labels('Threat Model', 'Final Value (avg. over seeds)')
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle(f'Final Summary for {defense.upper()} Defense', y=1.03)

        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(15)
                label.set_ha('right')

        plot_file = output_dir / f"plot3_final_summary_{defense}.png"
        plt.savefig(plot_file, bbox_inches='tight')
        plt.clf()
        plt.close('all')

def plot_martfl_analysis(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 4: In-depth analysis for Martfl.
    (UPDATED to remove vertical "attack start" line)
    """
    print("Generating Plot 4: In-Depth Martfl Gradient Norm Analysis...")

    # No need to filter df_martfl, it's already pre-filtered in main()
    if df.empty:
        print("  Skipping Plot 4: No 'martfl' data found.")
        return

    # Convert 'selected' (0/1) to 'Selected'/'Not Selected' for a clearer legend
    df['Selection Status'] = df['selected'].apply(
        lambda x: 'Selected' if x == 1 else 'Not Selected'
    )

    # Use relplot to create a faceted scatter plot
    g = sns.relplot(
        data=df,
        x='round',
        y='gradient_norm',
        hue='seller_type',
        style='Selection Status',
        markers={'Selected': 'o', 'Not Selected': 'X'},
        palette={'Adversary': 'red', 'Benign': 'blue'},
        s=50,
        alpha=0.7,
        col='threat_label',
        row='adaptive_mode',
        height=4,
        aspect=1.5
    )

    # --- VERTICAL LINE REMOVED ---

    g.set_axis_labels('Training Round', 'Gradient Norm')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle('Plot 4: Martfl Analysis - Gradient Norm vs. Selection Status', y=1.03)

    plot_file = output_dir / "plot4_martfl_norm_analysis.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')

# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # --- Load Step 7 Data ---
    df_seller_ts, df_global_ts, df_summary = collect_all_results(BASE_RESULTS_DIR)

    if df_summary.empty:
        print("\nNo 'step7' data was loaded. Exiting.")
        return

    # --- (NEW) Load Step 2.5 Data ---
    step2_5_csv = Path(FIGURE_OUTPUT_DIR).parent / "step2.5_figures" / "step2.5_platform_metrics_with_selection_summary.csv"
    df_baseline = load_step2_5_baseline(step2_5_csv)

    # --- (NEW) Combine DataFrames ---
    if not df_baseline.empty:
        datasets_in_step7 = df_summary['dataset'].unique()
        df_baseline_filtered = df_baseline[df_baseline['dataset'].isin(datasets_in_step7)]

        df_summary_with_baseline = pd.concat([df_summary, df_baseline_filtered], ignore_index=True)
        print("  Successfully merged 'step7' data with 'step2.5' baseline.")
    else:
        print("  Proceeding without 'step2.5' baseline data.")
        df_summary_with_baseline = df_summary

    # --- (NEW) Pre-filter all DataFrames for 'martfl' only ---
    print("\n--- Filtering all data for 'martfl' defense ---")

    df_seller_ts_martfl = df_seller_ts[df_seller_ts['defense'] == 'martfl'].copy()
    df_global_ts_martfl = df_global_ts[df_global_ts['defense'] == 'martfl'].copy()
    df_summary_martfl = df_summary_with_baseline[df_summary_with_baseline['defense'] == 'martfl'].copy()

    if df_summary_martfl.empty:
        print("No data found for 'martfl' after filtering. Exiting.")
        return
    # --- END OF FILTERING ---

    # --- Save full time-series data (for debugging/appendix) ---
    csv_seller_ts = output_dir / "all_runs_seller_timeseries_martfl.csv"
    df_seller_ts_martfl.to_csv(csv_seller_ts, index=False)
    print(f"\n✅ Saved all martfl seller time-series data to: {csv_seller_ts}")

    csv_global_ts = output_dir / "all_runs_global_timeseries_martfl.csv"
    df_global_ts_martfl.to_csv(csv_global_ts, index=False)
    print(f"✅ Saved all martfl global time-series data to: {csv_global_ts}")

    # --- Save the combined summary data ---
    csv_summary = output_dir / "all_runs_summary_with_baseline_martfl.csv"
    df_summary_martfl.to_csv(csv_summary, index=False, float_format="%.4f")
    print(f"✅ Saved all martfl summary data (with baseline) to: {csv_summary}")

    # --- Call all plotting functions with the pre-filtered data ---
    plot_selection_rate_curves(df_seller_ts_martfl, output_dir)
    plot_global_performance_curves(df_global_ts_martfl, output_dir)
    plot_martfl_analysis(df_seller_ts_martfl, output_dir) # This plot is martfl-specific by default

    # Plot 3 uses the new combined summary data
    plot_final_summary(df_summary_martfl, output_dir)

    print("\n---")
    print("🔴 IMPORTANT NOTE ON STRATEGY PLOTS:")
    print("   The 'seller_metrics.csv' file does not contain the 'attack_strategy' column.")
    print("   Therefore, we CANNOT generate the 'Strategy Convergence' or 'Stealthy Blend' plots.")
    print("   To get this data, your experiment runner must save the attacker's per-round stats")
    print("   (from 'self.last_training_stats') into the 'marketplace_report.json'.")
    print("---")

    print(f"\n✅ Full analysis complete. Check the '{FIGURE_OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()