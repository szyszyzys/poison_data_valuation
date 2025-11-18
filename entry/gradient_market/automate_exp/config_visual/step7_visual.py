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
FIGURE_OUTPUT_DIR = "./figures/step7_martfl_only_figures" # (NEW) Changed output dir

# (UPDATED) One regex to rule them all
SCENARIO_PATTERN = re.compile(
    r'step7_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)'
)
# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    (UPDATED)
    Parses both 'step7_adaptive_*' and 'step7_baseline_no_attack_*' folders.
    """
    try:
        # Check for the new baseline folder first
        if scenario_name.startswith('step7_baseline_no_attack_'):
            match = re.search(r'step7_baseline_no_attack_([a-zA-Z0-9_]+)_(.*)', scenario_name)
            if match:
                return {
                    "threat_model": "baseline",
                    "adaptive_mode": "N/A",
                    "defense": match.group(1),
                    "dataset": match.group(2),
                    "threat_label": "0. Baseline (No Attack)"
                }

        # Check for the adaptive attack folders
        elif scenario_name.startswith('step7_adaptive_'):
            # Use the pattern from the top of the file
            match = re.search(r'step7_adaptive_([a-z_]+)_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)', scenario_name)
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

        # If neither matches
        print(f"Warning: Could not parse scenario name '{scenario_name}'. Ignoring folder.")
        return {"defense": "unknown"}
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"defense": "unknown"}

def collect_all_results(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (UPDATED)
    Walks the entire results directory, finds ALL 'step7_*' folders.
    """
    all_seller_dfs = []
    all_global_log_dfs = []
    all_summary_rows = []

    base_path = Path(base_dir)
    print(f"Searching for 'step7_*' directories in {base_path.resolve()}...")

    # (UPDATED) Use a wider glob to find both adaptive and baseline folders
    scenario_folders = list(base_path.glob("step7_*"))
    if not scenario_folders:
        print("Error: No 'step7_*' directories found in ./results")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(scenario_folders)} 'step7_*' base directories.")

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


# --- (DELETED) ---
# The 'load_step2_5_baseline' function is no longer needed.
# --- (DELETED) ---


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
    (Vertical line removed)
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
    (No changes needed, it's already set up for the new data)
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
    (Vertical line removed)
    """
    print("Generating Plot 4: In-Depth Martfl Gradient Norm Analysis...")

    # No need to filter df_martfl, it's already pre-filtered in main()
    if df.empty:
        print("  Skipping Plot 4: No 'martfl' data found.")
        return

    df['Selection Status'] = df['selected'].apply(
        lambda x: 'Selected' if x == 1 else 'Not Selected'
    )

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

    g.set_axis_labels('Training Round', 'Gradient Norm')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle('Plot 4: Martfl Analysis - Gradient Norm vs. Selection Status', y=1.03)

    plot_file = output_dir / "plot4_martfl_norm_analysis.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_comparative_attacker_curves(df: pd.DataFrame, df_summary_with_baseline: pd.DataFrame, output_dir: Path):
    """
    (NEW) Plots the Adversary Selection Rate for ALL threat models on ONE chart per defense.
    Adds a horizontal line for the "No Attack" Benign Baseline.
    """
    print("Generating Plot 5: Comparative Attacker Learning Curves...")

    if df.empty: return

    # 1. Get the Baseline Benign Rate (Target)
    baseline_lookup = {}
    if not df_summary_with_baseline.empty:
        baseline_rows = df_summary_with_baseline[df_summary_with_baseline['threat_label'] == '0. Baseline (No Attack)']
        # Average over datasets/seeds if multiple exist per defense
        baseline_lookup = baseline_rows.groupby('defense')['ben_sel_rate'].mean().to_dict()

    # 2. Prepare Time-Series Data
    # Filter for ADVERSARIES only
    df_adv = df[df['seller_type'] == 'Adversary'].copy()

    # Pre-calculate rolling means
    group_cols = ['seed_id', 'defense', 'threat_label', 'adaptive_mode', 'round']
    df_agg = df_adv.groupby(group_cols)['selected'].mean().reset_index()
    df_agg['rolling_sel_rate'] = df_agg.groupby(['seed_id', 'defense', 'threat_label', 'adaptive_mode'])['selected'] \
        .transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Define colors for consistency
    palette = {
        '1. Black-Box': 'red',
        '2. Grad-Inversion': 'orange',
        '3. Oracle': 'green'
    }

    for defense in df_agg['defense'].unique():
        df_defense = df_agg[df_agg['defense'] == defense]
        if df_defense.empty: continue

        plt.figure(figsize=(10, 6))

        # Plot lines for each threat model
        sns.lineplot(
            data=df_defense,
            x='round',
            y='rolling_sel_rate',
            hue='threat_label',
            style='adaptive_mode',  # Different dash styles for gradient vs data
            palette=palette,
            lw=2.5,
            errorbar=('ci', 95)
        )

        # Add Baseline Horizontal Line
        if defense in baseline_lookup:
            baseline_val = baseline_lookup[defense]
            plt.axhline(y=baseline_val, color='blue', linestyle='--', linewidth=2,
                        label='Target (No Attack Benign Rate)')

        plt.axvline(x=EXPLORATION_ROUNDS, color='grey', linestyle=':', linewidth=2, label='Attack Start')

        plt.title(f'Attacker Comparison: {defense.upper()} Selection Rate')
        plt.xlabel('Training Round')
        plt.ylabel('Adversary Selection Rate (5-round Avg)')
        plt.legend(title='Threat Model')
        plt.ylim(-0.05, 1.05)

        plot_file = output_dir / f"plot5_comparative_attacker_curves_{defense}.png"
        plt.savefig(plot_file, bbox_inches='tight')
        plt.clf()
        plt.close('all')


# --- MAIN ---
def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # Load Data
    df_seller_ts, df_global_ts, df_summary_with_baseline = collect_all_results(BASE_RESULTS_DIR)
    step2_5_csv = Path(
        FIGURE_OUTPUT_DIR).parent / "step2.5_figures" / "step2.5_platform_metrics_with_selection_summary.csv"
    df_baseline = load_step2_5_baseline(step2_5_csv)
    if not df_baseline.empty:
        datasets_in_step7 = df_summary_with_baseline['dataset'].unique()
        df_baseline_filtered = df_baseline[df_baseline['dataset'].isin(datasets_in_step7)]
        df_summary_with_baseline = pd.concat([df_summary_with_baseline, df_baseline_filtered], ignore_index=True)

    # Filter for MartFL
    print("\n--- Filtering for 'martfl' ---")
    df_seller_martfl = df_seller_ts[df_seller_ts['defense'] == 'martfl'].copy()
    df_global_martfl = df_global_ts[df_global_ts['defense'] == 'martfl'].copy()
    df_sum_martfl = df_summary_with_baseline[df_summary_with_baseline['defense'] == 'martfl'].copy()

    # Run Plots
    plot_selection_rate_curves(df_seller_martfl, output_dir)
    plot_global_performance_curves(df_global_martfl, output_dir)
    plot_final_summary(df_sum_martfl, output_dir)
    plot_martfl_analysis(df_seller_martfl, output_dir)

    # --- Run the NEW Comparative Plot ---
    plot_comparative_attacker_curves(df_seller_martfl, df_sum_martfl, output_dir)

    print(f"\n✅ Analysis complete.")


if __name__ == "__main__":
    main()