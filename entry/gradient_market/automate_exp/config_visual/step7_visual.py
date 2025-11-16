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
FIGURE_OUTPUT_DIR = "./step7_analysis_figures"

# (From your previous script)
def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses the base scenario name
    e.g., 'step7_adaptive_gradient_inversion_gradient_manipulation_martfl_CIFAR100'
    """
    try:
        pattern = r'step7_adaptive_([a-z_]+)_([a-z_]+)_(fedavg|martfl|fltrust|skymask|krum|flame|multikrum|bulyan)_(.*)'
        match = re.search(pattern, scenario_name)

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
                "scenario": scenario_name,
                "threat_model": threat_model,
                "adaptive_mode": adaptive_mode,
                "defense": defense,
                "dataset": dataset,
                "threat_label": threat_label
            }
        else:
            print(f"Warning: Pattern not matched for: {scenario_name}")
            return {"scenario": scenario_name, "defense": "unknown"}
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name, "defense": "unknown"}

# ========================================================================
# 1. DATA COLLECTION
# ========================================================================

def collect_all_time_series_results(base_dir: str) -> pd.DataFrame:
    """
    Parses all 'marketplace_report.json' files to build a per-round,
    per-seller time-series DataFrame. This is the main data for plots 1, 2, 3.
    """
    all_runs_data = []
    base_path = Path(base_dir)
    print(f"Searching for time-series results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step7_adaptive_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        print(f"\n--- Processing Scenario: {scenario_path.name} ---") # DEBUG
        scenario_params = parse_scenario_name(scenario_path.name)

        report_files = list(scenario_path.rglob("marketplace_report.json"))
        print(f"  Found {len(report_files)} marketplace_report.json files.") # DEBUG

        for i, report_file in enumerate(report_files):
            seed_id = f"seed_{i}"
            print(f"    Parsing file: {report_file.relative_to(base_path)}") # DEBUG
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)

                # 1. Find the adversary
                adv_id = None
                for s_id, summary in report_data.get('seller_summaries', {}).items():
                    if summary.get('type') == 'adversary':
                        adv_id = s_id
                        break

                # 2. Parse per-seller history
                history = report_data.get('seller_history', {})
                if not history: # DEBUG
                    print("    ❌ WARNING: 'seller_history' key is missing or empty in this file.")
                    continue

                print(f"    Found history for {len(history.keys())} sellers.") # DEBUG
                sellers_processed_count = 0
                for seller_id, rounds_history in history.items():
                    seller_type = 'Adversary' if seller_id == adv_id else 'Benign'

                    if not rounds_history: # DEBUG
                         print(f"    WARNING: Seller {seller_id} has an empty history list.")
                         continue

                    sellers_processed_count += 1
                    for round_stat in rounds_history:
                        all_runs_data.append({
                            **scenario_params,
                            'seed_id': seed_id,
                            'seller_id': seller_id,
                            'seller_type': seller_type,
                            'round': round_stat.get('round'),
                            'was_selected': int(round_stat.get('was_selected', 0)),
                            'attack_strategy': round_stat.get('attack_strategy'),
                            'blend_phase': round_stat.get('blend_phase'),
                            'blend_attack_intensity': round_stat.get('blend_attack_intensity')
                        })
                print(f"    Successfully processed data for {sellers_processed_count} sellers.") # DEBUG
            except Exception as e:
                print(f"    ❌ ERROR parsing {report_file}: {e}")

    if not all_runs_data:
        print("\nError: No time-series data was successfully processed (all_runs_data is empty).")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs_data)

    # Clean data types
    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    df['was_selected'] = pd.to_numeric(df['was_selected'], errors='coerce')
    df['blend_attack_intensity'] = pd.to_numeric(df['blend_attack_intensity'], errors='coerce')
    df = df.dropna(subset=['round', 'was_selected'])

    return df

def collect_all_summary_results(base_dir: str) -> pd.DataFrame:
    """
    Parses 'final_metrics.json' and 'marketplace_report.json' (summaries)
    to build a one-row-per-seed summary DataFrame. This is for Plot 4.
    """
    all_runs = []
    base_path = Path(base_dir)
    print(f"\nSearching for summary results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step7_adaptive_*") if f.is_dir()]

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        metrics_files = list(scenario_path.rglob("final_metrics.json"))

        for i, metrics_file in enumerate(metrics_files):
            seed_id = f"seed_{i}"
            try:
                # 1. Load final ASR and ACC
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                acc = metrics.get('acc', 0)
                asr = metrics.get('asr', 0)

                # 2. Load final selection rates from report summary
                report_file = metrics_file.parent / "marketplace_report.json"
                if not report_file.exists():
                    continue

                with open(report_file, 'r') as f:
                    report_data = json.load(f)

                summaries = report_data.get('seller_summaries', {}).values()
                adv_rates = [s['selection_rate'] for s in summaries if s.get('type') == 'adversary']
                ben_rates = [s['selection_rate'] for s in summaries if s.get('type') == 'benign']

                adv_sel_rate = np.mean(adv_rates) if adv_rates else 0.0
                ben_sel_rate = np.mean(ben_rates) if ben_rates else 0.0

                all_runs.append({
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': acc,
                    'asr': asr,
                    'adv_sel_rate': adv_sel_rate,
                    'ben_sel_rate': ben_sel_rate
                })
            except Exception as e:
                print(f"Error parsing summary for {metrics_file}: {e}")

    if not all_runs:
        print("Error: No summary data was successfully processed.")
        return pd.DataFrame()

    return pd.DataFrame(all_runs)

# ========================================================================
# 2. PLOTTING FUNCTIONS
# ========================================================================

def plot_selection_rate_curve(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 1: The Core Plot
    Plots Adversary vs. Benign selection rate over time.
    """
    print("Generating Plot 1: Selection Rate Learning Curves...")

    # Calculate rolling average per-seed, per-seller-type
    df_plot = df.copy()
    df_plot = df_plot.sort_values(by=['seed_id', 'seller_type', 'round'])
    df_plot['rolling_sel_rate'] = df_plot.groupby(['seed_id', 'seller_type'])['was_selected'] \
                                         .transform(lambda x: x.rolling(10, min_periods=1).mean())

    # Use FacetGrid to create a matrix of plots
    g = sns.FacetGrid(
        df_plot,
        row='defense',
        col='threat_label',
        hue='seller_type',
        palette={'Adversary': 'red', 'Benign': 'blue'},
        height=3,
        aspect=1.5,
        margin_titles=True,
        sharey=True
    )

    # Map the lineplot onto the grid
    g.map_dataframe(
        sns.lineplot,
        x='round',
        y='rolling_sel_rate',
        lw=2
    )

    g.set_axis_labels('Training Round', 'Selection Rate (10-round Avg)')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(title='Seller Type')
    g.fig.suptitle('Plot 1: Attacker vs. Benign Selection Rate Over Time', y=1.03)

    plot_file = output_dir / "plot1_selection_rate_curves.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')

def plot_strategy_convergence(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 2: The "How" Plot
    Plots which strategy the Black-Box attacker converged on.
    """
    print("Generating Plot 2: Black-Box Strategy Convergence...")

    df_plot = df[
        (df['threat_model'] == 'black_box') &
        (df['seller_type'] == 'Adversary')
    ].copy()

    if df_plot.empty:
        print("  Skipping Plot 2: No black-box adversary data found.")
        return

    # Use histplot with multiple='fill' to get a 100% stacked bar chart
    g = sns.FacetGrid(
        df_plot,
        row='defense',
        col='adaptive_mode',
        height=3,
        aspect=1.5,
        margin_titles=True,
        sharey=True
    )

    g.map_dataframe(
        sns.histplot,
        x='round',
        hue='attack_strategy',
        multiple='fill',
        binwidth=10, # Groups rounds into bins of 10
        shrink=0.8
    )

    g.set_axis_labels('Training Round', 'Strategy Usage %')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(title='Attack Strategy')
    g.fig.suptitle('Plot 2: Black-Box Strategy Convergence', y=1.03)

    plot_file = output_dir / "plot2_strategy_convergence.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')

def plot_stealthy_blend_adaptation(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 3: The "Mechanism" Plot
    Shows how 'stealthy_blend' intensity adapts to selection feedback.
    """
    print("Generating Plot 3: 'Stealthy Blend' Adaptation...")

    df_plot = df[
        (df['attack_strategy'] == 'stealthy_blend') &
        (df['seller_type'] == 'Adversary')
    ].copy()

    if df_plot.empty:
        print("  Skipping Plot 3: No 'stealthy_blend' data found.")
        return

    # Calculate rolling selection rate for this strategy
    df_plot = df_plot.sort_values(by=['seed_id', 'round'])
    df_plot['rolling_sel_rate'] = df_plot.groupby('seed_id')['was_selected'] \
                                         .transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Helper function to create the dual-axis plot
    def dual_axis_plot(data, color1='blue', color2='green', **kwargs):
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Plot 1: Blend Intensity (averaged over seeds)
        sns.lineplot(data=data, x='round', y='blend_attack_intensity', ax=ax1, color=color1, label='Intensity')
        ax1.set_ylabel('Blend Attack Intensity', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Plot 2: Selection Rate (averaged over seeds)
        sns.lineplot(data=data, x='round', y='rolling_sel_rate', ax=ax2, color=color2, label='Selection Rate (5-round Avg)')
        ax2.set_ylabel('Selection Rate', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add mimicry line (find max round where phase is mimicry)
        if 'blend_phase' in data.columns:
            mimicry_end = data[data['blend_phase'] == 'mimicry']['round'].max()
            if pd.notna(mimicry_end):
                ax1.axvline(x=mimicry_end, color='red', linestyle='--', label='End Mimicry')

    g = sns.FacetGrid(
        df_plot,
        col='defense',
        height=4,
        aspect=1.5,
        margin_titles=True
    )

    g.map_dataframe(dual_axis_plot)
    g.set_axis_labels('Training Round', '')
    g.set_titles(col_template="{col_name}")
    g.add_legend()
    g.fig.suptitle("Plot 3: 'Stealthy Blend' Intensity vs. Selection Rate", y=1.05)

    plot_file = output_dir / "plot3_stealthy_blend.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')

def plot_final_summary(df: pd.DataFrame, output_dir: Path):
    """
    PLOT 4: The "Final" Plot
    Summarizes the end-state effectiveness of all threat models.
    """
    print("Generating Plot 4: Final Effectiveness Summary...")

    # Average over all seeds
    df_agg = df.groupby(['defense', 'threat_label', 'adaptive_mode']).mean().reset_index()

    # Melt the dataframe to long format for FacetGrid
    df_melted = df_agg.melt(
        id_vars=['defense', 'threat_label', 'adaptive_mode'],
        value_vars=['adv_sel_rate', 'ben_sel_rate', 'asr', 'acc'],
        var_name='Metric',
        value_name='Value'
    )

    # Create pretty labels for the metrics
    metric_map = {
        'adv_sel_rate': 'Adversary Selection Rate',
        'ben_sel_rate': 'Benign Selection Rate',
        'asr': 'Attack Success Rate (ASR)',
        'acc': 'Global Accuracy'
    }
    df_melted['Metric'] = df_melted['Metric'].map(metric_map)

    g = sns.catplot(
        data=df_melted,
        kind='bar',
        x='threat_label',
        y='Value',
        col='Metric',  # A column for each metric
        row='defense', # A row for each defense
        hue='adaptive_mode', # Grouped bars
        dodge=True,
        height=3,
        aspect=1.2,
        margin_titles=True,
        sharey=False # Each metric has its own y-axis
    )

    g.set_axis_labels('Threat Model', 'Final Value')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle('Plot 4: Final Experiment Summary (Averaged Over Seeds)', y=1.03)

    # Rotate x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(20)

    # Add value annotations
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=8)

    plot_file = output_dir / "plot4_final_summary.png"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')

# ========================================================================
# 3. MAIN EXECUTION
# ========================================================================

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis figures will be saved to: {output_dir.resolve()}")

    # --- 1. Load Time Series Data (for Plots 1, 2, 3) ---
    df_time_series = collect_all_time_series_results(BASE_RESULTS_DIR)
    if df_time_series.empty:
        print("No time-series data found. Stopping.")
        return
    csv_ts_output = output_dir / "step7_time_series_data.csv"
    df_time_series.to_csv(csv_ts_output, index=False)
    print(f"✅ Saved time-series data to {csv_ts_output}")

    # --- 2. Load Summary Data (for Plot 4) ---
    df_summary = collect_all_summary_results(BASE_RESULTS_DIR)
    if df_summary.empty:
        print("No summary data found. Stopping.")
        return
    csv_sum_output = output_dir / "step7_summary_data.csv"
    df_summary.to_csv(csv_sum_output, index=False)
    print(f"✅ Saved summary data to {csv_sum_output}")

    # --- 3. Generate Plots ---
    plot_selection_rate_curve(df_time_series, output_dir)
    plot_strategy_convergence(df_time_series, output_dir)
    plot_stealthy_blend_adaptation(df_time_series, output_dir)
    plot_final_summary(df_summary, output_dir)

    print("\n✅ Analysis complete. Check the 'step7_analysis_figures' folder.")

if __name__ == "__main__":
    main()