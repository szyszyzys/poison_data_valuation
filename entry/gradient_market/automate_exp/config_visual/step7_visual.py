import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step7_martfl_only_figures"

# Regex to parse 'step7_adaptive_black_box_gradient_manipulation_martfl_CIFAR100'
SCENARIO_PATTERN = re.compile(
    r'step7_adaptive_([a-z_]+)_([a-z_]+)_([a-zA-Z0-9_]+)_(.*)'
)

EXPLORATION_ROUNDS = 30


# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses both 'step7_adaptive_*' and 'step7_baseline_no_attack_*' folders.
    """
    try:
        # Check for the new baseline folder
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

        return {"defense": "unknown"}
    except Exception as e:
        print(f"Warning: Error parsing '{scenario_name}': {e}")
        return {"defense": "unknown"}


def collect_all_results(base_dir: str, target_defense: Optional[str] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walks the results directory.
    OPTIMIZATION: If target_defense is set, skips all folders that don't match.
    """
    all_seller_dfs = []
    all_global_log_dfs = []
    all_summary_rows = []

    base_path = Path(base_dir)
    print(f"Searching for 'step7_*' directories in {base_path.resolve()}...")

    scenario_folders = list(base_path.glob("step7_*"))
    if not scenario_folders:
        print("Error: No 'step7_*' directories found in ./results")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(scenario_folders)} 'step7_*' base directories.")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        # --- OPTIMIZATION: Filter by Path Name BEFORE Loading ---
        parsed_defense = scenario_params.get("defense", "unknown")

        if parsed_defense == "unknown":
            continue

        if target_defense and parsed_defense != target_defense:
            # Skip this folder entirely without reading files
            continue
        # -------------------------------------------------------

        marker_files = list(scenario_path.rglob('final_metrics.json'))
        if not marker_files:
            continue

        for final_metrics_file in marker_files:
            run_dir = final_metrics_file.parent
            seed_id = f"{scenario_path.name}__{run_dir.name}"

            # Load Seller Metrics
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
                except Exception:
                    pass

            # Load Training Log
            log_file = run_dir / 'training_log.csv'
            if log_file.exists():
                try:
                    use_cols = ['round', 'val_acc']
                    df_log = pd.read_csv(log_file, usecols=lambda c: c in use_cols, on_bad_lines='skip')
                    if 'val_acc' in df_log.columns:
                        df_log['seed_id'] = seed_id
                        df_log = df_log.assign(**scenario_params)
                        all_global_log_dfs.append(df_log)
                except Exception:
                    pass

            # Load Summary Data
            try:
                with open(final_metrics_file, 'r') as f:
                    final_metrics = json.load(f)

                adv_sel_rate = 0.0
                ben_sel_rate = 0.0
                if not df_seller.empty:
                    last = df_seller[df_seller['round'] == df_seller['round'].max()]
                    if not last.empty:
                        adv_sel_rate = last[last['seller_type'] == 'Adversary']['selected'].mean()
                        ben_sel_rate = last[last['seller_type'] == 'Benign']['selected'].mean()

                summary_row = {
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': final_metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel_rate,
                    'ben_sel_rate': ben_sel_rate
                }
                all_summary_rows.append(summary_row)
            except Exception:
                pass

    df_seller_ts = pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()
    df_global_ts = pd.concat(all_global_log_dfs, ignore_index=True) if all_global_log_dfs else pd.DataFrame()
    df_summary = pd.DataFrame(all_summary_rows)

    return df_seller_ts, df_global_ts, df_summary


# ========================================================================
# PLOTTING FUNCTIONS (Unchanged)
# ========================================================================

def plot_selection_rate_curves(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    # Pre-calculate rolling means
    group_cols = ['seed_id', 'seller_type', 'defense', 'threat_label', 'adaptive_mode', 'round']
    df_agg = df.groupby(group_cols)['selected'].mean().reset_index()
    roll_cols = ['seed_id', 'seller_type', 'defense', 'threat_label', 'adaptive_mode']
    df_agg['rolling_sel_rate'] = df_agg.groupby(roll_cols)['selected'] \
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    for defense in df_agg['defense'].unique():
        for threat in df_agg['threat_label'].unique():
            df_facet = df_agg[(df_agg['defense'] == defense) & (df_agg['threat_label'] == threat)]
            if df_facet.empty: continue
            threat_file = threat.replace(' ', '').replace('.', '')

            # 1a. Selection Rate
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_facet, x='round', y='rolling_sel_rate', hue='seller_type', style='adaptive_mode',
                         palette={'Adversary': 'red', 'Benign': 'blue'}, lw=2.5, errorbar=('ci', 95))
            plt.title(f'Selection Rate: {defense.upper()} vs {threat}')
            plt.ylim(-0.05, 1.05)
            plt.savefig(output_dir / f"plot1_sel_rate_{defense}_{threat_file}.png", bbox_inches='tight')
            plt.close('all')

            # 1b. Advantage
            df_piv = df_facet.pivot_table(index=['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode'],
                                          columns='seller_type', values='rolling_sel_rate').reset_index()
            if 'Adversary' in df_piv.columns and 'Benign' in df_piv.columns:
                df_piv['Advantage'] = df_piv['Adversary'] - df_piv['Benign']
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df_piv, x='round', y='Advantage', hue='adaptive_mode', style='adaptive_mode', lw=2.5)
                plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
                plt.title(f"Attacker Advantage: {defense.upper()} vs {threat}")
                plt.savefig(output_dir / f"plot1_sel_ADVANTAGE_{defense}_{threat_file}.png", bbox_inches='tight')
                plt.close('all')


def plot_global_performance_curves(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    for defense in df['defense'].unique():
        for threat in df['threat_label'].unique():
            data = df[(df['defense'] == defense) & (df['threat_label'] == threat)]
            if data.empty: continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x='round', y='val_acc', hue='adaptive_mode', style='adaptive_mode',
                         palette='Greens_d', lw=2.5)
            plt.title(f'Global Accuracy: {defense.upper()} vs {threat}')
            plt.savefig(output_dir / f"plot2_global_acc_{defense}_{threat.replace(' ', '')}.png", bbox_inches='tight')
            plt.close('all')


def plot_martfl_analysis(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    df['Selection Status'] = df['selected'].apply(lambda x: 'Selected' if x == 1 else 'Not Selected')
    g = sns.relplot(data=df, x='round', y='gradient_norm', hue='seller_type', style='Selection Status',
                    palette={'Adversary': 'red', 'Benign': 'blue'}, col='threat_label', row='adaptive_mode', height=4,
                    aspect=1.5)
    g.fig.suptitle('Plot 4: Martfl Gradient Norm Analysis', y=1.03)
    plt.savefig(output_dir / "plot4_martfl_norm_analysis.png", bbox_inches='tight')
    plt.close('all')


def plot_final_summary(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    df_long = df.melt(id_vars=['defense', 'threat_label', 'adaptive_mode'],
                      value_vars=['adv_sel_rate', 'ben_sel_rate', 'acc'], var_name='Metric', value_name='Value')
    df_long['Value'] *= 100
    x_order = ['0. Baseline (No Attack)', '1. Black-Box', '2. Grad-Inversion', '3. Oracle']
    x_order = [x for x in x_order if x in df_long['threat_label'].unique()]
    for defense in df['defense'].unique():
        data = df_long[df_long['defense'] == defense]
        if data.empty: continue
        g = sns.catplot(data=data, kind='bar', x='threat_label', y='Value', col='Metric', hue='adaptive_mode',
                        order=x_order, height=4, aspect=1.0, sharey=False)
        g.set_xticklabels(rotation=15, ha='right')
        plt.savefig(output_dir / f"plot3_final_summary_{defense}.png", bbox_inches='tight')
        plt.close('all')


def plot_comparative_attacker_curves(df_ts: pd.DataFrame, df_summary: pd.DataFrame, output_dir: Path):
    if df_ts.empty: return
    baseline_val = None
    baseline_rows = df_summary[df_summary['threat_label'] == '0. Baseline (No Attack)']
    if not baseline_rows.empty:
        baseline_val = baseline_rows['ben_sel_rate'].mean()
    df_adv = df_ts[df_ts['seller_type'] == 'Adversary'].copy()
    group_cols = ['seed_id', 'defense', 'threat_label', 'adaptive_mode', 'round']
    df_agg = df_adv.groupby(group_cols)['selected'].mean().reset_index()
    df_agg['rolling_sel_rate'] = df_agg.groupby(['seed_id', 'defense', 'threat_label', 'adaptive_mode'])['selected'] \
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    palette = {'1. Black-Box': 'red', '2. Grad-Inversion': 'orange', '3. Oracle': 'green'}
    for defense in df_agg['defense'].unique():
        data = df_agg[df_agg['defense'] == defense]
        if data.empty: continue
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='round', y='rolling_sel_rate', hue='threat_label', style='adaptive_mode',
                     palette=palette, lw=2.5)
        if baseline_val is not None:
            plt.axhline(y=baseline_val, color='blue', linestyle='--', linewidth=2,
                        label='Target (No Attack Benign Rate)')
        plt.title(f'Attacker Comparison: {defense.upper()} Selection Rate')
        plt.ylim(-0.05, 1.05)
        plt.savefig(output_dir / f"plot5_comparative_attacker_curves_{defense}.png", bbox_inches='tight')
        plt.close('all')


# --- MAIN ---
def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # --- LOAD DATA (Optimized for MartFL) ---
    print("\nLoading data for 'martfl' only...")
    # Pass 'martfl' to filter during loading
    df_seller, df_global, df_summary = collect_all_results(BASE_RESULTS_DIR, target_defense='martfl')

    if df_summary.empty:
        print("No data found. Exiting.")
        return

    # --- PLOTTING ---
    print("Generating plots...")
    plot_selection_rate_curves(df_seller, output_dir)
    plot_global_performance_curves(df_global, output_dir)
    plot_martfl_analysis(df_seller, output_dir)
    plot_final_summary(df_summary, output_dir)
    plot_comparative_attacker_curves(df_seller, df_summary, output_dir)

    print(f"\n✅ Analysis complete. Check '{FIGURE_OUTPUT_DIR}'")


if __name__ == "__main__":
    main()