import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step7_manipulation_visuals"
EXPLORATION_ROUNDS = 30

# --- Global Style ---
def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'figure.dpi': 300
    })

COLOR_MAP = {
    'Adversary': '#D62728',   # Red
    'Benign': '#1F77B4',      # Blue
    '0. Baseline (No Attack)': 'gray',
    '1. Black-Box': '#FF7F0E',
    '2. Grad-Inversion': '#2CA02C',
    '3. Oracle': '#9467BD'
}

# ---------------------
# PARSING (Unchanged)
# ---------------------
def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    # ... (Keep your existing robust parser here) ...
    try:
        if scenario_name.startswith('step7_baseline_no_attack_'):
            parts = scenario_name.replace('step7_baseline_no_attack_', '').split('_')
            return {
                "threat_model": "baseline", "adaptive_mode": "N/A",
                "defense": parts[0], "dataset": parts[1] if len(parts)>1 else "CIFAR100",
                "threat_label": "0. Baseline (No Attack)"
            }
        elif scenario_name.startswith('step7_adaptive_'):
            rest = scenario_name.replace('step7_adaptive_', '')
            threat_model = 'black_box' if 'black_box' in rest else \
                           'gradient_inversion' if 'gradient_inversion' in rest else \
                           'oracle' if 'oracle' in rest else 'unknown'

            clean_rest = rest.replace(f"{threat_model}_", "")
            adaptive_mode = 'data_poisoning' if 'data_poisoning' in clean_rest else \
                            'gradient_manipulation' if 'gradient_manipulation' in clean_rest else 'unknown'

            clean_rest = clean_rest.replace(f"{adaptive_mode}_", "")
            parts = clean_rest.split('_')

            threat_map = {'black_box': '1. Black-Box', 'gradient_inversion': '2. Grad-Inversion', 'oracle': '3. Oracle'}

            return {
                "threat_model": threat_model,
                "adaptive_mode": adaptive_mode,
                "defense": parts[0],
                "dataset": parts[1] if len(parts)>1 else "CIFAR100",
                "threat_label": threat_map.get(threat_model, threat_model)
            }
        return {"defense": "unknown"}
    except: return {"defense": "unknown"}

# ---------------------
# DATA LOADING (Enhanced)
# ---------------------
def collect_all_results(base_dir: str, target_defense: Optional[str] = None):
    all_seller_dfs, all_summary_rows = [], []
    base_path = Path(base_dir)

    for scenario_path in list(base_path.glob("step7_*")):
        scenario_params = parse_scenario_name(scenario_path.name)
        if scenario_params["defense"] == "unknown": continue
        if target_defense and scenario_params["defense"] != target_defense: continue

        for final_metrics_file in scenario_path.rglob('final_metrics.json'):
            try:
                run_dir = final_metrics_file.parent
                seed_id = f"{scenario_path.name}/{run_dir.name}"

                # LOAD SELLER METRICS (Now looks for 'strategy' column)
                seller_file = run_dir / 'seller_metrics.csv'
                df_seller_run = pd.DataFrame()
                if seller_file.exists():
                    df_seller_run = pd.read_csv(seller_file, on_bad_lines='skip')
                    df_seller_run['seed_id'] = seed_id
                    df_seller_run = df_seller_run.assign(**scenario_params)
                    df_seller_run['seller_type'] = df_seller_run['seller_id'].apply(
                        lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign')

                    # Ensure strategy column exists (fill N/A for benign)
                    if 'strategy' not in df_seller_run.columns:
                        df_seller_run['strategy'] = 'honest'

                    all_seller_dfs.append(df_seller_run)

                # LOAD SUMMARY
                with open(final_metrics_file, 'r') as f: metrics = json.load(f)

                # Calculate simple selection rates
                if not df_seller_run.empty:
                    df_run = df_seller_run[df_seller_run['round'] > EXPLORATION_ROUNDS]
                    adv_sel = df_run[df_run['seller_type'] == 'Adversary']['selected'].mean()
                    ben_sel = df_run[df_run['seller_type'] == 'Benign']['selected'].mean()
                else:
                    adv_sel, ben_sel = 0, 0

                all_summary_rows.append({
                    **scenario_params, 'seed_id': seed_id,
                    'acc': metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel, 'ben_sel_rate': ben_sel
                })
            except Exception: pass

    return (pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()), \
           pd.DataFrame(all_summary_rows)

# ---------------------
# PLOTTING FUNCTIONS
# ---------------------

def plot_selection_dynamics(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Rolling selection rates (Unchanged - Keeps it strictly dynamics)."""
    if df.empty: return
    print("Generating Plot 1: Dynamics...")

    # Calculate rolling averages
    df = df.sort_values('round')
    df['rolling_sel'] = df.groupby(['seed_id', 'seller_id'])['selected'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    # Aggregate by type
    df_agg = df.groupby(['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode', 'seller_type'])['rolling_sel'].mean().reset_index()

    for defense in df_agg['defense'].unique():
        data = df_agg[df_agg['defense'] == defense]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='round', y='rolling_sel', hue='seller_type',
                     style='adaptive_mode', palette=COLOR_MAP, lw=2.5)

        # plt.title(f'Marketplace Manipulation Dynamics: {defense.upper()}')
        plt.ylabel("Selection Rate (Smoothed)")
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.savefig(output_dir / f"1_dynamics_{defense}.pdf", bbox_inches='tight')
        plt.close()

def plot_economic_exploitation(df_sum: pd.DataFrame, output_dir: Path):
    """
    Plot 2 (REVISED): Economic Exploitation Analysis.
    Instead of 'Damage', we look for 'Free Riding' (High Selection + Good Accuracy).
    """
    if df_sum.empty: return
    print("Generating Plot 2: Economic Exploitation...")

    attacks = df_sum[df_sum['threat_label'] != '0. Baseline (No Attack)'].copy()

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1. Define "Free Rider" Zone (High Selection, High Accuracy)
    # This is where the attacker mimics benign behavior perfectly or cheats successfully
    rect = patches.Rectangle((0.5, 0.7), 0.5, 0.3, linewidth=0, edgecolor='none', facecolor='orange', alpha=0.15)
    ax.add_patch(rect)
    plt.text(0.95, 0.95, "ECONOMIC EXPLOITATION\n(Cheating / Free Riding)", color='#D35400',
             ha='right', va='top', fontsize=12, fontweight='bold')

    # 2. Scatter
    sns.scatterplot(data=attacks, x='adv_sel_rate', y='acc', hue='threat_label',
                    style='adaptive_mode', palette=COLOR_MAP, s=150, alpha=0.85, edgecolor='black')

    # 3. Baseline Reference
    baseline = df_sum[df_sum['threat_label'] == '0. Baseline (No Attack)']
    if not baseline.empty:
        base_acc = baseline['acc'].mean()
        base_sel = baseline['ben_sel_rate'].mean()
        plt.axhline(base_acc, color='gray', linestyle='--', label='Baseline Acc')
        plt.axvline(base_sel, color='gray', linestyle=':', label='Baseline Sel Rate')
        plt.plot(base_sel, base_acc, marker='*', color='gold', markersize=25, markeredgecolor='black', label='Honest Behavior')

    # plt.title("Manipulation Impact: Stealth vs. Model Utility")
    plt.xlabel("Adversary Selection Rate (Stealth)")
    plt.ylabel("Global Model Accuracy")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.savefig(output_dir / "2_exploitation_scatter.pdf", bbox_inches='tight')
    plt.close()

def plot_strategy_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3 (NEW): Strategy Convergence Heatmap.
    Shows which strategies the adaptive attacker actually chose over time.
    """
    # Filter for adversaries in Black-Box mode (where strategy selection happens)
    df_adv = df[(df['seller_type'] == 'Adversary') & (df['threat_model'] == 'black_box')].copy()

    if df_adv.empty:
        print("Skipping Plot 3: No black-box strategy data found.")
        return
    print("Generating Plot 3: Strategy Evolution...")

    # Bin rounds to make heatmap readable
    df_adv['Round_Bin'] = pd.cut(df_adv['round'], bins=10, labels=False)

    for defense in df_adv['defense'].unique():
        data = df_adv[df_adv['defense'] == defense]
        if data.empty: continue

        # Count strategy usage per bin
        # Normalize to get probability distribution
        heatmap_data = data.groupby(['Round_Bin', 'strategy']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) # Normalize rows

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data.T, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={'label': 'Usage Probability'})

        # plt.title(f"Adaptive Strategy Convergence: {defense.upper()}")
        plt.xlabel("Training Phase (Round Bins)")
        plt.ylabel("Strategy Selected")

        plt.savefig(output_dir / f"3_strategy_heatmap_{defense}.pdf", bbox_inches='tight')
        plt.close()

def plot_manipulation_fairness(df_sum: pd.DataFrame, output_dir: Path):
    """
    Plot 4 (NEW): Fairness Gap.
    How much more is the adversary selected compared to the average benign seller?
    """
    if df_sum.empty: return
    print("Generating Plot 4: Fairness Gap...")

    df_sum['Advantage'] = df_sum['adv_sel_rate'] - df_sum['ben_sel_rate']

    plt.figure(figsize=(10, 6))

    # Sort for cleaner visual
    order = df_sum.groupby('threat_label')['Advantage'].median().sort_values().index

    sns.boxplot(data=df_sum, x='threat_label', y='Advantage', hue='adaptive_mode',
                order=order, palette="coolwarm")

    plt.axhline(0, color='black', linestyle='-')
    plt.text(0.5, 0.05, "Adversary Wins (Unfair)", transform=plt.gca().transAxes, ha='center', color='red', alpha=0.5)
    plt.text(0.5, -0.05, "Defense Wins (Fair)", transform=plt.gca().transAxes, ha='center', color='blue', alpha=0.5)

    # plt.title("Marketplace Fairness Analysis")
    plt.ylabel("Selection Advantage (Adv Rate - Benign Rate)")
    plt.xlabel("")
    plt.legend(loc='upper left')

    plt.savefig(output_dir / "4_fairness_gap.pdf", bbox_inches='tight')
    plt.close()

# --- MAIN ---
def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Load Data
    df_ts, df_sum = collect_all_results(BASE_RESULTS_DIR, target_defense='martfl')

    if df_sum.empty:
        print("No data found.")
        return

    # Generate Plots
    plot_selection_dynamics(df_ts, output_dir)
    plot_economic_exploitation(df_sum, output_dir)
    plot_strategy_heatmap(df_ts, output_dir)
    plot_manipulation_fairness(df_sum, output_dir)

    print(f"\n✅ Visualization Complete. Check {output_dir}")

if __name__ == "__main__":
    main()