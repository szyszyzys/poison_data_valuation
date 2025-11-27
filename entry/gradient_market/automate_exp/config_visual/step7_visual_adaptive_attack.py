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
FIGURE_OUTPUT_DIR = "./figures/step7_martfl_final_visuals"
EXPLORATION_ROUNDS = 30

# --- Global Style Settings ---
def set_publication_style():
    """Sets professional aesthetics for academic figures."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8

# Consistent Colors
COLOR_MAP = {
    'Adversary': '#D62728',   # Red
    'Benign': '#1F77B4',      # Blue
    '0. Baseline (No Attack)': 'gray',
    '1. Black-Box': '#FF7F0E', # Orange
    '2. Grad-Inversion': '#2CA02C', # Green
    '3. Oracle': '#9467BD'     # Purple
}

# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Robust parser that handles variable underscores in threat/mode names.
    """
    try:
        # 1. Handle Baseline
        if scenario_name.startswith('step7_baseline_no_attack_'):
            parts = scenario_name.replace('step7_baseline_no_attack_', '').split('_')
            defense = parts[0]
            dataset = parts[1] if len(parts) > 1 else "CIFAR100"

            return {
                "threat_model": "baseline",
                "adaptive_mode": "N/A",
                "defense": defense,
                "dataset": dataset,
                "threat_label": "0. Baseline (No Attack)"
            }

        # 2. Handle Adaptive
        elif scenario_name.startswith('step7_adaptive_'):
            rest = scenario_name.replace('step7_adaptive_', '')

            threat_model = "unknown"
            if rest.startswith('black_box_'):
                threat_model = 'black_box'
                rest = rest.replace('black_box_', '')
            elif rest.startswith('gradient_inversion_'):
                threat_model = 'gradient_inversion'
                rest = rest.replace('gradient_inversion_', '')
            elif rest.startswith('oracle_'):
                threat_model = 'oracle'
                rest = rest.replace('oracle_', '')

            if threat_model == "unknown":
                return {"defense": "unknown"}

            adaptive_mode = "unknown"
            if rest.startswith('data_poisoning_'):
                adaptive_mode = 'data_poisoning'
                rest = rest.replace('data_poisoning_', '')
            elif rest.startswith('gradient_manipulation_'):
                adaptive_mode = 'gradient_manipulation'
                rest = rest.replace('gradient_manipulation_', '')

            parts = rest.split('_')
            defense = parts[0]
            dataset = parts[1] if len(parts) > 1 else "CIFAR100"

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

def plot_selection_gap_highlight(df: pd.DataFrame, output_dir: Path):
    """
    Visualizes the GAP between Benign and Adversary.
    Shades the area Red if Adv > Benign (Advantage), Blue otherwise.
    """
    if df.empty: return
    print("Generating 'Gap Highlight' plots...")

    # Aggregate across seeds first to get a clean mean line for the area fill
    group_cols = ['defense', 'threat_label', 'adaptive_mode', 'round', 'seller_type']
    df_agg = df.groupby(group_cols)['selected'].mean().reset_index()

    # Pivot so we have 'Adversary' and 'Benign' as columns for easy math
    df_pivoted = df_agg.pivot_table(
        index=['defense', 'threat_label', 'adaptive_mode', 'round'],
        columns='seller_type',
        values='selected'
    ).reset_index()

    # Ensure we have both columns (handle cases where one might be missing)
    if 'Adversary' not in df_pivoted.columns: df_pivoted['Adversary'] = 0.0
    if 'Benign' not in df_pivoted.columns: df_pivoted['Benign'] = 0.0

    # Smooth the lines slightly for better visualization
    df_pivoted['Adv_Smooth'] = df_pivoted.groupby(['defense', 'threat_label'])['Adversary'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_pivoted['Ben_Smooth'] = df_pivoted.groupby(['defense', 'threat_label'])['Benign'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    unique_defenses = df_pivoted['defense'].unique()

    for defense in unique_defenses:
        defense_data = df_pivoted[df_pivoted['defense'] == defense]

        for threat in defense_data['threat_label'].unique():
            subset = defense_data[defense_data['threat_label'] == threat]
            if subset.empty: continue

            plt.figure(figsize=(10, 6))

            # 1. Draw the "Event Horizon" (Round 30)
            plt.axvline(x=EXPLORATION_ROUNDS, color='black', linestyle='--', linewidth=2, alpha=0.8)
            plt.text(EXPLORATION_ROUNDS + 1, 0.95, "Attack Starts", fontsize=12, fontweight='bold', ha='left')

            # 2. Draw Lines
            plt.plot(subset['round'], subset['Ben_Smooth'], color=COLOR_MAP['Benign'], label='Benign', lw=2)
            plt.plot(subset['round'], subset['Adv_Smooth'], color=COLOR_MAP['Adversary'], label='Adversary', lw=2)

            # 3. FILL THE GAP (The Key Visual)
            # Fill Green/Blue where Benign > Adversary (System Safe)
            plt.fill_between(
                subset['round'],
                subset['Ben_Smooth'],
                subset['Adv_Smooth'],
                where=(subset['Ben_Smooth'] >= subset['Adv_Smooth']),
                interpolate=True, color=COLOR_MAP['Benign'], alpha=0.15, label='Defense Advantage'
            )

            # Fill Red where Adversary > Benign (System Compromised)
            plt.fill_between(
                subset['round'],
                subset['Ben_Smooth'],
                subset['Adv_Smooth'],
                where=(subset['Adv_Smooth'] > subset['Ben_Smooth']),
                interpolate=True, color=COLOR_MAP['Adversary'], alpha=0.25, hatch='//', label='Attacker Advantage'
            )

            threat_file = threat.replace(' ', '').replace('.', '')
            plt.title(f"Attack Impact Analysis: {defense.upper()}\n{threat}")
            plt.ylabel("Selection Rate")
            plt.xlabel("Round")
            plt.ylim(0, 1.05)
            plt.legend(loc='upper right')

            plt.savefig(output_dir / f"visual_gap_{defense}_{threat_file}.pdf", bbox_inches='tight')
            plt.close()


def plot_pre_post_change_bar(df: pd.DataFrame, output_dir: Path):
    """
    Quantifies the "Change": Bar chart comparing Pre-Attack vs Post-Attack averages.
    """
    if df.empty: return
    print("Generating 'Pre/Post Change' plots...")

    # 1. Label data as Pre-Attack or Post-Attack
    df_calc = df.copy()
    df_calc['Phase'] = df_calc['round'].apply(lambda x: 'Pre-Attack (Exploration)' if x <= EXPLORATION_ROUNDS else 'Post-Attack')

    # 2. Filter only for Adversary (since we care about their gain)
    df_adv = df_calc[df_calc['seller_type'] == 'Adversary']

    # 3. Calculate Mean Selection Rate per Phase per Seed
    bar_data = df_adv.groupby(['defense', 'threat_label', 'adaptive_mode', 'Phase'])['selected'].mean().reset_index()

    for defense in bar_data['defense'].unique():
        subset = bar_data[bar_data['defense'] == defense]
        if subset.empty: continue

        plt.figure(figsize=(10, 6))

        # Create Bar Chart
        ax = sns.barplot(
            data=subset,
            x='threat_label',
            y='selected',
            hue='Phase',
            palette={'Pre-Attack (Exploration)': 'gray', 'Post-Attack': '#D62728'},
            edgecolor='black',
            alpha=0.9
        )

        # Add specific value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)

        plt.title(f"Impact of Attack Activation: {defense.upper()}")
        plt.ylabel("Avg Adversary Selection Rate")
        plt.xlabel("")
        plt.legend(title="Experiment Phase")
        plt.xticks(rotation=15, ha='right')

        plt.savefig(output_dir / f"visual_bar_change_{defense}.pdf", bbox_inches='tight')
        plt.close()
def collect_all_results(base_dir: str, target_defense: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_seller_dfs, all_global_log_dfs, all_summary_rows = [], [], []
    base_path = Path(base_dir)

    scenario_folders = list(base_path.glob("step7_*"))
    print(f"Found {len(scenario_folders)} folders matching 'step7_*'")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        parsed_defense = scenario_params.get("defense", "unknown")
        if parsed_defense == "unknown": continue
        if target_defense and parsed_defense != target_defense: continue

        marker_files = list(scenario_path.rglob('final_metrics.json'))

        for final_metrics_file in marker_files:
            try:
                run_dir = final_metrics_file.parent
                seed_id = f"{scenario_path.name}/{run_dir.name}"

                # --- LOAD SELLER METRICS ---
                seller_file = run_dir / 'seller_metrics.csv'
                df_seller_run = pd.DataFrame()
                if seller_file.exists():
                    try:
                        df_seller_run = pd.read_csv(seller_file, on_bad_lines='skip')
                        df_seller_run['seed_id'] = seed_id
                        df_seller_run = df_seller_run.assign(**scenario_params)
                        df_seller_run['seller_type'] = df_seller_run['seller_id'].apply(
                            lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign')
                        all_seller_dfs.append(df_seller_run)
                    except Exception: pass

                # --- LOAD TRAINING LOG ---
                log_file = run_dir / 'training_log.csv'
                if log_file.exists():
                    try:
                        df_log = pd.read_csv(log_file, usecols=lambda c: c in ['round', 'val_acc'], on_bad_lines='skip')
                        if 'val_acc' in df_log.columns:
                            df_log['seed_id'] = seed_id
                            df_log = df_log.assign(**scenario_params)
                            all_global_log_dfs.append(df_log)
                    except Exception: pass

                # --- LOAD SUMMARY ---
                with open(final_metrics_file, 'r') as f:
                    final_metrics = json.load(f)

                adv_sel, ben_sel = 0.0, 0.0

                if not df_seller_run.empty:
                    # Filter for stability
                    valid_history = df_seller_run[df_seller_run['round'] > EXPLORATION_ROUNDS]
                    if valid_history.empty: valid_history = df_seller_run

                    if not valid_history.empty:
                        ben_sellers = valid_history[valid_history['seller_type'] == 'Benign']
                        if not ben_sellers.empty: ben_sel = ben_sellers['selected'].mean()

                        if scenario_params['threat_model'] == 'baseline':
                            proxy_sellers = valid_history[valid_history['seller_id'].isin(['bn_0', 'bn_1', 'bn_2'])]
                            if not proxy_sellers.empty: adv_sel = proxy_sellers['selected'].mean()
                        else:
                            adv_sellers = valid_history[valid_history['seller_type'] == 'Adversary']
                            if not adv_sellers.empty: adv_sel = adv_sellers['selected'].mean()

                all_summary_rows.append({
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': final_metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel,
                    'ben_sel_rate': ben_sel
                })
            except Exception as e:
                print(f"Error processing {final_metrics_file}: {e}")

    df_s = pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()
    df_g = pd.concat(all_global_log_dfs, ignore_index=True) if all_global_log_dfs else pd.DataFrame()
    df_sum = pd.DataFrame(all_summary_rows)

    print(f"Loaded {len(df_sum)} summary rows.")
    return df_s, df_g, df_sum


# --- PLOTTING FUNCTIONS ---

def print_plot_sources(plot_name: str, df: pd.DataFrame):
    print(f"\n>>> [DEBUG] Sources for {plot_name}:")
    if df.empty:
        print("    No data.")
        return
    unique_sources = df['seed_id'].unique()
    print(f"    Total Unique Files: {len(unique_sources)}")


def plot_selection_rate_curves(df: pd.DataFrame, baseline_sel: float, baseline_adv: float, output_dir: Path):
    """
    Plot 1: Dynamic Selection Rates over Time.
    """
    if df.empty: return
    print_plot_sources("Plot 1 (Selection Rate Curves)", df)

    df_benign = df[df['seller_type'] == 'Benign'].copy()
    df_adv = df[df['seller_type'] == 'Adversary'].copy()

    group_cols = ['seed_id', 'defense', 'threat_label', 'adaptive_mode', 'round']

    # Calculate Benign Smoothing (Slow Window)
    benign_means = df_benign.groupby(group_cols)['selected'].mean().reset_index()
    benign_means['seller_type'] = 'Benign'
    benign_means = benign_means.sort_values('round')
    benign_means['rolling_sel_rate'] = benign_means.groupby(['seed_id'])['selected'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    # Calculate Adversary Smoothing (Fast Window)
    adv_means = df_adv.groupby(group_cols)['selected'].mean().reset_index()
    adv_means['seller_type'] = 'Adversary'
    adv_means = adv_means.sort_values('round')
    adv_means['rolling_sel_rate'] = adv_means.groupby(['seed_id'])['selected'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df_agg = pd.concat([benign_means, adv_means], ignore_index=True)

    for defense in df_agg['defense'].unique():
        for threat in df_agg['threat_label'].unique():
            df_facet = df_agg[(df_agg['defense'] == defense) & (df_agg['threat_label'] == threat)]
            if df_facet.empty: continue
            threat_file = threat.replace(' ', '').replace('.', '')

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_facet, x='round', y='rolling_sel_rate', hue='seller_type', style='adaptive_mode',
                         palette=COLOR_MAP, lw=2.5, errorbar=('ci', 95))

            if baseline_sel:
                plt.axhline(y=baseline_sel, color=COLOR_MAP['Benign'], linestyle=':', alpha=0.7, label='Benign Baseline')

            plt.title(f'Adaptive Defense Dynamics: {defense.upper()}\nThreat: {threat}')
            plt.ylabel("Selection Rate (Smoothed)")
            plt.xlabel("Communication Round")
            plt.ylim(-0.05, 1.05)
            plt.legend(loc='lower right')

            plt.savefig(output_dir / f"plot1_sel_rate_{defense}_{threat_file}.pdf", bbox_inches='tight')
            plt.close('all')


def plot_attacker_advantage(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: Attacker Advantage (Adv Rate - Benign Rate).
    Visualizes the "Cat and Mouse" game.
    """
    if df.empty: return
    print_plot_sources("Plot 2 (Attacker Advantage)", df)

    group_cols = ['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode', 'seller_type']
    df_agg = df.groupby(group_cols)['selected'].mean().reset_index()

    df_piv = df_agg.pivot_table(index=['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode'],
                                columns='seller_type', values='selected').reset_index()

    if 'Adversary' in df_piv.columns and 'Benign' in df_piv.columns:
        df_piv['Evasion_Score'] = df_piv['Adversary'] - df_piv['Benign']

        # Smooth
        df_piv = df_piv.sort_values('round')
        df_piv['Evasion_Score_Smooth'] = df_piv.groupby(['seed_id', 'threat_label'])['Evasion_Score'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        for defense in df_piv['defense'].unique():
            data = df_piv[df_piv['defense'] == defense]
            if data.empty: continue

            plt.figure(figsize=(10, 6))

            # Zones
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
            plt.axhspan(0, 1.0, facecolor=COLOR_MAP['Adversary'], alpha=0.1, label='Attack Wins (Evasion)')
            plt.axhspan(-1.0, 0, facecolor=COLOR_MAP['Benign'], alpha=0.1, label='Defense Wins (Detection)')

            sns.lineplot(data=data, x='round', y='Evasion_Score_Smooth', hue='threat_label', style='adaptive_mode',
                         palette=COLOR_MAP, lw=2.5)

            plt.title(f'Evasion Analysis: {defense.upper()}')
            plt.ylabel('Advantage Score (Adversary Rate - Benign Rate)')
            plt.xlabel('Communication Round')
            plt.legend(title="Threat Model", bbox_to_anchor=(1.02, 1), loc='upper left')

            plt.savefig(output_dir / f"plot2_advantage_{defense}.pdf", bbox_inches='tight')
            plt.close('all')


def plot_final_summary_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Summary Distribution (Box + Strip).
    Shows the variance across seeds.
    """
    if df.empty: return
    print_plot_sources("Plot 3 (Summary Distribution)", df)

    x_order = ['0. Baseline (No Attack)', '1. Black-Box', '2. Grad-Inversion', '3. Oracle']
    x_order = [x for x in x_order if x in df['threat_label'].unique()]

    for defense in df['defense'].unique():
        data = df[df['defense'] == defense]
        if data.empty: continue

        plt.figure(figsize=(11, 6))

        # Box Plot
        sns.boxplot(data=data, x='threat_label', y='adv_sel_rate', hue='adaptive_mode',
                    order=x_order, showfliers=False, boxprops={'alpha': 0.5})

        # Strip Plot (for individual seeds)
        sns.stripplot(data=data, x='threat_label', y='adv_sel_rate', hue='adaptive_mode',
                      order=x_order, dodge=True, jitter=True, color='black', size=5, alpha=0.7)

        # Reference Line
        baseline_avg = data[data['threat_label'] == '0. Baseline (No Attack)']['ben_sel_rate'].mean()
        if pd.notna(baseline_avg):
            plt.axhline(y=baseline_avg, color=COLOR_MAP['Benign'], linestyle='--', alpha=0.6, label='Normal Selection Rate')

        plt.title(f"Defense Resilience: {defense.upper()}\n(Lower is Better)")
        plt.ylabel("Final Adversary Selection Rate")
        plt.xlabel("")
        plt.xticks(rotation=15, ha='right')

        # Fix Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.savefig(output_dir / f"plot3_distribution_summary_{defense}.pdf", bbox_inches='tight')
        plt.close('all')


def plot_stealth_vs_damage(df_sum: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Stealth vs Damage (Scatter).
    """
    if df_sum.empty: return
    print_plot_sources("Plot 4 (Stealth vs Damage)", df_sum)

    attacks = df_sum[df_sum['threat_label'] != '0. Baseline (No Attack)'].copy()
    if attacks.empty: return

    plt.figure(figsize=(10, 8))

    # 1. Danger Zone (High Stealth, Low Accuracy - for poisoning)
    # Note: If MartFL is successful, accuracy should be high and selection low.
    # If Attack is successful (untargeted), accuracy is low and selection high.
    current_axis = plt.gca()
    rect = patches.Rectangle((0.5, 0.0), 0.5, 0.5, linewidth=0, edgecolor='none', facecolor='red', alpha=0.1)
    current_axis.add_patch(rect)
    plt.text(0.95, 0.05, "DANGER ZONE\n(Attack Wins)", color='red',
             ha='right', va='bottom', fontsize=12, fontweight='bold')

    # 2. Scatter Plot
    sns.scatterplot(data=attacks, x='adv_sel_rate', y='acc',
                    hue='threat_label', style='adaptive_mode',
                    palette=COLOR_MAP, s=120, alpha=0.85, edgecolor='black')

    # 3. Baseline Reference
    baseline_row = df_sum[df_sum['threat_label'] == '0. Baseline (No Attack)']
    if not baseline_row.empty:
        base_acc = baseline_row['acc'].mean()
        base_sel = baseline_row['ben_sel_rate'].mean()
        plt.axhline(y=base_acc, color='gray', linestyle='--', label='Baseline Accuracy')
        plt.axvline(x=base_sel, color='gray', linestyle=':', label='Baseline Selection Rate')
        plt.plot(base_sel, base_acc, marker='*', color='gold', markersize=20, markeredgecolor='black', label='Ideal State')

    plt.title("Stealth vs. Impact Analysis")
    plt.xlabel("Attacker Selection Rate (Stealth)")
    plt.ylabel("Global Model Accuracy (Performance)")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.savefig(output_dir / "plot4_stealth_damage_correlation.pdf", bbox_inches='tight')
    plt.close('all')


# --- MAIN ---
def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Apply Style
    set_publication_style()

    # 2. Load Data (Targeting MartFL or remove arg to see all)
    df_seller_ts, df_global_ts, df_summary = collect_all_results(BASE_RESULTS_DIR, target_defense='martfl')

    if df_summary.empty:
        print("No data found. Exiting.")
        return

    # 3. Extract Baselines
    baseline_row = df_summary[df_summary['threat_label'] == '0. Baseline (No Attack)']
    baseline_sel = baseline_row['ben_sel_rate'].mean() if not baseline_row.empty else None
    baseline_adv = baseline_row['adv_sel_rate'].mean() if not baseline_row.empty else None

    # 4. Generate Plots
    print("\n--- Generating Plots ---")
    plot_selection_rate_curves(df_seller_ts, baseline_sel, baseline_adv, output_dir)
    plot_attacker_advantage(df_seller_ts, output_dir)
    plot_final_summary_distribution(df_summary, output_dir)
    plot_stealth_vs_damage(df_summary, output_dir)
    print("\n--- Generating Enhanced Visuals ---")
    plot_selection_gap_highlight(df_seller_ts, output_dir)
    plot_pre_post_change_bar(df_seller_ts, output_dir)
    print(f"\n✅ Analysis complete. Check '{FIGURE_OUTPUT_DIR}'")


if __name__ == "__main__":
    main()