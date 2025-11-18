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
FIGURE_OUTPUT_DIR = "./figures/step7_martfl_final_visuals"

EXPLORATION_ROUNDS = 30


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


def collect_all_results(base_dir: str, target_defense: Optional[str] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_seller_dfs, all_global_log_dfs, all_summary_rows = [], [], []
    base_path = Path(base_dir)

    scenario_folders = list(base_path.glob("step7_*"))
    print(f"Found {len(scenario_folders)} folders matching 'step7_*'")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        parsed_defense = scenario_params.get("defense", "unknown")
        if parsed_defense == "unknown":
            continue

        if target_defense and parsed_defense != target_defense:
            continue

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
                    except Exception:
                        pass

                # --- LOAD TRAINING LOG ---
                log_file = run_dir / 'training_log.csv'
                if log_file.exists():
                    try:
                        df_log = pd.read_csv(log_file, usecols=lambda c: c in ['round', 'val_acc'], on_bad_lines='skip')
                        if 'val_acc' in df_log.columns:
                            df_log['seed_id'] = seed_id
                            df_log = df_log.assign(**scenario_params)
                            all_global_log_dfs.append(df_log)
                    except Exception:
                        pass

                # --- LOAD SUMMARY (FIXED: Time-Average Logic) ---
                with open(final_metrics_file, 'r') as f:
                    final_metrics = json.load(f)

                adv_sel, ben_sel = 0.0, 0.0

                if not df_seller_run.empty:
                    # Use valid history AFTER exploration to avoid skewing results
                    valid_history = df_seller_run[df_seller_run['round'] > EXPLORATION_ROUNDS]
                    if valid_history.empty: valid_history = df_seller_run  # Fallback

                    if not valid_history.empty:
                        # 1. Benign Average
                        ben_sellers = valid_history[valid_history['seller_type'] == 'Benign']
                        if not ben_sellers.empty:
                            ben_sel = ben_sellers['selected'].mean()

                        # 2. Adversary Average
                        if scenario_params['threat_model'] == 'baseline':
                            # PROXY: Use bn_0, bn_1, bn_2
                            proxy_sellers = valid_history[valid_history['seller_id'].isin(['bn_0', 'bn_1', 'bn_2'])]
                            if not proxy_sellers.empty:
                                adv_sel = proxy_sellers['selected'].mean()
                        else:
                            # REAL
                            adv_sellers = valid_history[valid_history['seller_type'] == 'Adversary']
                            if not adv_sellers.empty:
                                adv_sel = adv_sellers['selected'].mean()

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
    if len(unique_sources) < 5:
        for s in unique_sources: print(f"      - {s}")
    else:
        print("      (List truncated...)")


def plot_selection_rate_curves(df: pd.DataFrame, baseline_sel: float, baseline_adv: float, output_dir: Path):
    """
    UPDATED: Uses Window=10 for Benign (Smooth) and Window=3 for Adversary (Reactive).
    """
    if df.empty: return
    print_plot_sources("Plot 1 (Selection Rate Curves)", df)

    # Separate for different smoothing
    df_benign = df[df['seller_type'] == 'Benign'].copy()
    df_adv = df[df['seller_type'] == 'Adversary'].copy()

    group_cols = ['seed_id', 'defense', 'threat_label', 'adaptive_mode', 'round']
    roll_cols = ['seed_id', 'defense', 'threat_label', 'adaptive_mode']

    # 1. Smooth Benign
    df_benign_agg = df_benign.groupby(group_cols)['selected'].mean().reset_index()
    df_benign_agg['seller_type'] = 'Benign'
    df_benign_agg['rolling_sel_rate'] = df_benign_agg.groupby(roll_cols)['selected'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()  # Window 10
    )

    # 2. Reactive Adversary
    df_adv_agg = df_adv.groupby(group_cols)['selected'].mean().reset_index()
    df_adv_agg['seller_type'] = 'Adversary'
    df_adv_agg['rolling_sel_rate'] = df_adv_agg.groupby(roll_cols)['selected'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()  # Window 3
    )

    df_agg = pd.concat([df_benign_agg, df_adv_agg], ignore_index=True)

    for defense in df_agg['defense'].unique():
        for threat in df_agg['threat_label'].unique():
            df_facet = df_agg[(df_agg['defense'] == defense) & (df_agg['threat_label'] == threat)]
            if df_facet.empty: continue
            threat_file = threat.replace(' ', '').replace('.', '')

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_facet, x='round', y='rolling_sel_rate', hue='seller_type', style='adaptive_mode',
                         palette={'Adversary': 'red', 'Benign': 'blue'}, lw=2.5, errorbar=('ci', 95))

            if baseline_sel:
                plt.axhline(y=baseline_sel, color='blue', linestyle='--', alpha=0.5, label='Benign Avg (Baseline)')

            plt.legend()
            plt.title(f'Selection Rate: {defense.upper()} vs {threat}')
            plt.ylim(-0.05, 1.05)
            plt.savefig(output_dir / f"plot1_sel_rate_{defense}_{threat_file}.pdf", bbox_inches='tight')
            plt.close('all')


def plot_attacker_advantage(df: pd.DataFrame, output_dir: Path):
    """
    NEW: Plots (Adversary - Benign) to visualize evasion success explicitly.
    """
    if df.empty: return
    print_plot_sources("Plot 1B (Attacker Advantage)", df)

    group_cols = ['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode', 'seller_type']
    df_agg = df.groupby(group_cols)['selected'].mean().reset_index()

    df_piv = df_agg.pivot_table(index=['round', 'seed_id', 'defense', 'threat_label', 'adaptive_mode'],
                                columns='seller_type', values='selected').reset_index()

    if 'Adversary' in df_piv.columns and 'Benign' in df_piv.columns:
        df_piv['Evasion_Score'] = df_piv['Adversary'] - df_piv['Benign']

        # Smooth the advantage line slightly for readability
        df_piv['Evasion_Score_Smooth'] = df_piv.groupby(['seed_id', 'threat_label'])['Evasion_Score'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        for defense in df_piv['defense'].unique():
            data = df_piv[df_piv['defense'] == defense]
            if data.empty: continue

            plt.figure(figsize=(10, 6))
            # Zones
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
            plt.axhspan(0, 1.0, facecolor='red', alpha=0.05, label='Evasion Zone (Attacker Wins)')
            plt.axhspan(-1.0, 0, facecolor='green', alpha=0.05, label='Detection Zone (Defense Wins)')

            sns.lineplot(data=data, x='round', y='Evasion_Score_Smooth', hue='threat_label', style='adaptive_mode',
                         lw=2.5)

            plt.title(f'Evasion Success: {defense.upper()} (Positive = Avoiding Detection)')
            plt.ylabel('Selection Advantage (Attacker - Benign)')
            plt.legend()
            plt.savefig(output_dir / f"plot_NEW_advantage_{defense}.pdf", bbox_inches='tight')
            plt.close('all')


def plot_final_summary_distribution(df: pd.DataFrame, output_dir: Path):
    """
    UPDATED: Uses Box Plot + Strip Plot to show the "Possibility" of evasion (outliers).
    """
    if df.empty: return
    print_plot_sources("Plot 3 (Summary Distribution)", df)

    x_order = ['0. Baseline (No Attack)', '1. Black-Box', '2. Grad-Inversion', '3. Oracle']
    x_order = [x for x in x_order if x in df['threat_label'].unique()]

    for defense in df['defense'].unique():
        data = df[df['defense'] == defense]
        if data.empty: continue

        plt.figure(figsize=(12, 6))

        # 1. Box Plot for Distribution Stats
        sns.boxplot(data=data, x='threat_label', y='adv_sel_rate', hue='adaptive_mode',
                    order=x_order, showfliers=False, boxprops={'alpha': 0.4})

        # 2. Strip Plot for Individual Experiments (The "Possibility")
        sns.stripplot(data=data, x='threat_label', y='adv_sel_rate', hue='adaptive_mode',
                      order=x_order, dodge=True, jitter=True, color='black', size=6, alpha=0.7)

        # Formatting
        plt.axhline(y=data[data['threat_label'] == '0. Baseline (No Attack)']['ben_sel_rate'].mean(),
                    color='blue', linestyle='--', alpha=0.5, label='Baseline Honest Rate')

        plt.title(f"Range of Attacker Evasion: {defense.upper()} (Dots = Individual Seeds)")
        plt.ylabel("Adversary Selection Rate (Avg post-exploration)")
        plt.xticks(rotation=15)

        # Fix duplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter to keep unique labels only
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig(output_dir / f"plot3_distribution_summary_{defense}.pdf", bbox_inches='tight')
        plt.close('all')


def plot_stealth_vs_damage(df_sum: pd.DataFrame, output_dir: Path):
    """
    NEW: Correlates Stealth (Selection Rate) vs Damage (Accuracy).
    """
    if df_sum.empty: return
    print_plot_sources("Plot 4 (Stealth vs Damage)", df_sum)

    plt.figure(figsize=(9, 7))

    # Filter out baseline for the scatter points
    attacks = df_sum[df_sum['threat_label'] != '0. Baseline (No Attack)']
    if attacks.empty: return

    # Scatter Plot
    sns.scatterplot(data=attacks, x='adv_sel_rate', y='acc',
                    hue='adaptive_mode', style='threat_label', s=150, alpha=0.8)

    # Reference Lines
    baseline_row = df_sum[df_sum['threat_label'] == '0. Baseline (No Attack)']
    if not baseline_row.empty:
        base_acc = baseline_row['acc'].mean()
        base_sel = baseline_row['ben_sel_rate'].mean()
        plt.axhline(y=base_acc, color='grey', linestyle='--', label='Baseline Accuracy')
        plt.axvline(x=base_sel, color='grey', linestyle=':', label='Baseline Selection Rate')

    plt.title("Stealth vs. Damage: The 'Danger Zone' is Bottom-Right")
    plt.xlabel("Attacker Selection Rate (Stealth)")
    plt.ylabel("Global Accuracy (Performance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_dir / "plot_NEW_stealth_damage_correlation.pdf", bbox_inches='tight')
    plt.close('all')


# --- MAIN ---
def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Load Data
    df_seller_ts, df_global_ts, df_summary = collect_all_results(BASE_RESULTS_DIR, target_defense='martfl')
    if df_summary.empty: return

    # 2. Extract Baselines
    baseline_row = df_summary[df_summary['threat_label'] == '0. Baseline (No Attack)']
    baseline_sel = baseline_row['ben_sel_rate'].mean() if not baseline_row.empty else None
    baseline_adv = baseline_row['adv_sel_rate'].mean() if not baseline_row.empty else None

    # 3. Run Plots
    # A. Standard Selection Curves (Smoothed)
    plot_selection_rate_curves(df_seller_ts, baseline_sel, baseline_adv, output_dir)

    # B. New "Advantage" Plot
    plot_attacker_advantage(df_seller_ts, output_dir)

    # C. New "Possibility/Distribution" Box Plot
    plot_final_summary_distribution(df_summary, output_dir)

    # D. New "Stealth vs Damage" Scatter
    plot_stealth_vs_damage(df_summary, output_dir)

    print(f"\n✅ Analysis complete. Check '{FIGURE_OUTPUT_DIR}'")


if __name__ == "__main__":
    main()