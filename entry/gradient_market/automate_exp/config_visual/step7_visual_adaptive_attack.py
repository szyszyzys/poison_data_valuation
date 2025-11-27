import pandas as pd
import json
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
TARGET_DEFENSE = "martfl"

# --- Global Style Settings ---
def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['legend.frameon'] = True

# Consistent Colors
COLOR_MAP = {
    'Adversary': '#D62728',   # Red
    'Benign': '#1F77B4',      # Blue
    '0. Baseline': 'gray',
    '1. Black-Box': '#FF7F0E', # Orange
    '2. Grad-Inversion': '#2CA02C', # Green
    '3. Oracle': '#9467BD'     # Purple
}

# ---------------------
# 1. ROBUST PARSER (Copied from your working Debug Script)
# ---------------------
def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    try:
        # 1. Handle Baseline
        if 'baseline_no_attack' in scenario_name:
            parts = scenario_name.replace('step7_baseline_no_attack_', '').split('_')
            return {
                "threat_model": "baseline",
                "adaptive_mode": "N/A",
                "defense": parts[0],
                "dataset": parts[1] if len(parts) > 1 else "CIFAR100",
                "threat_label": "0. Baseline"
            }

        # 2. Handle Adaptive
        if not scenario_name.startswith('step7_adaptive_'):
            return {"defense": "unknown"}

        rest = scenario_name.replace('step7_adaptive_', '')

        # Threat Model Extraction
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

        # Adaptive Mode Extraction
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

        threat_labels = {
            'black_box': '1. Black-Box',
            'gradient_inversion': '2. Grad-Inversion',
            'oracle': '3. Oracle'
        }

        return {
            "threat_model": threat_model,
            "adaptive_mode": adaptive_mode,
            "defense": defense,
            "dataset": dataset,
            "threat_label": threat_labels.get(threat_model, threat_model)
        }

    except Exception as e:
        print(f"Error parsing '{scenario_name}': {e}")
        return {"defense": "unknown"}


# ---------------------
# 2. DATA COLLECTION
# ---------------------
def collect_all_results(base_dir: str, target_defense: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_seller_dfs = []
    all_summary_rows = []
    base_path = Path(base_dir)

    scenario_folders = list(base_path.glob("step7_*"))
    print(f"Scanning {len(scenario_folders)} folders for defense='{target_defense}'...")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        # Exact logic from Debug script:
        if scenario_params.get("defense") != target_defense:
            continue

        marker_files = list(scenario_path.rglob('final_metrics.json'))

        for final_metrics_file in marker_files:
            try:
                run_dir = final_metrics_file.parent
                seed_id = f"{scenario_path.name}/{run_dir.name}"

                # 1. Load Seller Metrics
                seller_file = run_dir / 'seller_metrics.csv'
                if seller_file.exists():
                    df_seller = pd.read_csv(seller_file, on_bad_lines='skip')
                    df_seller['seed_id'] = seed_id
                    df_seller = df_seller.assign(**scenario_params)

                    df_seller['seller_type'] = df_seller['seller_id'].apply(
                        lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign')

                    all_seller_dfs.append(df_seller)

                    # Calc stats for summary
                    valid = df_seller[df_seller['round'] > EXPLORATION_ROUNDS]
                    if valid.empty: valid = df_seller
                    adv_sel = valid[valid['seller_type'] == 'Adversary']['selected'].mean()
                else:
                    adv_sel = 0.0

                # 2. Load Final Accuracy
                with open(final_metrics_file, 'r') as f:
                    metrics = json.load(f)

                all_summary_rows.append({
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel if pd.notna(adv_sel) else 0.0
                })

            except Exception as e:
                print(f"Error reading {run_dir}: {e}")

    df_s = pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()
    df_sum = pd.DataFrame(all_summary_rows)

    print(f"Loaded {len(df_sum)} runs. Threats found: {df_sum['threat_label'].unique()}")
    return df_s, df_sum


# ---------------------
# 3. PLOTTING FUNCTIONS
# ---------------------

def plot_gap_highlight_martfl(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    print("Generating 'Gap Highlight' plots...")

    # Aggregate
    df_agg = df.groupby(['threat_label', 'adaptive_mode', 'round', 'seller_type'])['selected'].mean().reset_index()

    # Pivot
    df_piv = df_agg.pivot_table(
        index=['threat_label', 'adaptive_mode', 'round'],
        columns='seller_type', values='selected'
    ).reset_index()

    if 'Adversary' not in df_piv.columns: df_piv['Adversary'] = 0.0
    if 'Benign' not in df_piv.columns: df_piv['Benign'] = 0.0

    # Smooth
    df_piv['Adv_Smooth'] = df_piv.groupby(['threat_label', 'adaptive_mode'])['Adversary'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_piv['Ben_Smooth'] = df_piv.groupby(['threat_label', 'adaptive_mode'])['Benign'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    threats = df_piv['threat_label'].unique()

    for threat in threats:
        if threat == '0. Baseline': continue

        threat_data = df_piv[df_piv['threat_label'] == threat]

        for mode in threat_data['adaptive_mode'].unique():
            subset = threat_data[threat_data['adaptive_mode'] == mode]

            plt.figure(figsize=(10, 6))

            # Vertical Line
            plt.axvline(x=EXPLORATION_ROUNDS, color='black', linestyle='--', linewidth=2, alpha=0.7)
            plt.text(EXPLORATION_ROUNDS + 1, 0.95, "Attack Starts", fontsize=12, fontweight='bold', va='top')

            # Lines
            plt.plot(subset['round'], subset['Ben_Smooth'], color=COLOR_MAP['Benign'], label='Honest', alpha=0.8)
            plt.plot(subset['round'], subset['Adv_Smooth'], color=COLOR_MAP['Adversary'], label='Attacker', linewidth=3.5)

            # Highlight Gaps
            plt.fill_between(
                subset['round'], subset['Ben_Smooth'], subset['Adv_Smooth'],
                where=(subset['Adv_Smooth'] > subset['Ben_Smooth']),
                interpolate=True, color=COLOR_MAP['Adversary'], alpha=0.2, hatch='//', label='Defense Evasion'
            )

            plt.fill_between(
                subset['round'], subset['Ben_Smooth'], subset['Adv_Smooth'],
                where=(subset['Ben_Smooth'] >= subset['Adv_Smooth']),
                interpolate=True, color=COLOR_MAP['Benign'], alpha=0.1, label='Defense Robustness'
            )

            # Titles
            readable_mode = mode.replace('_', ' ').title()
            plt.title(f"MartFL Resilience: {threat}\nStrategy: {readable_mode}")
            plt.ylabel("Selection Rate")
            plt.xlabel("Communication Round")
            plt.ylim(0, 1.05)
            plt.legend(loc='lower right')

            fname = f"gap_martfl_{threat}_{mode}".replace(" ", "").replace(".", "")
            plt.savefig(output_dir / f"{fname}.pdf", bbox_inches='tight')
            plt.close()


def plot_stealth_vs_damage_martfl(df_sum: pd.DataFrame, output_dir: Path):
    if df_sum.empty: return
    print("Generating 'Stealth vs Damage' scatter...")

    attacks = df_sum[df_sum['threat_label'] != '0. Baseline'].copy()
    if attacks.empty: return

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=attacks,
        x='adv_sel_rate', y='acc',
        hue='threat_label', style='adaptive_mode',
        palette=COLOR_MAP, s=200, alpha=0.9, edgecolor='black', linewidth=1.5
    )

    plt.axvspan(0.5, 1.0, color='red', alpha=0.05)
    plt.text(0.95, 0.05, "High Evasion Zone", color='red',
             ha='right', transform=plt.gca().transAxes, fontweight='bold')

    plt.title("Attack Effectiveness Summary (MartFL)")
    plt.xlabel("Adversary Selection Rate (Stealth)")
    plt.ylabel("Global Model Accuracy")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.savefig(output_dir / "martfl_stealth_damage_scatter.pdf", bbox_inches='tight')
    plt.close()


# --- MAIN ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Generating MartFL Visuals ---")
    set_publication_style()

    # 1. Load Data
    df_s, df_sum = collect_all_results(BASE_RESULTS_DIR, target_defense=TARGET_DEFENSE)

    if df_s.empty:
        print("No data found!")
        return

    # 2. Plots
    plot_gap_highlight_martfl(df_s, output_dir)
    plot_stealth_vs_damage_martfl(df_sum, output_dir)

    print(f"\n✅ Visuals saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()