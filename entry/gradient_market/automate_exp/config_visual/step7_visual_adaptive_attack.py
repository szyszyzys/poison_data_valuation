import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step7_martfl_adaptive_analysis"
EXPLORATION_ROUNDS = 30  # Round where attack starts (Vertical Line)
TARGET_DEFENSE = "martfl" # Lock analysis to MartFL as requested

# --- Global Style Settings ---
def set_publication_style():
    """Sets professional aesthetics for academic figures."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.95
    plt.rcParams['figure.figsize'] = (10, 6)

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
# 1. PARSING LOGIC (Tailored for your ls output)
# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses 'step7_adaptive_<THREAT>_<MODE>_<DEFENSE>_<DATASET>'
    """
    try:
        # 1. Handle Baseline (if present in folder)
        if 'baseline_no_attack' in scenario_name:
            parts = scenario_name.replace('step7_baseline_no_attack_', '').split('_')
            defense = parts[0]
            dataset = parts[1] if len(parts) > 1 else "CIFAR100"
            return {
                "threat_model": "baseline",
                "adaptive_mode": "N/A",
                "defense": defense,
                "dataset": dataset,
                "threat_label": "0. Baseline"
            }

        # 2. Handle Adaptive Attacks
        if not scenario_name.startswith('step7_adaptive_'):
            return {"defense": "unknown"}

        rest = scenario_name.replace('step7_adaptive_', '')

        # A. Extract Threat Model
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

        # B. Extract Adaptive Mode
        adaptive_mode = "unknown"
        if rest.startswith('data_poisoning_'):
            adaptive_mode = 'data_poisoning'
            rest = rest.replace('data_poisoning_', '')
        elif rest.startswith('gradient_manipulation_'):
            adaptive_mode = 'gradient_manipulation'
            rest = rest.replace('gradient_manipulation_', '')

        # C. Extract Defense & Dataset
        parts = rest.split('_')
        defense = parts[0]
        dataset = parts[1] if len(parts) > 1 else "CIFAR100"

        # Pretty Labels
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
        print(f"Warning: Error parsing '{scenario_name}': {e}")
        return {"defense": "unknown"}


# ---------------------
# 2. DATA COLLECTION
# ---------------------

def collect_all_results(base_dir: str, target_defense: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_seller_dfs = []
    all_summary_rows = []
    base_path = Path(base_dir)

    # Use strict glob matching for step7
    scenario_folders = list(base_path.glob("step7_adaptive_*")) + list(base_path.glob("step7_baseline_*"))
    print(f"Scanning {len(scenario_folders)} folders for defense='{target_defense}'...")

    for scenario_path in scenario_folders:
        scenario_params = parse_scenario_name(scenario_path.name)

        # Filter: Only process the requested defense (MartFL)
        if scenario_params.get("defense") != target_defense:
            continue

        marker_files = list(scenario_path.rglob('final_metrics.json'))

        for final_metrics_file in marker_files:
            try:
                run_dir = final_metrics_file.parent
                seed_id = f"{scenario_path.name}/{run_dir.name}"

                # 1. Load Seller Metrics (Crucial for Selection Rate)
                seller_file = run_dir / 'seller_metrics.csv'
                if seller_file.exists():
                    df_seller = pd.read_csv(seller_file, on_bad_lines='skip')
                    df_seller['seed_id'] = seed_id
                    df_seller = df_seller.assign(**scenario_params)

                    # Identify Adversary vs Benign
                    df_seller['seller_type'] = df_seller['seller_id'].apply(
                        lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign')

                    all_seller_dfs.append(df_seller)

                    # Calculate Stats for Summary
                    # Only calculate based on the exploitation phase (post-round 30)
                    valid = df_seller[df_seller['round'] > EXPLORATION_ROUNDS]
                    if valid.empty: valid = df_seller

                    adv_sel = valid[valid['seller_type'] == 'Adversary']['selected'].mean()

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

    print(f"Loaded {len(df_sum)} valid runs for {target_defense}.")
    return df_s, df_sum


# ---------------------
# 3. PLOTTING FUNCTIONS
# ---------------------

def plot_gap_highlight_martfl(df: pd.DataFrame, output_dir: Path):
    """
    The 'Money Shot': Shows the Red Zone where the attacker overtakes the defense.
    """
    if df.empty: return
    print("Generating 'Gap Highlight' plots...")

    # Aggregate to get smooth mean lines across seeds
    group_cols = ['threat_label', 'adaptive_mode', 'round', 'seller_type']
    df_agg = df.groupby(group_cols)['selected'].mean().reset_index()

    # Pivot for plotting logic
    df_piv = df_agg.pivot_table(
        index=['threat_label', 'adaptive_mode', 'round'],
        columns='seller_type',
        values='selected'
    ).reset_index()

    if 'Adversary' not in df_piv.columns: df_piv['Adversary'] = 0.0
    if 'Benign' not in df_piv.columns: df_piv['Benign'] = 0.0

    # Smooth lines for publication quality
    df_piv['Adv_Smooth'] = df_piv.groupby(['threat_label', 'adaptive_mode'])['Adversary'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_piv['Ben_Smooth'] = df_piv.groupby(['threat_label', 'adaptive_mode'])['Benign'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    threats = df_piv['threat_label'].unique()

    for threat in threats:
        if threat == '0. Baseline': continue

        # Filter by threat
        threat_data = df_piv[df_piv['threat_label'] == threat]

        # Iterate over modes (Data Poisoning vs Grad Manipulation)
        for mode in threat_data['adaptive_mode'].unique():
            subset = threat_data[threat_data['adaptive_mode'] == mode]

            plt.figure(figsize=(10, 6))

            # 1. Attack Start Line
            plt.axvline(x=EXPLORATION_ROUNDS, color='black', linestyle='--', linewidth=2, alpha=0.7)
            plt.text(EXPLORATION_ROUNDS + 1, 0.95, "Attack Starts\n(Exploitation)", fontsize=12, fontweight='bold', va='top')

            # 2. Plot Lines
            plt.plot(subset['round'], subset['Ben_Smooth'], color=COLOR_MAP['Benign'], label='Honest Clients', alpha=0.8)
            plt.plot(subset['round'], subset['Adv_Smooth'], color=COLOR_MAP['Adversary'], label='Adaptive Attacker', linewidth=3.5)

            # 3. THE GAP FILLS

            # RED ZONE: Attack Winning (Evasion)
            plt.fill_between(
                subset['round'], subset['Ben_Smooth'], subset['Adv_Smooth'],
                where=(subset['Adv_Smooth'] > subset['Ben_Smooth']),
                interpolate=True, color=COLOR_MAP['Adversary'], alpha=0.2, hatch='//', label='Defense Evasion'
            )

            # BLUE ZONE: Defense Winning (Robustness)
            plt.fill_between(
                subset['round'], subset['Ben_Smooth'], subset['Adv_Smooth'],
                where=(subset['Ben_Smooth'] >= subset['Adv_Smooth']),
                interpolate=True, color=COLOR_MAP['Benign'], alpha=0.1, label='Defense Robustness'
            )

            # Formatting
            readable_mode = mode.replace('_', ' ').title()
            plt.title(f"MartFL Resilience Analysis\nThreat: {threat} | Strategy: {readable_mode}")
            plt.ylabel("Selection Rate")
            plt.xlabel("Communication Round")
            plt.ylim(0, 1.05)
            plt.legend(loc='lower right')

            # Save
            fname = f"gap_martfl_{threat}_{mode}".replace(" ", "").replace(".", "")
            plt.savefig(output_dir / f"{fname}.pdf", bbox_inches='tight')
            plt.close()


def plot_bar_change_martfl(df: pd.DataFrame, output_dir: Path):
    """
    Bar chart: Pre vs Post Selection Rate for the Attacker.
    """
    if df.empty: return
    print("Generating 'Pre/Post' bar charts...")

    # Label phases
    df_calc = df.copy()
    df_calc['Phase'] = df_calc['round'].apply(
        lambda x: 'Exploration' if x <= EXPLORATION_ROUNDS else 'Exploitation'
    )

    # Filter for Adversary only
    df_adv = df_calc[df_calc['seller_type'] == 'Adversary']

    # Calc means
    bar_data = df_adv.groupby(['threat_label', 'adaptive_mode', 'Phase'])['selected'].mean().reset_index()

    plt.figure(figsize=(12, 6))

    # Plot
    ax = sns.barplot(
        data=bar_data,
        x='threat_label',
        y='selected',
        hue='Phase',
        palette={'Exploration': 'silver', 'Exploitation': '#D62728'}, # Gray -> Red
        edgecolor='black',
        linewidth=1.5
    )

    # Labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.title("Adaptive Strategy Impact: MartFL")
    plt.ylabel("Adversary Selection Rate")
    plt.xlabel("")
    plt.legend(title="Attack Phase")
    plt.ylim(0, 1.1)

    plt.savefig(output_dir / "martfl_pre_post_impact.pdf", bbox_inches='tight')
    plt.close()


def plot_stealth_vs_damage_martfl(df_sum: pd.DataFrame, output_dir: Path):
    """
    Scatter: Selection Rate (Stealth) vs Final Accuracy (Damage).
    """
    if df_sum.empty: return
    print("Generating 'Stealth vs Damage' scatter...")

    # Filter out baseline to focus on attacks
    attacks = df_sum[df_sum['threat_label'] != '0. Baseline'].copy()
    if attacks.empty: return

    plt.figure(figsize=(10, 8))

    # Plot Scatter
    sns.scatterplot(
        data=attacks,
        x='adv_sel_rate', y='acc',
        hue='threat_label', style='adaptive_mode',
        palette=COLOR_MAP, s=200, alpha=0.9, edgecolor='black', linewidth=1.5
    )

    # Draw "Danger Zone" (High Stealth, any accuracy impact)
    # If selection > 0.5, the attacker effectively controls the baseline often.
    plt.axvspan(0.5, 1.0, color='red', alpha=0.05)
    plt.text(0.95, 0.05, "Critical Failure Zone\n(High Selection)", color='red',
             ha='right', transform=plt.gca().transAxes, fontweight='bold')

    plt.title("MartFL: Attack Effectiveness Summary")
    plt.xlabel("Adversary Selection Rate (Stealth)")
    plt.ylabel("Global Model Accuracy")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Attack Configuration")

    plt.savefig(output_dir / "martfl_stealth_damage_scatter.pdf", bbox_inches='tight')
    plt.close()


# --- MAIN ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- MartFL Analysis: Step 7 Adaptive ---")
    print(f"Output Directory: {output_dir.resolve()}")

    set_publication_style()

    # 1. Load Data
    df_s, df_sum = collect_all_results(BASE_RESULTS_DIR, target_defense=TARGET_DEFENSE)

    if df_s.empty:
        print(f"No results found for defense='{TARGET_DEFENSE}'. Check your paths.")
        return

    # 2. Generate Plots
    plot_gap_highlight_martfl(df_s, output_dir)
    plot_bar_change_martfl(df_s, output_dir)
    plot_stealth_vs_damage_martfl(df_sum, output_dir)

    print(f"\n✅ Analysis Complete.")

if __name__ == "__main__":
    main()