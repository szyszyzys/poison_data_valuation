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
FIGURE_OUTPUT_DIR = "./figures/step7_adaptive_visuals"
EXPLORATION_ROUNDS = 30  # The round where the adaptive attack starts

# --- Global Style Settings ---
def set_publication_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['legend.frameon'] = True

# Consistent Colors for Step 7
COLOR_MAP = {
    'Adversary': '#D62728',   # Red
    'Benign': '#1F77B4',      # Blue
    '0. Baseline': 'gray',
    '1. Black-Box': '#FF7F0E', # Orange
    '2. Grad-Inversion': '#2CA02C', # Green
    '3. Oracle': '#9467BD'     # Purple
}

# ---------------------
# 1. PARSING LOGIC (Robust for Step 7)
# ---------------------

def parse_scenario_name(scenario_name: str) -> Dict[str, Any]:
    """
    Parses 'step7_adaptive_<THREAT>_<MODE>_<DEFENSE>_<DATASET>'
    """
    try:
        # A. Handle Baseline
        if 'baseline_no_attack' in scenario_name:
            # Format: step7_baseline_no_attack_martfl_CIFAR100
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

        # B. Handle Adaptive Attacks
        if not scenario_name.startswith('step7_adaptive_'):
            return {"defense": "unknown"}

        rest = scenario_name.replace('step7_adaptive_', '')

        # Extract Threat Model
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

        # Extract Mode (Poisoning vs Manipulation)
        adaptive_mode = "unknown"
        if rest.startswith('data_poisoning_'):
            adaptive_mode = 'data_poisoning'
            rest = rest.replace('data_poisoning_', '')
        elif rest.startswith('gradient_manipulation_'):
            adaptive_mode = 'gradient_manipulation'
            rest = rest.replace('gradient_manipulation_', '')

        # Remaining parts are Defense and Dataset
        parts = rest.split('_')
        defense = parts[0]
        dataset = parts[1] if len(parts) > 1 else "CIFAR100"

        # Label Mapping
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

def collect_all_results(base_dir: str, target_defense: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_seller_dfs, all_summary_rows = [], []
    base_path = Path(base_dir)

    # Search for step7 folders
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

                # 1. Load Seller Metrics (Crucial for Selection Rate)
                seller_file = run_dir / 'seller_metrics.csv'
                if seller_file.exists():
                    df_seller = pd.read_csv(seller_file, on_bad_lines='skip')
                    df_seller['seed_id'] = seed_id
                    df_seller = df_seller.assign(**scenario_params)
                    # Label seller type
                    df_seller['seller_type'] = df_seller['seller_id'].apply(
                        lambda x: 'Adversary' if str(x).startswith('adv_') else 'Benign')
                    all_seller_dfs.append(df_seller)

                # 2. Load Summary
                with open(final_metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Calculate averages for summary
                adv_sel, ben_sel = 0.0, 0.0
                if seller_file.exists() and not df_seller.empty:
                    # Only look at rounds after exploration
                    valid = df_seller[df_seller['round'] > EXPLORATION_ROUNDS]
                    if valid.empty: valid = df_seller

                    ben_mean = valid[valid['seller_type'] == 'Benign']['selected'].mean()
                    if pd.notna(ben_mean): ben_sel = ben_mean

                    # For baseline, proxy adversarial stats using specific benign IDs if needed
                    if scenario_params['threat_model'] == 'baseline':
                         # Just use 0 for baseline adversary
                         adv_sel = 0.0
                    else:
                        adv_mean = valid[valid['seller_type'] == 'Adversary']['selected'].mean()
                        if pd.notna(adv_mean): adv_sel = adv_mean

                all_summary_rows.append({
                    **scenario_params,
                    'seed_id': seed_id,
                    'acc': metrics.get('acc', 0),
                    'adv_sel_rate': adv_sel,
                    'ben_sel_rate': ben_sel
                })

            except Exception as e:
                print(f"Error reading {run_dir}: {e}")

    df_s = pd.concat(all_seller_dfs, ignore_index=True) if all_seller_dfs else pd.DataFrame()
    df_sum = pd.DataFrame(all_summary_rows)
    return df_s, pd.DataFrame(), df_sum


# ---------------------
# 3. PLOTTING FUNCTIONS
# ---------------------

def plot_selection_gap_highlight(df: pd.DataFrame, output_dir: Path):
    """
    Step 7 Visual: The "Gap" Plot.
    Shades Red if Attack > Benign, Blue if Benign > Attack.
    """
    if df.empty: return
    print("Generating 'Gap Highlight' plots...")

    # Aggregate over seeds to get smooth mean lines
    group_cols = ['defense', 'threat_label', 'adaptive_mode', 'round', 'seller_type']
    df_agg = df.groupby(group_cols)['selected'].mean().reset_index()

    # Pivot to columns: [round, Adversary, Benign]
    df_piv = df_agg.pivot_table(
        index=['defense', 'threat_label', 'adaptive_mode', 'round'],
        columns='seller_type',
        values='selected'
    ).reset_index()

    # Fill NaNs
    if 'Adversary' not in df_piv.columns: df_piv['Adversary'] = 0.0
    if 'Benign' not in df_piv.columns: df_piv['Benign'] = 0.0

    # Rolling average for visual smoothness
    df_piv['Adv_Smooth'] = df_piv.groupby(['defense', 'threat_label'])['Adversary'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df_piv['Ben_Smooth'] = df_piv.groupby(['defense', 'threat_label'])['Benign'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    for defense in df_piv['defense'].unique():
        def_data = df_piv[df_piv['defense'] == defense]

        for threat in def_data['threat_label'].unique():
            if threat == '0. Baseline': continue # Skip baseline for gap plot

            data = def_data[def_data['threat_label'] == threat]
            if data.empty: continue

            # We might have different adaptive modes, take the first or iterate
            for mode in data['adaptive_mode'].unique():
                subset = data[data['adaptive_mode'] == mode]

                plt.figure(figsize=(10, 6))

                # Attack Start Line
                plt.axvline(x=EXPLORATION_ROUNDS, color='black', linestyle='--', alpha=0.5)
                plt.text(EXPLORATION_ROUNDS+1, 0.95, "Attack Starts", fontsize=10)

                # Plot Lines
                plt.plot(subset['round'], subset['Ben_Smooth'], color=COLOR_MAP['Benign'], label='Honest Selection', alpha=0.8)
                plt.plot(subset['round'], subset['Adv_Smooth'], color=COLOR_MAP['Adversary'], label='Attacker Selection', linewidth=3)

                # Fill GAP
                # Red Zone: Attack Winning
                plt.fill_between(
                    subset['round'], subset['Ben_Smooth'], subset['Adv_Smooth'],
                    where=(subset['Adv_Smooth'] > subset['Ben_Smooth']),
                    color=COLOR_MAP['Adversary'], alpha=0.2, hatch='///', label='Defense Evasion'
                )

                # Blue Zone: Defense Winning
                plt.fill_between(
                    subset['round'], subset['Ben_Smooth'], subset['Adv_Smooth'],
                    where=(subset['Ben_Smooth'] >= subset['Adv_Smooth']),
                    color=COLOR_MAP['Benign'], alpha=0.1, label='Defense Robustness'
                )

                plt.title(f"Adaptive Attack Analysis: {defense.upper()}\n{threat} ({mode})")
                plt.ylabel("Selection Rate")
                plt.xlabel("Round")
                plt.ylim(0, 1.05)
                plt.legend(loc='upper right')

                fname = f"gap_{defense}_{threat}_{mode}".replace(" ", "").replace(".", "")
                plt.savefig(output_dir / f"{fname}.pdf", bbox_inches='tight')
                plt.close()

def plot_pre_post_change_bar(df: pd.DataFrame, output_dir: Path):
    """
    Step 7 Visual: Bar chart showing Adversary Selection Rate Before vs After Round 30.
    """
    if df.empty: return
    print("Generating 'Pre/Post' bar charts...")

    df_calc = df.copy()
    df_calc['Phase'] = df_calc['round'].apply(
        lambda x: 'Exploration\n(Pre-Attack)' if x <= EXPLORATION_ROUNDS else 'Exploitation\n(Post-Attack)'
    )

    # We only care about Adversary success
    df_adv = df_calc[df_calc['seller_type'] == 'Adversary']
    if df_adv.empty: return

    # Calculate means
    bar_data = df_adv.groupby(['defense', 'threat_label', 'adaptive_mode', 'Phase'])['selected'].mean().reset_index()

    for defense in bar_data['defense'].unique():
        subset = bar_data[bar_data['defense'] == defense]
        if subset.empty: continue

        plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            data=subset,
            x='threat_label',
            y='selected',
            hue='Phase',
            palette={'Exploration\n(Pre-Attack)': 'silver', 'Exploitation\n(Post-Attack)': '#D62728'},
            edgecolor='black'
        )

        # Add labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)

        plt.title(f"Impact of Adaptive Strategy: {defense.upper()}")
        plt.ylabel("Avg Adversary Selection Rate")
        plt.xlabel("")
        plt.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc='upper left')

        fname = f"bar_change_{defense}".replace(" ", "")
        plt.savefig(output_dir / f"{fname}.pdf", bbox_inches='tight')
        plt.close()

def plot_stealth_vs_damage(df_sum: pd.DataFrame, output_dir: Path):
    """
    Step 7 Visual: Trade-off Scatter.
    """
    if df_sum.empty: return
    print("Generating 'Stealth vs Damage' scatter...")

    # Filter out baseline
    attacks = df_sum[df_sum['threat_label'] != '0. Baseline'].copy()
    if attacks.empty: return

    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=attacks,
        x='adv_sel_rate', y='acc',
        hue='threat_label', style='defense',
        palette='tab10', s=150, alpha=0.9, edgecolor='black'
    )

    # Danger Zone
    plt.axvspan(0.5, 1.0, color='red', alpha=0.05)
    plt.text(0.95, 0.05, "Critical Failure Zone\n(High Selection)", color='red', ha='right', transform=plt.gca().transAxes)

    plt.title("Adaptive Attack Effectiveness Summary")
    plt.xlabel("Adversary Selection Rate (Stealth)")
    plt.ylabel("Global Model Accuracy")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(output_dir / "scatter_stealth_damage.pdf", bbox_inches='tight')
    plt.close()

# --- MAIN ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving Step 7 visuals to: {output_dir.resolve()}")

    set_publication_style()

    # Load all defenses to compare
    df_s, _, df_sum = collect_all_results(BASE_RESULTS_DIR, target_defense=None)

    if df_s.empty:
        print("No Step 7 data found in ./results. Make sure folders start with 'step7_'.")
        return

    # Generate Plots
    plot_selection_gap_highlight(df_s, output_dir)
    plot_pre_post_change_bar(df_s, output_dir)
    plot_stealth_vs_damage(df_sum, output_dir)

    print(f"\n✅ Step 7 Analysis Complete.")

if __name__ == "__main__":
    main()