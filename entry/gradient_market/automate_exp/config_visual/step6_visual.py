import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step6_figures"


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'adv_0.3_poison_0.5_blend_alpha_0.1')
    """
    hps = {}

    # adv_rate and poison_rate are always present
    adv_match = re.search(r'adv_([0-9\.]+)', hp_folder_name)
    poison_match = re.search(r'poison_([0-9\.]+)', hp_folder_name)
    if adv_match:
        hps['adv_rate'] = float(adv_match.group(1))
    if poison_match:
        hps['poison_rate'] = float(poison_match.group(1))

    # blend_alpha is optional
    blend_match = re.search(r'blend_alpha_([0-9\.]+)', hp_folder_name)
    if blend_match:
        hps['blend_alpha'] = float(blend_match.group(1))

    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name (e.g., 'step6_adv_sybil_oracle_blend_martfl')"""
    try:
        # This regex captures the strategy name (which could have underscores)
        # and the defense name (which is the last part)
        match = re.search(r'step6_adv_sybil_(.+)_(fedavg|martfl|fltrust|skymask)', scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "strategy": match.group(1),  # e.g., 'baseline_no_sybil', 'oracle_blend'
                "defense": match.group(2),  # e.g., 'martfl'
            }
        else:
            raise ValueError("Pattern not matched")
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads key data from final_metrics.json and marketplace_report.json
    """
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        report_file = metrics_file.parent / "marketplace_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)

            sellers = report.get('seller_summaries', {}).values()
            adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
            ben_sellers = [s for s in sellers if s.get('type') == 'benign']

            run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
            run_data['benign_selection_rate'] = np.mean(
                [s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0

        return run_data

    except Exception as e:
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    # Look for 'step6_adv_sybil_*' directories
    scenario_folders = [f for f in base_path.glob("step6_adv_sybil_*") if f.is_dir()]
    if not scenario_folders:
        print(f"Error: No 'step6_adv_sybil_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts:
                    continue

                hp_folder_name = relative_parts[0]

                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({
                        **run_scenario,
                        **run_hps,
                        **run_metrics,
                        "hp_suffix": hp_folder_name
                    })
            except Exception as e:
                print(f"Error processing file {metrics_file} under scenario {scenario_name}: {e}")

    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    return df


def plot_sybil_comparison(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- 1. CONFIGURATION ---
    # "talk" context makes fonts larger (approx 1.3x default)
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    # --- 2. LOGIC: Standardize Alpha (Centroid Weight) ---
    def get_standardized_alpha_and_order(label):
        """
        Returns tuple: (Family_Order, Alpha_Centroid_Weight)
        Family 0: Baseline
        Family 1: Standard Mimic (Blind)
        Family 2: Oracle Mimic (Smart)
        """
        # --- BASELINE ---
        if label == 'baseline_no_sybil':
            return (0, 0.0)

        # --- FAMILY 1: STANDARD MIMIC ---
        if label == 'mimic':
            return (1, 0.10) # User defined base mimicry
        if label == 'knock_out':
            return (1, 0.20) # Knockout is 2x base
        if label == 'pivot':
            return (1, 1.00) # Pivot is pure replacement

        # --- FAMILY 2: ORACLE BLEND ---
        # Assumption: 'oracle_blend_0.8' means 0.8 ATTACK weight.
        # We convert to CENTROID weight to be consistent with Mimic.
        # Alpha = 1.0 - Attack_Weight
        if label.startswith('oracle_blend'):
            try:
                attack_weight = float(label.split('_')[-1])
                alpha_centroid = 1.0 - attack_weight
                return (2, round(alpha_centroid, 2))
            except:
                return (2, 0.5)

        # Fallback for unknown
        return (3, 0.0)

    # Get unique labels and sort them using the logic above
    unique_labels = defense_df['strategy_label'].unique()
    sorted_labels = sorted(unique_labels, key=get_standardized_alpha_and_order)

    # --- 3. DATA PREP ---
    metric_map = {
        'acc': 'Model Accuracy',
        'asr': 'Attack Success Rate',
        'adv_selection_rate': 'Adv. Selection Rate',
        'benign_selection_rate': 'Benign Selection Rate'
    }
    metrics_to_plot = [m for m in metric_map.keys() if m in defense_df.columns]

    plot_df = defense_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- 4. PLOTTING ---
    plt.figure(figsize=(20, 8)) # Wide figure for readability

    ax = sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels,
        palette="deep",
        edgecolor="black",
        linewidth=1.2 # Thicker borders for "Publication Quality"
    )

    # --- 5. STYLING ---
    # Use LaTeX rendering for mathematical symbols if available, otherwise standard text
    plt.title(f'Impact of Mimicry Factor ($\\alpha$) on Attack Efficacy ({defense.upper()})',
              fontsize=24, fontweight='bold', pad=20)
    plt.ylabel('Rate', fontsize=22, fontweight='bold')
    plt.xlabel('Sybil Strategy ($\\alpha$ = Centroid Weight)', fontsize=20, fontweight='bold', labelpad=15)

    # --- 6. FORMATTING LABELS ---
    def format_label(l):
        sort_key = get_standardized_alpha_and_order(l)
        family_id = sort_key[0]
        alpha_val = sort_key[1]

        if family_id == 0:
            return "Baseline"
        elif family_id == 1:
            return f"Mimic\n($\\alpha={alpha_val}$)"
        elif family_id == 2:
            return f"Oracle\n($\\alpha={alpha_val}$)"
        else:
            return l.replace('_', '\n').title()

    formatted_labels = [format_label(l) for l in sorted_labels]

    # Rotation 0 makes it horizontal (easiest to read)
    ax.set_xticklabels(formatted_labels, rotation=0, fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18)

    # --- 7. LEGEND ---
    # Place legend horizontally at the bottom
    plt.legend(
        title=None,
        fontsize=18,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.30),
        ncol=4,
        frameon=False
    )

    # Adjust layout to prevent cutting off the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save as PDF (Vector graphics for LaTeX)
    plot_file = output_dir / f"plot_sybil_standardized_{defense}.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # --- FIX 1: Filter out 'systematic_probe' ---
    df = df[df['strategy'] != 'systematic_probe'].copy()
    print(f"\nFiltered out 'systematic_probe'. Remaining runs: {len(df)}")

    # --- FIX 2: Create the strategy_label column once ---
    df['strategy_label'] = df['strategy']
    blend_rows = df['blend_alpha'].notna()
    df.loc[blend_rows, 'strategy_label'] = df.loc[blend_rows].apply(
        lambda row: f"oracle_blend_{row['blend_alpha']}", axis=1
    )

    # --- FIX 3: Generate the aggregated CSV ---
    print("\n--- Generating Aggregated Summary CSV ---")
    metrics_to_agg = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate', 'rounds']
    metrics_to_agg = [m for m in metrics_to_agg if m in df.columns]

    # Aggregate by defense and the new strategy label
    # This averages over all seeds for each strategy
    df_agg = df.groupby(['defense', 'strategy_label'])[metrics_to_agg].mean()
    df_agg = df_agg.reset_index()

    # Save to CSV
    csv_path = output_dir / "step6_sybil_summary.csv"
    df_agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✅ Saved aggregated summary to: {csv_path}")
    # --- END OF FIXES ---

    # Get all unique defenses
    defenses = df_agg['defense'].unique()

    for defense in defenses:
        # Pass the pre-aggregated and pre-labeled dataframe chunk
        plot_sybil_comparison(df_agg[df_agg['defense'] == defense].copy(), defense, output_dir)

    print("\nAnalysis complete. Check 'step6_figures' folder for plots and CSV.")


if __name__ == "__main__":
    main()
