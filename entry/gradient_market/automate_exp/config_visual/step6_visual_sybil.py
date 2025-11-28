import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step6_figures"

def set_publication_style():
    """
    Sets the 'Big & Bold' professional style globally.
    """
    sns.set_theme(style="whitegrid")
    # Global Scaling
    sns.set_context("paper", font_scale=2.0)

    # Specific Parameter Overrides
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',              # Bold all text
        'axes.labelweight': 'bold',         # Bold axis labels
        'axes.titleweight': 'bold',         # Bold titles

        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,

        'legend.fontsize': 18,
        'legend.title_fontsize': 20,

        'axes.linewidth': 2.5,              # Thicker axis borders
        'axes.edgecolor': '#333333',
        'lines.linewidth': 3.0,
        'figure.figsize': (20, 9)           # Default large size
    })

# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """Parses the HP suffix folder name."""
    hps = {}
    adv_match = re.search(r'adv_([0-9\.]+)', hp_folder_name)
    poison_match = re.search(r'poison_([0-9\.]+)', hp_folder_name)
    if adv_match: hps['adv_rate'] = float(adv_match.group(1))
    if poison_match: hps['poison_rate'] = float(poison_match.group(1))

    blend_match = re.search(r'blend_alpha_([0-9\.]+)', hp_folder_name)
    if blend_match: hps['blend_alpha'] = float(blend_match.group(1))

    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """Parses the base scenario name."""
    try:
        match = re.search(r'step6_adv_sybil_(.+)_(fedavg|martfl|fltrust|skymask|skymask_small)', scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "strategy": match.group(1),
                "defense": match.group(2),
            }
        else:
            return {"scenario": scenario_name, "defense": "unknown"}
    except Exception:
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """Loads key data from final_metrics.json and marketplace_report.json"""
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
            run_data['benign_selection_rate'] = np.mean([s['selection_rate'] for s in ben_sellers]) if ben_sellers else 0.0

        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)

    scenario_folders = [f for f in base_path.glob("step6_adv_sybil_*") if f.is_dir()]
    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if run_scenario.get("defense") == "unknown": continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

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
            except Exception:
                continue

    df = pd.DataFrame(all_runs)

    if not df.empty and 'defense' in df.columns:
        # Basic standardization for defense names
        name_map = {'skymask_small': 'SkyMask', 'skymask': 'SkyMask',
                    'fedavg': 'FedAvg', 'fltrust': 'FLTrust', 'martfl': 'MARTFL'}
        df['defense'] = df['defense'].map(lambda x: name_map.get(x, x.title()))

    return df

# ==========================================
# 3. PLOTTING LOGIC
# ==========================================

def plot_sybil_comparison(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- LOGIC: Standardize Alpha ---
    def get_standardized_alpha_and_order(label):
        if label == 'baseline_no_sybil': return (0, 0.0)
        if label == 'mimic': return (1, 0.10) # Set baseline mimicry alpha here
        if label == 'knock_out': return (1, 0.20)
        if label == 'pivot': return (1, 1.00)
        if label.startswith('oracle_blend'):
            try:
                attack_weight = float(label.split('_')[-1])
                alpha_centroid = 1.0 - attack_weight
                return (2, round(alpha_centroid, 2))
            except:
                return (2, 0.5)
        return (3, 0.0)

    unique_labels = defense_df['strategy_label'].unique()
    sorted_labels = sorted(unique_labels, key=get_standardized_alpha_and_order)

    # --- DATA PREP ---
    metric_map = {
        'acc': 'Model Accuracy',
        'asr': 'Attack Success Rate',
        'adv_selection_rate': 'Adv. Selection Rate',
        'benign_selection_rate': 'Benign Selection Rate'
    }
    metrics_to_plot = [m for m in metric_map.keys() if m in defense_df.columns]
    plot_df = defense_df.copy()
    for m in metrics_to_plot:
        plot_df[m] = plot_df[m] * 100

    plot_df = plot_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- PLOTTING ---
    plt.figure(figsize=(22, 9))

    ax = sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels,
        palette="deep",
        edgecolor="black",
        linewidth=2.0
    )

    plt.ylabel('Rate (%)')
    plt.xlabel('Sybil Strategy (Mimicry Factor)')
    plt.title(f"Sybil Attack Effectiveness vs {defense}", pad=20)

    # --- FORMATTING TICKS ---
    def format_tick_label(l):
        sort_key = get_standardized_alpha_and_order(l)
        family_id = sort_key[0]
        alpha_val = sort_key[1]

        if family_id == 0:
            return "Baseline"
        elif family_id == 1:
            # UPDATE: Now showing alpha for Mimic too
            return f"Mimic\n($\\alpha={alpha_val}$)"
        elif family_id == 2:
            return f"Oracle\n($\\alpha={alpha_val}$)"
        else:
            return l.replace('_', '\n').title()

    formatted_labels = [format_tick_label(l) for l in sorted_labels]

    ax.set_xticklabels(formatted_labels, rotation=0)
    ax.set_ylim(0, 105)

    plt.legend(
        title=None,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    safe_defense = re.sub(r'[^\w]', '', defense)
    plot_file = output_dir / f"plot_sybil_standardized_{safe_defense}.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')


def plot_sybil_comparison_new(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- 1. FILTERING LOGIC (Show Only Best Attacks) ---
    # We want to show:
    # 1. Baseline (No Attack)
    # 2. Pivot (Extreme Baseline)
    # 3. Best Mimic (Lowest Alpha)
    # 4. Best Oracle (Lowest Alpha / Highest Selection)

    strategies_to_keep = ['baseline_no_sybil', 'pivot']

    # Find Best Mimic (Lowest Alpha usually means highest stealth/selection)
    # Assuming mimic doesn't have alpha in label, or if it does:
    mimic_rows = defense_df[defense_df['strategy_label'].str.contains('mimic')]
    if not mimic_rows.empty:
        # If you have different alphas for mimic, sort and pick best.
        # If mimic is just 'mimic', keep it.
        strategies_to_keep.append('mimic')

    # Find Best Oracle Blend (Lowest Alpha = Highest Attack Weight = Most Aggressive but Successful)
    # OR Lowest Alpha that has > 50% selection rate.
    # Let's pick the one with the highest 'adv_selection_rate'
    oracle_rows = defense_df[defense_df['strategy_label'].str.contains('oracle_blend')]
    if not oracle_rows.empty:
        # Sort by selection rate descending
        best_oracle = oracle_rows.sort_values('adv_selection_rate', ascending=False).iloc[0]['strategy_label']
        strategies_to_keep.append(best_oracle)

    # Filter the DataFrame
    filtered_df = defense_df[defense_df['strategy_label'].isin(strategies_to_keep)].copy()

    # --- 2. ORDERING LOGIC ---
    def get_order(label):
        if 'baseline' in label: return 0
        if 'mimic' in label: return 1
        if 'oracle' in label: return 2
        if 'pivot' in label: return 3
        return 4

    sorted_labels = sorted(strategies_to_keep, key=get_order)

    # --- 3. DATA PREP (Same as before) ---
    metric_map = {
        'adv_selection_rate': 'Attacker Selection Rate',
        'benign_selection_rate': 'Honest Selection Rate'
        # Removed Accuracy/ASR to make figure smaller/cleaner as requested
    }
    metrics_to_plot = list(metric_map.keys())

    plot_df = filtered_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Value'] = plot_df['Value'] * 100
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- 4. PLOTTING (Compact Size) ---
    # Made figure smaller (10x6 instead of 22x9)
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels,
        palette={'Attacker Selection Rate': '#D62728', 'Honest Selection Rate': '#1F77B4'}, # Red/Blue
        edgecolor="black",
        linewidth=1.5
    )

    plt.ylabel('Selection Rate (%)')
    plt.xlabel('') # Remove redundant label
    plt.title(f"Sybil Market Capture: {defense}", pad=15)
    plt.ylim(0, 105)

    # --- 5. CLEAN TICKS ---
    def format_tick(l):
        if 'baseline' in l: return "No Attack"
        if 'mimic' in l: return "Mimicry\n(Noise)"
        if 'oracle' in l:
            # Extract alpha if present
            try:
                alpha = l.split('_')[-1]
                return f"Buyer Collusion\n($\\alpha={alpha}$)"
            except: return "Buyer Collusion"
        if 'pivot' in l: return "Pivot"
        return l

    ax.set_xticklabels([format_tick(l) for l in sorted_labels])

    # Legend Inside to save space
    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.tight_layout()

    safe_defense = re.sub(r'[^\w]', '', defense)
    # Save as "compact" version
    plot_file = output_dir / f"plot_sybil_compact_{safe_defense}.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved compact plot: {plot_file}")
    plt.clf()
    plt.close('all')
def plot_sybil_comparison_compact(defense_df: pd.DataFrame, defense: str, output_dir: Path):
    if defense_df.empty:
        print(f"No data for defense: {defense}")
        return

    print(f"\n--- Plotting Sybil Effectiveness for: {defense} ---")

    # --- 1. STRICT FILTERING LOGIC ---
    # You asked for:
    # 1. Baseline
    # 2. Mimic (Alpha 0.1) -> In your data, this is labeled just "mimic"
    # 3. Oracle (Alpha 0.1) -> Labeled "oracle_blend_0.1"

    target_strategies = [
        'baseline_no_sybil',
        'mimic',
        'oracle_blend_0.1'
    ]

    # Filter the dataframe to keep ONLY these rows
    filtered_df = defense_df[defense_df['strategy_label'].isin(target_strategies)].copy()

    if filtered_df.empty:
        print("Warning: No matching strategies found for this defense.")
        return

    # --- 2. ORDERING LOGIC ---
    # Ensure they appear in the exact order you listed
    def get_order(label):
        if 'baseline' in label: return 0
        if 'mimic' in label: return 1
        if 'oracle' in label: return 2
        return 99

    sorted_labels = sorted(filtered_df['strategy_label'].unique(), key=get_order)

    # --- 3. DATA PREP ---
    # We only show Selection Rate as requested previously
    metric_map = {
        'adv_selection_rate': 'Attacker Selection Rate',
        'benign_selection_rate': 'Honest Selection Rate'
    }
    metrics_to_plot = list(metric_map.keys())

    plot_df = filtered_df.melt(
        id_vars=['strategy_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    plot_df['Value'] = plot_df['Value'] * 100
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)

    # --- 4. PLOTTING ---
    plt.figure(figsize=(8, 6)) # Smaller figure for just 3 bars

    ax = sns.barplot(
        data=plot_df,
        x='strategy_label',
        y='Value',
        hue='Metric',
        order=sorted_labels,
        palette={'Attacker Selection Rate': '#D62728', 'Honest Selection Rate': '#1F77B4'},
        edgecolor="black",
        linewidth=1.5
    )

    plt.ylabel('Selection Rate (%)')
    plt.xlabel('')
    plt.title(f"Sybil Attack Impact: {defense}", pad=15)
    plt.ylim(0, 105)

    # --- 5. CUSTOM TICKS ---
    def format_tick(l):
        if 'baseline' in l: return "No Attack"
        if 'mimic' in l: return "Mimicry\n(Noise)"
        if 'oracle' in l: return "Buyer Collusion\n(Alpha=0.1)"
        return l

    ax.set_xticklabels([format_tick(l) for l in sorted_labels])

    # Legend
    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.tight_layout()

    safe_defense = re.sub(r'[^\w]', '', defense)
    plot_file = output_dir / f"plot_sybil_selected_{safe_defense}.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved selected plot: {plot_file}")
    plt.clf()
    plt.close('all')
def main():
    set_publication_style()

    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # Filter systematic_probe
    df = df[df['strategy'] != 'systematic_probe'].copy()

    # Label Creation
    df['strategy_label'] = df['strategy']
    blend_rows = df['blend_alpha'].notna()
    df.loc[blend_rows, 'strategy_label'] = df.loc[blend_rows].apply(
        lambda row: f"oracle_blend_{row['blend_alpha']}", axis=1
    )

    # Aggregate
    metrics_to_agg = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate', 'rounds']
    metrics_to_agg = [m for m in metrics_to_agg if m in df.columns]
    df_agg = df.groupby(['defense', 'strategy_label'])[metrics_to_agg].mean().reset_index()

    csv_path = output_dir / "step6_sybil_summary.csv"
    df_agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✅ Saved aggregated summary to: {csv_path}")

    # Plot
    defenses = df_agg['defense'].unique()
    for defense in defenses:
        # plot_sybil_comparison(df_agg[df_agg['defense'] == defense].copy(), defense, output_dir)
        plot_sybil_comparison_compact(df_agg[df_agg['defense'] == defense].copy(), defense, output_dir)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()