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
FIGURE_OUTPUT_DIR = "./figures/step2.5_figures"

# Define the relative 'usability' threshold
RELATIVE_ACC_THRESHOLD = 0.90


# ---------------------

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    Parses the HP suffix folder name (e.g., 'opt_Adam_lr_0.001_epochs_2')
    """
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)

    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    else:
        pass  # Silence warnings for cleaner output
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Parses the base scenario name (e.g., 'step2.5_find_hps_martfl_image_CIFAR10')
    """
    try:
        # UPDATE 1: Added 'skymask_small' to the regex group so it matches successfully
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask_small|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)

        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "modality": match.group(2),
                "dataset": match.group(3),
            }
        else:
            raise ValueError(f"Pattern not matched for: {scenario_name}")

    except Exception as e:
        # print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    """
    Loads metrics from final_metrics.json (acc, rounds)
    AND calculates both benign and adversary selection rates from marketplace_report.json.
    """
    run_data = {}
    try:
        # 1. Load basic metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

        # 2. Load Selection Rates
        report_file = metrics_file.parent / "marketplace_report.json"

        if not report_file.exists():
            run_data['benign_selection_rate'] = np.nan
            run_data['adv_selection_rate'] = np.nan
            return run_data

        with open(report_file, 'r') as f:
            report = json.load(f)

        sellers = report.get('seller_summaries', {}).values()
        adv_sellers = [s for s in sellers if s.get('type') == 'adversary']
        ben_sellers = [s for s in sellers if s.get('type') == 'benign']

        run_data['adv_selection_rate'] = np.mean(
            [s['selection_rate'] for s in adv_sellers]
        ) if adv_sellers else 0.0

        run_data['benign_selection_rate'] = np.mean(
            [s['selection_rate'] for s in ben_sellers]
        ) if ben_sellers else np.nan

        return run_data
    except Exception as e:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    """
    Walks the results directory, aggregates all run data,
    filters out '_nolocalclip', and SWAPS 'skymask_small' in place of 'skymask'.
    """
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [
        f for f in base_path.glob("step2.5_find_hps_*")
        if f.is_dir() and not f.name.endswith("_nolocalclip")
    ]

    if not scenario_folders:
        print(f"Error: No 'step2.5_find_hps_*' directories (excluding '_nolocalclip') found.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

    for scenario_path in scenario_folders:
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)

        # Skip if regex failed
        if "defense" not in run_scenario:
            continue

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
                    })
            except Exception as e:
                continue

    if not all_runs:
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # --- UPDATE 2: SWAP LOGIC ---
    # If we have 'skymask_small' data, we want to discard the old 'skymask'
    # and rename 'skymask_small' to 'skymask' so it shows up correctly in plots.
    if 'skymask_small' in df['defense'].unique():
        print("ℹ️  Found 'skymask_small' results. Replacing old 'skymask' results with them.")

        # 1. Drop rows that are exactly 'skymask' (the old/failed ones)
        df = df[df['defense'] != 'skymask']

        # 2. Rename 'skymask_small' to 'skymask' (so the label is correct in plots)
        df.loc[df['defense'] == 'skymask_small', 'defense'] = 'skymask'
    # ----------------------------

    print("Calculating thresholds...")
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    return df


def style_axis(ax, title, xlabel, ylabel):
    """
    Helper to apply the consistent "Publication Quality" typography.
    """
    # Title
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)

    # Labels
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=22, fontweight='bold', labelpad=15)

    # Ticks
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18)


def plot_platform_usability_with_selection(df: pd.DataFrame, output_dir: Path):
    """
    Plots metrics with Hybrid Accuracy Logic (Mean for Usable, Max for Failed).
    Adds hatching to failed bars and a threshold line.
    """
    print("\n--- Plotting Platform Metrics (High-Res Style) ---")

    if df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    # --- Metric Calculations ---

    # 1. Usability Rate
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100
    df_usability['Metric'] = 'Usability Rate (%)'

    # 2. Accuracy (UPDATED LOGIC)
    def calc_best_acc(group):
        # Check if there are ANY usable runs
        usable = group[group['platform_usable'] == True]
        if not usable.empty:
            return usable['acc'].mean()  # Return Mean of Stable Runs
        else:
            return group['acc'].max()  # Return Max of Unstable Runs (Best Effort)

    df_perf = df.groupby(['defense', 'dataset']).apply(calc_best_acc).reset_index(name='acc')
    df_perf['Value'] = df_perf['acc'] * 100
    df_perf['Metric'] = 'Avg. Usable Accuracy (%)'

    # 3. Speed (Keep strict filtering for rounds context)
    df_speed = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['rounds'].mean().reset_index()
    df_speed['Value'] = df_speed['rounds']
    df_speed['Metric'] = 'Avg. Usable Rounds'

    # 4. Stability
    df_speed_stability = df.groupby(['defense', 'dataset'])['rounds'].std().reset_index()
    df_speed_stability['Value'] = df_speed_stability['rounds']
    df_speed_stability['Metric'] = 'Rounds Instability (Std)'

    df_metrics_1_4 = pd.concat([df_usability, df_perf, df_speed, df_speed_stability], ignore_index=True)

    # 5. Selection Metrics
    df_selection_raw = df.groupby(['defense', 'dataset'])[
        ['benign_selection_rate', 'adv_selection_rate']].mean().reset_index()

    df_selection_raw = df_selection_raw.rename(columns={
        'benign_selection_rate': 'Benign',
        'adv_selection_rate': 'Adversary'
    })

    df_selection_melted = df_selection_raw.melt(
        id_vars=['defense', 'dataset'],
        value_vars=['Benign', 'Adversary'],
        var_name='Seller Type',
        value_name='Selection Rate'
    )
    df_selection_melted['Selection Rate'] *= 100

    # --- PLOTTING LOOP ---
    all_datasets = df_metrics_1_4['dataset'].unique()

    def format_defense(d):
        d_map = {'fedavg': 'FedAvg', 'fltrust': 'FLTrust', 'martfl': 'MARTFL', 'skymask': 'SkyMask'}
        return d_map.get(d.lower(), d.title())

    for dataset in all_datasets:
        if dataset in ['CIFAR10', 'CIFAR100']:
            defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
        else:
            defense_order = ['fedavg', 'fltrust', 'martfl']

        formatted_labels = [format_defense(d) for d in defense_order]

        # === A. Plot Metrics 1-4 ===
        for metric in df_metrics_1_4['Metric'].unique():
            plot_df = df_metrics_1_4[
                (df_metrics_1_4['dataset'] == dataset) &
                (df_metrics_1_4['Metric'] == metric)
                ]
            plot_df = plot_df[plot_df['defense'].isin(defense_order)]

            if plot_df.empty: continue

            plt.figure(figsize=(10, 8))
            ax = sns.barplot(
                data=plot_df,
                x='defense',
                y='Value',
                order=defense_order,
                palette='viridis',
                edgecolor="black",
                linewidth=1.2
            )

            # --- SPECIAL HANDLING FOR ACCURACY (Hatching & Thresholds) ---
            if "Accuracy" in metric:
                # 1. Add Threshold Line
                # Find the max acc for this dataset across all runs to recalc the threshold
                raw_max_acc = df[df['dataset'] == dataset]['acc'].max()
                threshold_val = (raw_max_acc * 0.90) * 100  # Convert to percentage

                ax.axhline(threshold_val, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
                # Place label near the right edge
                ax.text(len(defense_order) - 0.5, threshold_val + 1, 'Usability Threshold',
                        color='red', fontsize=14, fontweight='bold', va='bottom', ha='right')

                # 2. Apply Hatching to "Failed" Defenses
                for i, defense in enumerate(defense_order):
                    # Check if this defense had 0% usability
                    usability_val = df_metrics_1_4[
                        (df_metrics_1_4['dataset'] == dataset) &
                        (df_metrics_1_4['defense'] == defense) &
                        (df_metrics_1_4['Metric'] == 'Usability Rate (%)')
                        ]['Value']

                    if not usability_val.empty and usability_val.values[0] == 0:
                        # Apply stripes to the bar
                        bar = ax.patches[i]
                        bar.set_hatch('///')
                        bar.set_edgecolor('black')

            # Labels & Style
            y_label = "Value"
            if "%" in metric:
                y_label = "Percentage (%)"
            elif "Rounds" in metric:
                y_label = "Rounds"

            style_axis(ax, f"{metric} ({dataset})", "Defense Strategy", y_label)
            ax.set_xticklabels(formatted_labels)

            # Annotations
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.1f}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom',
                                xytext=(0, 5), textcoords='offset points',
                                fontsize=16, fontweight='bold')

            plt.tight_layout()
            safe_metric = re.sub(r'[^\w]', '', metric.split(' ')[0]) + "_" + re.sub(r'[^\w]', '', metric.split(' ')[-1])
            plt.savefig(output_dir / f"plot_{dataset}_{safe_metric}.pdf", bbox_inches='tight', format='pdf', dpi=300)
            plt.close('all')

        # === B. Plot Selection Rates (Unchanged) ===
        sel_df = df_selection_melted[
            (df_selection_melted['dataset'] == dataset) &
            (df_selection_melted['defense'].isin(defense_order))
            ]

        if not sel_df.empty:
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(
                data=sel_df,
                x='defense',
                y='Selection Rate',
                hue='Seller Type',
                order=defense_order,
                palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'},
                edgecolor="black",
                linewidth=1.2
            )
            style_axis(ax, f"Avg. Selection Rates ({dataset})", "Defense Strategy", "Selection Rate (%)")
            ax.set_xticklabels(formatted_labels)
            ax.set_ylim(0, 105)

            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f'{h:.1f}', (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                                fontsize=14, fontweight='bold')

            plt.legend(title=None, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25)
            plt.savefig(output_dir / f"plot_{dataset}_Selection_Rates.pdf", format='pdf', dpi=300)
            plt.close('all')
            print(f"  Saved metrics for {dataset}")


def plot_composite_row(df: pd.DataFrame, output_dir: Path):
    """
    4-in-1 Composite Plot with updated Accuracy logic (Mean vs Max).
    """
    print("\n--- Plotting Composite Row (4-in-1) ---")
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.4)

    dataset = "CIFAR100"  # Or parameterize this
    subset = df[df['dataset'] == dataset]

    if subset.empty:
        return

    defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
    formatted_labels = ['FedAvg', 'FLTrust', 'MARTFL', 'SkyMask']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

    # --- DATA PREP ---

    # 1. Usability
    d1 = subset.groupby('defense')['platform_usable'].mean().reindex(defense_order).reset_index()
    d1['Value'] = d1['platform_usable'] * 100

    # 2. Accuracy (UPDATED LOGIC)
    def calc_best_acc_composite(group):
        usable = group[group['platform_usable'] == True]
        if not usable.empty:
            return usable['acc'].mean()
        return group['acc'].max()

    d2 = subset.groupby('defense').apply(calc_best_acc_composite).reindex(defense_order).reset_index(name='acc')
    d2['Value'] = d2['acc'] * 100

    # 3. Rounds
    d3 = subset[subset['platform_usable'] == True].groupby('defense')['rounds'].mean().reindex(
        defense_order).reset_index()
    d3['Value'] = d3['rounds']

    # 4. Selection
    d4 = subset.groupby('defense')[['benign_selection_rate', 'adv_selection_rate']].mean().reindex(
        defense_order).reset_index()
    d4 = d4.melt(id_vars='defense', var_name='Type', value_name='Rate')
    d4['Rate'] = d4['Rate'] * 100
    d4['Type'] = d4['Type'].replace({'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})

    # --- PLOTTING ---

    # Plot 1: Usability
    sns.barplot(ax=axes[0], data=d1, x='defense', y='Value', order=defense_order, palette='viridis', edgecolor='black')
    axes[0].set_title("Usability Rate (%)", fontweight='bold')
    axes[0].set_ylabel("Percentage", fontweight='bold')
    axes[0].set_ylim(0, 105)

    # Plot 2: Accuracy
    sns.barplot(ax=axes[1], data=d2, x='defense', y='Value', order=defense_order, palette='viridis', edgecolor='black')
    axes[1].set_title("Avg. Usable Acc (%)", fontweight='bold')
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 105)

    # ADD HATCHING to Composite Accuracy Plot
    for i, defense in enumerate(defense_order):
        usability = d1[d1['defense'] == defense]['Value'].values[0]
        if usability == 0:
            axes[1].patches[i].set_hatch('///')
            axes[1].patches[i].set_edgecolor('black')

    # Plot 3: Rounds
    sns.barplot(ax=axes[2], data=d3, x='defense', y='Value', order=defense_order, palette='viridis', edgecolor='black')
    axes[2].set_title("Avg. Cost (Rounds)", fontweight='bold')
    axes[2].set_ylabel("Rounds", fontweight='bold')

    # Plot 4: Selection
    sns.barplot(ax=axes[3], data=d4, x='defense', y='Rate', hue='Type', order=defense_order,
                palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor='black')
    axes[3].set_title("Avg. Selection Rates", fontweight='bold')
    axes[3].set_ylabel("")
    axes[3].set_ylim(0, 105)
    axes[3].legend(title=None, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    # Common Styling
    for ax in axes:
        ax.set_xticklabels(formatted_labels, fontsize=14, fontweight='bold')
        ax.set_xlabel("")
        ax.grid(axis='y', alpha=0.5)
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}', (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 2),
                            textcoords='offset points')

    outfile = output_dir / f"plot_row_combined_{dataset}.pdf"
    plt.savefig(outfile, bbox_inches='tight', format='pdf', dpi=300)
    print(f"Saved composite row to: {outfile}")


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    plot_platform_usability_with_selection(df, output_dir)
    plot_composite_row(df, output_dir)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
