import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step2.5_figures_flexible"
RELATIVE_ACC_THRESHOLD = 0.90

# 1. DEFENSE LABELS: Map your folder names (keys) to Plot Titles (values)
#    If your folder is named 'skymask_lite', add 'skymask_lite': 'SkyMask (Small)' here.
DEFENSE_LABELS = {
    'fedavg': 'FedAvg',
    'fltrust': 'FLTrust',
    'martfl': 'MARTFL',
    'skymask': 'SkyMask',
}

# 2. PLOT ORDER & VISIBILITY
#    Only defenses listed here will be plotted.
#    Comment out 'skymask_small' to hide it.
#    Reorder this list to change the bar order.
DEFENSE_ORDER = [
    'fedavg',
    'fltrust',
    'martfl',
    'skymask',
]


# ==========================================

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    pattern = r'opt_(\w+)_lr_([0-9\.]+)_epochs_([0-9]+)'
    match = re.search(pattern, hp_folder_name)
    if match:
        hps['optimizer'] = match.group(1)
        hps['learning_rate'] = float(match.group(2))
        hps['local_epochs'] = int(match.group(3))
    return hps


def save_utility_csv(df: pd.DataFrame, output_dir: Path):
    print("\n--- Generating Utility Breakdown CSV ---")

    # We calculate the stats that go into the Utility Rate
    summary = df.groupby(['dataset', 'defense']).agg(
        total_runs=('acc', 'count'),  # Denominator
        usable_runs=('platform_usable', 'sum'),  # Numerator
        utility_rate=('platform_usable', 'mean'),  # Result
        avg_acc=('acc', 'mean'),
        max_acc=('acc', 'max'),
        threshold_used=('platform_usable_threshold', 'mean')
    ).reset_index()

    # Add explicit percentage column for readability
    summary['utility_rate_percent'] = summary['utility_rate'] * 100

    # Sort for cleaner CSV
    summary = summary.sort_values(by=['dataset', 'utility_rate_percent'], ascending=[True, False])

    # Save to file
    csv_path = output_dir / "utility_rate_breakdown.csv"
    summary.to_csv(csv_path, index=False)

    print(f"✅ Saved CSV to: {csv_path}")
    print("Preview of calculation:")
    print(summary[['dataset', 'defense', 'total_runs', 'usable_runs', 'utility_rate_percent']].head(10).to_string())


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    Universal Parser: Captures ANY defense string between prefix and modality.
    """
    try:
        # Regex explanation:
        # step2\.5_find_hps_  -> Prefix
        # (?P<defense>.+?)    -> Capture Defense (Non-greedy, takes anything until next underscore)
        # _                   -> Separator
        # (?P<modality>image|text|tabular) -> Modality
        pattern = r'step2\.5_find_hps_(?P<defense>.+?)_(?P<modality>image|text|tabular)_(?P<dataset>.+)'
        match = re.search(pattern, scenario_name)

        if match:
            return match.groupdict()
        else:
            return {}
    except Exception as e:
        return {}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['rounds'] = metrics.get('completed_rounds', 0)

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

        run_data['adv_selection_rate'] = np.mean([s['selection_rate'] for s in adv_sellers]) if adv_sellers else 0.0
        run_data['benign_selection_rate'] = np.mean(
            [s['selection_rate'] for s in ben_sellers]) if ben_sellers else np.nan
        return run_data
    except Exception:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if
                        f.is_dir() and not f.name.endswith("_nolocalclip")]

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        if not run_scenario: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            try:
                relative_parts = metrics_file.parent.relative_to(scenario_path).parts
                if not relative_parts: continue

                hp_folder_name = relative_parts[0]
                run_hps = parse_hp_suffix(hp_folder_name)
                run_metrics = load_run_data(metrics_file)

                if run_metrics:
                    all_runs.append({**run_scenario, **run_hps, **run_metrics})
            except Exception:
                continue

    if not all_runs:
        print("No run data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # --- Filtering Logic ---
    # Only keep defenses present in the User's DEFENSE_ORDER list
    available_defenses = set(df['defense'].unique())
    requested_defenses = set(DEFENSE_ORDER)

    # Check for missing config
    missing_in_config = available_defenses - requested_defenses
    if missing_in_config:
        print(
            f"⚠️  Note: The following defenses were found in data but are NOT in DEFENSE_ORDER and will be skipped: {missing_in_config}")

    df = df[df['defense'].isin(DEFENSE_ORDER)].copy()

    if df.empty:
        print("❌ Error: No data matched your DEFENSE_ORDER configuration.")
        return pd.DataFrame()

    print("Calculating thresholds...")
    dataset_max_acc = df.groupby('dataset')['acc'].max().to_dict()
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc)
    df['platform_usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
    df['platform_usable'] = (df['acc'] >= df['platform_usable_threshold'])

    print(f"✅ Data loaded for: {df['defense'].unique()}")
    return df


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=22, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18)


def get_formatted_labels(defense_list):
    """Uses the User Config to format labels"""
    return [DEFENSE_LABELS.get(d, d) for d in defense_list]


def plot_platform_usability_with_selection(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Platform Metrics ---")
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    # Metric Prep
    df_usability = df.groupby(['defense', 'dataset'])['platform_usable'].mean().reset_index()
    df_usability['Value'] = df_usability['platform_usable'] * 100
    df_usability['Metric'] = 'Usability Rate (%)'

    def calc_best_acc(group):
        usable = group[group['platform_usable'] == True]
        return usable['acc'].mean() if not usable.empty else group['acc'].max()

    df_perf = df.groupby(['defense', 'dataset']).apply(calc_best_acc).reset_index(name='acc')
    df_perf['Value'] = df_perf['acc'] * 100
    df_perf['Metric'] = 'Avg. Usable Accuracy (%)'

    df_speed = df[df['platform_usable'] == True].groupby(['defense', 'dataset'])['rounds'].mean().reset_index()
    df_speed['Value'] = df_speed['rounds']
    df_speed['Metric'] = 'Avg. Usable Rounds'

    df_speed_stability = df.groupby(['defense', 'dataset'])['rounds'].std().reset_index()
    df_speed_stability['Value'] = df_speed_stability['rounds']
    df_speed_stability['Metric'] = 'Rounds Instability (Std)'

    df_metrics = pd.concat([df_usability, df_perf, df_speed, df_speed_stability], ignore_index=True)

    # Selection Prep
    df_sel = df.groupby(['defense', 'dataset'])[['benign_selection_rate', 'adv_selection_rate']].mean().reset_index()
    df_sel = df_sel.rename(columns={'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})
    df_sel_melt = df_sel.melt(id_vars=['defense', 'dataset'], value_vars=['Benign', 'Adversary'], var_name='Type',
                              value_name='Rate')
    df_sel_melt['Rate'] *= 100

    all_datasets = df['dataset'].unique()

    for dataset in all_datasets:
        # Filter Order based on what exists for this dataset AND is in User Config
        ds_defenses = df[df['dataset'] == dataset]['defense'].unique()
        # Sort by the user's preferred order
        current_order = [d for d in DEFENSE_ORDER if d in ds_defenses]

        if not current_order: continue

        labels = get_formatted_labels(current_order)

        # Plot 1-4 Metrics
        for metric in df_metrics['Metric'].unique():
            plot_df = df_metrics[(df_metrics['dataset'] == dataset) & (df_metrics['Metric'] == metric) & (
                df_metrics['defense'].isin(current_order))]
            if plot_df.empty: continue

            plt.figure(figsize=(12, 8))
            ax = sns.barplot(data=plot_df, x='defense', y='Value', order=current_order, palette='viridis',
                             edgecolor="black", linewidth=1.2)

            # Thresholds
            if "Accuracy" in metric:
                raw_max = df[df['dataset'] == dataset]['acc'].max()
                threshold_val = raw_max * 0.90 * 100
                ax.axhline(threshold_val, color='red', linestyle='--', linewidth=2.5, alpha=0.7)

                # Hatching
                for i, defense in enumerate(current_order):
                    usab = df_metrics[(df_metrics['dataset'] == dataset) & (df_metrics['defense'] == defense) & (
                            df_metrics['Metric'] == 'Usability Rate (%)')]['Value']
                    if not usab.empty and usab.values[0] == 0:
                        ax.patches[i].set_hatch('///')
                        ax.patches[i].set_edgecolor('black')

            y_lbl = "Percentage (%)" if "%" in metric else "Rounds"
            style_axis(ax, f"{metric} ({dataset})", "Defense", y_lbl)
            ax.set_xticklabels(labels)

            # Annotations
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                                va='bottom', fontsize=14, fontweight='bold', xytext=(0, 5), textcoords='offset points')

            plt.tight_layout()
            safe_metric = re.sub(r'[^\w]', '', metric.split(' ')[0]) + "_" + re.sub(r'[^\w]', '', metric.split(' ')[-1])
            plt.savefig(output_dir / f"plot_{dataset}_{safe_metric}.pdf", format='pdf', dpi=300)
            plt.close('all')

        # Plot Selection Rates
        sel_plot = df_sel_melt[(df_sel_melt['dataset'] == dataset) & (df_sel_melt['defense'].isin(current_order))]
        if not sel_plot.empty:
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(data=sel_plot, x='defense', y='Rate', hue='Type', order=current_order,
                             palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor="black", linewidth=1.2)
            style_axis(ax, f"Avg. Selection Rates ({dataset})", "Defense", "Selection Rate (%)")
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 105)

            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                                va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=14, fontweight='bold')

            plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25)
            plt.savefig(output_dir / f"plot_{dataset}_Selection_Rates.pdf", format='pdf', dpi=300)
            plt.close('all')


def save_threshold_debug_csv(df: pd.DataFrame, output_dir: Path, target_dataset: str = "CIFAR100"):
    """
    Generates a row-by-row CSV showing exactly which configs passed/failed the threshold.
    Useful for tuning the RELATIVE_ACC_THRESHOLD.
    """
    print(f"\n--- Generating Threshold Debug CSV for {target_dataset} ---")

    # 1. Filter for the specific dataset
    subset = df[df['dataset'] == target_dataset].copy()

    if subset.empty:
        print(f"⚠️  No data found for dataset: {target_dataset}")
        return

    # 2. Select useful columns for debugging
    cols = [
        'defense',
        'learning_rate',
        'local_epochs',
        'acc',  # The run's accuracy
        'platform_usable_threshold',  # The cutoff value
        'platform_usable',  # Did it pass?
        'dataset_max_acc'  # The Global Max (Baseline)
    ]

    # Handle missing columns gracefully
    cols = [c for c in cols if c in subset.columns]
    debug_df = subset[cols].copy()

    # 3. Add a "Distance to Threshold" column (Positive = Safe, Negative = Failed)
    if 'platform_usable_threshold' in debug_df.columns:
        debug_df['diff_from_threshold'] = debug_df['acc'] - debug_df['platform_usable_threshold']

    # 4. Sort: Defenses first, then Best Accuracy descending
    debug_df = debug_df.sort_values(by=['defense', 'acc'], ascending=[True, False])

    # 5. Save
    csv_name = f"debug_thresholds_{target_dataset}.csv"
    csv_path = output_dir / csv_name
    debug_df.to_csv(csv_path, index=False)

    print(f"✅ Saved Debug CSV to: {csv_path}")
    print("   (Check the 'diff_from_threshold' column to see how close runs were to passing)")

    # Preview top rows
    print("\nPreview (Top 5 Best Runs per Defense):")
    print(debug_df.groupby('defense').head(2).to_string(index=False))


def plot_composite_row(df: pd.DataFrame, output_dir: Path):
    print("\n--- Plotting Composite Row (Compact Legend) ---")

    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.1)

    unique_datasets = df['dataset'].unique()
    markers = ['(a)', '(b)', '(c)', '(d)']

    for target_dataset in unique_datasets:
        print(f"   > Processing composite row for: {target_dataset}")

        subset = df[df['dataset'] == target_dataset]

        current_order = [d for d in DEFENSE_ORDER if d in subset['defense'].unique()]
        if not current_order:
            continue

        labels = get_formatted_labels(current_order)

        # Reduced height slightly since we don't need extra bottom space for legend
        fig, axes = plt.subplots(1, 4, figsize=(26, 6), constrained_layout=True)

        # --- Data Prep ---
        d1 = subset.groupby('defense')['platform_usable'].mean().reindex(current_order).reset_index()
        d1['Value'] = d1['platform_usable'] * 100

        def calc_best_acc_comp(g):
            u = g[g['platform_usable'] == True]
            return u['acc'].mean() if not u.empty else g['acc'].max()

        d2 = subset.groupby('defense').apply(calc_best_acc_comp).reindex(current_order).reset_index(name='acc')
        d2['Value'] = d2['acc'] * 100

        d3 = subset[subset['platform_usable'] == True].groupby('defense')['rounds'].mean().reindex(current_order).reset_index()
        d3['Value'] = d3['rounds']

        d4 = subset.groupby('defense')[['benign_selection_rate', 'adv_selection_rate']].mean().reindex(current_order).reset_index()
        d4 = d4.melt(id_vars='defense', var_name='Type', value_name='Rate')
        d4['Rate'] *= 100
        d4['Type'] = d4['Type'].replace({'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})

        # --- Plotting ---

        # (a) Usability
        sns.barplot(ax=axes[0], data=d1, x='defense', y='Value', order=current_order, palette='viridis', edgecolor='black', linewidth=2)
        axes[0].set_title(f"{markers[0]} Usability Rate (%)", fontweight='bold', fontsize=24, pad=15)
        axes[0].set_ylim(0, 105)

        # (b) Accuracy
        sns.barplot(ax=axes[1], data=d2, x='defense', y='Value', order=current_order, palette='viridis', edgecolor='black', linewidth=2)
        axes[1].set_title(f"{markers[1]} Avg. Usable Acc (%)", fontweight='bold', fontsize=24, pad=15)
        axes[1].set_ylim(0, 105)
        for i, defense in enumerate(current_order):
            if d1[d1['defense'] == defense]['Value'].values[0] == 0:
                axes[1].patches[i].set_hatch('///')
                axes[1].patches[i].set_edgecolor('black')

        # (c) Cost
        sns.barplot(ax=axes[2], data=d3, x='defense', y='Value', order=current_order, palette='viridis', edgecolor='black', linewidth=2)
        axes[2].set_title(f"{markers[2]} Avg. Cost (Rounds)", fontweight='bold', fontsize=24, pad=15)

        # (d) Selection
        sns.barplot(ax=axes[3], data=d4, x='defense', y='Rate', hue='Type', order=current_order,
                    palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor='black', linewidth=2)
        axes[3].set_title(f"{markers[3]} Avg. Selection Rates", fontweight='bold', fontsize=24, pad=15)
        axes[3].set_ylim(0, 105)

        # --- LEGEND INSIDE FIGURE ---
        # loc='upper center': Top middle of the plot box
        # ncol=2: Side-by-side labels
        # framealpha=0.9: Semi-transparent white background so bars don't make text unreadable
        axes[3].legend(loc='upper center', ncol=2, frameon=True, framealpha=0.9, fontsize=16)

        # --- Common Styling ---
        for ax in axes:
            ax.set_xticklabels(labels, fontsize=18, fontweight='bold')
            ax.set_xlabel("")

            ax.tick_params(axis='y', labelsize=16)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')

            ax.grid(axis='y', alpha=0.5, linewidth=1.5)

            for p in ax.patches:
                h = p.get_height()
                if not np.isnan(h) and h > 0:
                    ax.annotate(f'{h:.0f}',
                                (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom',
                                fontsize=16, fontweight='bold',
                                xytext=(0, 4), textcoords='offset points')

        save_path = output_dir / f"plot_row_combined_{target_dataset}.pdf"
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
        print(f"     Saved composite row to: {save_path}")
        plt.close(fig)

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    df = collect_all_results(BASE_RESULTS_DIR)
    if not df.empty:
        # Generate Utility CSV
        # save_utility_csv(df, output_dir)
        #
        # # Plot Individual Metrics (already loops internally)
        # plot_platform_usability_with_selection(df, output_dir)

        # Plot Composite Rows (Now loops internally for all datasets)
        plot_composite_row(df, output_dir)

        # Generate Debug CSVs for ALL datasets found
        # for ds in df['dataset'].unique():
        #     save_threshold_debug_csv(df, output_dir, target_dataset=ds)


if __name__ == "__main__":
    main()
