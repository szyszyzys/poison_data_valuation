import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
BASE_RESULTS_DIR = "./results"
FIGURE_OUTPUT_DIR = "./figures/step3_figures"

# Define the relative 'usability' threshold
RELATIVE_ACC_THRESHOLD = 0.90
# We can keep this for the fairness filter
REASONABLE_BSR_THRESHOLD = 0.50


# --- Styling Helper ---
def set_plot_style():
    """Sets a consistent professional style for all plots."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.5


# --- Functions from Step 2.5 (for lookup) ---

def parse_scenario_name_step2_5(scenario_name: str) -> Dict[str, str]:
    try:
        pattern = r'step2\.5_find_hps_(fedavg|martfl|fltrust|skymask)_(image|text|tabular)_(.+)'
        match = re.search(pattern, scenario_name)
        if match:
            return {"dataset": match.group(3), "defense": match.group(1)}
        else:
            return {}
    except Exception:
        return {}


def load_run_data_step2_5(metrics_file: Path) -> Dict[str, Any]:
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return {'acc': metrics.get('acc', 0)}
    except Exception:
        return {}


def get_step2_5_max_acc_lookup(base_dir: str) -> Dict[str, float]:
    print("--- Loading Step 2.5 data to find 'best' accuracy... ---")
    all_runs = []
    base_path = Path(base_dir)
    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]

    if not scenario_folders:
        print("Warning: No 'step2.5_find_hps_*' directories found.")
        return {}

    for scenario_path in scenario_folders:
        scenario_info = parse_scenario_name_step2_5(scenario_path.name)
        if not scenario_info: continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_metrics = load_run_data_step2_5(metrics_file)
            if run_metrics:
                all_runs.append({
                    "dataset": scenario_info['dataset'],
                    "defense": scenario_info['defense'],
                    "acc": run_metrics['acc']
                })

    if not all_runs:
        return {}

    df_step2_5 = pd.DataFrame(all_runs)
    dataset_max_acc = df_step2_5.groupby('dataset')['acc'].max().to_dict()
    return dataset_max_acc


# --- Step 3 Parsing Functions ---

def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    hps = {}
    patterns = {
        'martfl.max_k': r'martfl\.max_k_([0-9]+)',
        'clip_norm': r'clip_norm_([0-9\.]+|None)',
        'mask_epochs': r'mask_epochs_([0-9]+)',
        'mask_lr': r'mask_lr_([0-9e\.\+]+)',
        'mask_threshold': r'mask_threshold_([0-9\.]+)',
        'mask_clip': r'mask_clip_([0-9e\.\-]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, hp_folder_name)
        if match:
            value_str = match.group(1)
            clean_key = key.split('.')[-1]
            if value_str == "None":
                hps[clean_key] = 'None'
            else:
                try:
                    val_float = float(value_str)
                    if val_float == int(val_float):
                        hps[clean_key] = int(val_float)
                    else:
                        hps[clean_key] = val_float
                except ValueError:
                    hps[clean_key] = value_str
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    try:
        # Explicitly handle the _new suffix in the regex
        pattern = r'step3_tune_(fedavg|martfl|fltrust|skymask)_([a-z]+)_(image|text|tabular)_(.+?)_(.+?)(_new|_old)?$'
        match = re.search(pattern, scenario_name)
        if match:
            return {
                "scenario": scenario_name,
                "defense": match.group(1),
                "attack": match.group(2),
                "modality": match.group(3),
                "dataset": match.group(4),
                "model": match.group(5),
            }
        else:
            return {"scenario": scenario_name}
    except Exception as e:
        return {"scenario": scenario_name}


def load_run_data(metrics_file: Path) -> Dict[str, Any]:
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
            if adv_sellers:
                run_data['adv_selection_rate'] = pd.Series([s['selection_rate'] for s in adv_sellers]).mean()
            if ben_sellers:
                run_data['benign_selection_rate'] = pd.Series([s['selection_rate'] for s in ben_sellers]).mean()
            if not adv_sellers and ben_sellers:
                run_data['adv_selection_rate'] = 0.0
        return run_data
    except Exception as e:
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)

    # --- STRICT FILTERING: ONLY FOLDERS ENDING IN _new ---
    print(f"Searching for 'step3_tune_*_new' results in {base_path.resolve()}...")
    scenario_folders = [f for f in base_path.glob("step3_tune_*_new") if f.is_dir()]

    if not scenario_folders:
        print(f"Error: No directories matching 'step3_tune_*_new' found.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} valid '_new' scenario directories.")

    for scenario_path in scenario_folders:
        run_scenario = parse_scenario_name(scenario_path.name)
        for hp_path in scenario_path.iterdir():
            if not hp_path.is_dir(): continue
            hp_folder_name = hp_path.name
            run_hps = parse_hp_suffix(hp_folder_name)

            for metrics_file in hp_path.rglob("final_metrics.json"):
                try:
                    run_metrics = load_run_data(metrics_file)
                    if run_metrics:
                        all_runs.append({
                            **run_scenario,
                            **run_hps,
                            **run_metrics,
                            "hp_suffix": hp_folder_name
                        })
                except Exception as e:
                    pass

    if not all_runs:
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)
    df['defense_score'] = df['acc'] - df['asr']
    if 'mask_clip' in df.columns:
        df['mask_clip'] = df['mask_clip'].fillna(1.0)

    # Add a 'Filter Passed' flag to explain discrepancies between plots and tables
    if 'benign_selection_rate' in df.columns:
        df['filter_failure'] = (df['benign_selection_rate'] >= 0.99) & (df['adv_selection_rate'] >= 0.99)

        # Check Fairness Threshold
        df['is_fair'] = df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD
    else:
        df['filter_failure'] = np.nan
        df['is_fair'] = True  # Assume fair if no metric

    return df


# --- PLOTTING FUNCTIONS ---

def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: Path):
    """
    Generates the scatter plot (Acc vs ASR) with improved consistency logic.
    """
    print("\n--- Generating Trade-off Scatter Plots ---")
    set_plot_style()

    combinations = df[['dataset', 'attack']].drop_duplicates().values

    for dataset, attack in combinations:
        subset = df[(df['dataset'] == dataset) & (df['attack'] == attack)].copy()
        if subset.empty: continue

        # Convert to percentages
        subset['acc_pct'] = subset['acc'] * 100
        subset['asr_pct'] = subset['asr'] * 100

        # Mark points that pass the Table's criteria
        # (Relative Acc check is done per-row later, but here we check fairness)
        subset['status'] = 'Unfair/Filtered'
        subset.loc[subset['is_fair'], 'status'] = 'Valid Candidate'

        plt.figure(figsize=(10, 8))

        # Scatter Plot
        sns.scatterplot(
            data=subset, x="acc_pct", y="asr_pct",
            hue="defense",
            style="status",  # Different marker for filtered vs valid points
            markers={"Valid Candidate": "o", "Unfair/Filtered": "X"},
            palette="deep", s=120, alpha=0.8, edgecolor="black"
        )

        # Draw "Target Zone"
        ax = plt.gca()
        rect = patches.Rectangle((85, 0), 20, 10, linewidth=2,
                                 edgecolor='green', facecolor='green', alpha=0.1, linestyle='--')
        ax.add_patch(rect)

        plt.text(92.5, 5, "Target Zone", color='green',
                 fontsize=12, fontweight='bold', ha='center', va='center')

        plt.title(f"Defense Trade-off: {dataset} ({attack})", fontsize=16, fontweight='bold')
        plt.xlabel("Model Utility (Accuracy %)", fontsize=14)
        plt.ylabel("Attack Success Rate (ASR %)", fontsize=14)

        plt.xlim(0, 105)
        plt.ylim(-5, 105)

        plt.axhline(y=90, color='red', linestyle=':', alpha=0.5, label="Attack Goal")

        plt.legend(title="Defense & Status", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()

        filename = output_dir / f"plot_tradeoff_scatter_{dataset}_{attack}.pdf"
        plt.savefig(filename, bbox_inches='tight', format='pdf')
        plt.close('all')
        print(f"Saved scatter plot: {filename}")


def plot_step3_composite_summary(df: pd.DataFrame, output_dir: Path):
    """
    Generates the 4-panel composite row (Usability, Acc, Rounds, Selection).
    """
    print("\n--- Generating Composite Summary Plots (4-in-1) ---")
    set_plot_style()

    groups = df.groupby(['dataset', 'attack'])

    for (dataset, attack), group_df in groups:
        defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
        defense_order = [d for d in defense_order if d in group_df['defense'].unique()]
        formatted_labels = [d.capitalize().replace('Martfl', 'MARTFL').replace('Fedavg', 'FedAvg').replace('Fltrust',
                                                                                                           'FLTrust').replace(
            'Skymask', 'SkyMask') for d in defense_order]

        if not defense_order: continue

        d1 = group_df.groupby('defense')['platform_usable'].mean().reindex(defense_order).reset_index()
        d1['Value'] = d1['platform_usable'] * 100

        usable_runs = group_df[group_df['platform_usable'] == True]
        if not usable_runs.empty:
            d2 = usable_runs.groupby('defense')['acc'].mean().reindex(defense_order).reset_index()
            d2['Value'] = d2['acc'] * 100
            d3 = usable_runs.groupby('defense')['rounds'].mean().reindex(defense_order).reset_index()
            d3['Value'] = d3['rounds']
        else:
            d2 = pd.DataFrame({'defense': defense_order, 'Value': 0})
            d3 = pd.DataFrame({'defense': defense_order, 'Value': 0})

        d4 = group_df.groupby('defense')[['benign_selection_rate', 'adv_selection_rate']].mean().reindex(
            defense_order).reset_index()
        d4 = d4.melt(id_vars='defense', var_name='Type', value_name='Rate')
        d4['Rate'] = d4['Rate'] * 100
        d4['Type'] = d4['Type'].replace({'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})

        fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

        sns.barplot(ax=axes[0], data=d1, x='defense', y='Value', order=defense_order, palette='viridis',
                    edgecolor='black')
        axes[0].set_title("Usability Rate (%)", fontweight='bold')
        axes[0].set_ylabel("Percentage")
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 105)

        sns.barplot(ax=axes[1], data=d2, x='defense', y='Value', order=defense_order, palette='viridis',
                    edgecolor='black')
        axes[1].set_title("Avg. Usable Acc (%)", fontweight='bold')
        axes[1].set_ylabel("")
        axes[1].set_xlabel("")
        axes[1].set_ylim(0, 105)

        sns.barplot(ax=axes[2], data=d3, x='defense', y='Value', order=defense_order, palette='viridis',
                    edgecolor='black')
        axes[2].set_title("Avg. Cost (Rounds)", fontweight='bold')
        axes[2].set_ylabel("Rounds")
        axes[2].set_xlabel("")

        sns.barplot(ax=axes[3], data=d4, x='defense', y='Rate', hue='Type', order=defense_order,
                    palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor='black')
        axes[3].set_title("Avg. Selection Rates", fontweight='bold')
        axes[3].set_ylabel("")
        axes[3].set_xlabel("")
        axes[3].set_ylim(0, 105)
        axes[3].legend(title=None, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

        for ax in axes:
            ax.set_xticklabels(formatted_labels, fontsize=12, fontweight='bold', rotation=15)
            ax.grid(axis='y', alpha=0.5)

        fig.suptitle(f"Parameter Tuning Summary: {dataset} ({attack})", fontsize=18, fontweight='bold', y=1.1)
        filename = output_dir / f"plot_composite_{dataset}_{attack}.pdf"
        plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
        plt.close('all')

    print("Composite plots saved.")


def plot_skymask_deep_dive(df_all: pd.DataFrame, output_dir: Path):
    print("\n--- Generating SkyMask Deep-Dive Analysis Plot ---")
    set_plot_style()
    df_sky = df_all[df_all['defense'] == 'skymask'].copy()
    if df_sky.empty: return

    for col, val in [('mask_clip', '1.0'), ('mask_lr', '0.01'), ('mask_epochs', '20'), ('mask_threshold', '0.5')]:
        if col not in df_sky.columns:
            df_sky[col] = val
        else:
            df_sky[col] = df_sky[col].astype(str)

    def categorize_hps(row):
        is_official_lr = row['mask_lr'] in ['10000000.0', '100000000.0']
        is_official_clip = row['mask_clip'] == '1e-07'
        if is_official_lr and is_official_clip:
            return "Official (Trick)"
        elif is_official_lr:
            return "Partial (High LR)"
        elif is_official_clip:
            return "Partial (Low Clip)"
        else:
            return "Normal HPs"

    df_sky['hp_type'] = df_sky.apply(categorize_hps, axis=1)
    plot_df = df_sky.melt(id_vars=['dataset', 'attack', 'hp_type'], value_vars=['defense_score'], var_name='Metric',
                          value_name='Value')

    g = sns.catplot(
        data=plot_df, x='hp_type', y='Value', col='attack',
        kind='bar', palette='viridis', height=4, aspect=1.5,
        sharey='row', legend_out=True, edgecolor='black'
    )
    g.fig.suptitle("SkyMask Analysis: HP Trick Impact", y=1.05, fontweight='bold')
    g.set_titles(col_template="{col_name} Attack")

    plot_file = output_dir / "plot_skymask_deep_dive_analysis.pdf"
    g.fig.savefig(plot_file, bbox_inches='tight', format='pdf')
    plt.close('all')


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str, output_dir: Path):
    scenario_df = df[df['scenario'] == scenario].copy()
    if scenario_df.empty: return
    set_plot_style()

    hp_cols_present = [c for c in ['clip_norm', 'max_k', 'mask_epochs', 'mask_lr', 'mask_threshold', 'mask_clip'] if
                       c in scenario_df.columns]
    hp_cols_to_plot = [c for c in hp_cols_present if scenario_df[c].nunique() > 1]

    if not hp_cols_to_plot:
        scenario_df['hp_label'] = 'default'
    else:
        scenario_df['hp_label'] = scenario_df[hp_cols_to_plot].apply(
            lambda row: ','.join([f"{col.split('.')[-1]}:{row[col]}" for col in hp_cols_to_plot]), axis=1)

    scenario_df = scenario_df.sort_values(by='defense_score', ascending=False)
    metrics = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    metrics = [m for m in metrics if m in scenario_df.columns]
    plot_df = scenario_df.melt(id_vars=['hp_label'], value_vars=metrics, var_name='Metric', value_name='Value')

    metric_map = {'acc': 'Accuracy', 'asr': 'ASR', 'adv_selection_rate': 'Adv. Select',
                  'benign_selection_rate': 'Benign Select'}
    plot_df['Metric'] = plot_df['Metric'].replace(metric_map)

    plt.figure(figsize=(max(12, 0.4 * scenario_df['hp_label'].nunique()), 6))
    sns.barplot(data=plot_df, x='hp_label', y='Value', hue='Metric', palette='deep', edgecolor='black')
    plt.title(f'{defense.upper()}: Performance vs HPs ({scenario})', fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"plot_{scenario}_performance.pdf", bbox_inches='tight', format='pdf')
    plt.close('all')


# --- MAIN FUNCTION ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # 1. Step 2.5 Lookup
    dataset_max_acc_lookup = get_step2_5_max_acc_lookup(BASE_RESULTS_DIR)

    # 2. Load Data (STRICT FILTERING)
    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No 'step3_tune_*_new' results data was loaded. Exiting.")
        return

    df.to_csv(output_dir / "step3_full_results_summary.csv", index=False, float_format="%.4f")

    # 3. Calculate Usability
    if dataset_max_acc_lookup:
        df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc_lookup)
        if df['dataset_max_acc'].isnull().any():
            step3_max_acc = df.groupby('dataset')['acc'].max().to_dict()
            df['dataset_max_acc'] = df['dataset_max_acc'].fillna(df['dataset'].map(step3_max_acc))
        df['usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD
        df['platform_usable'] = df['acc'] >= df['usable_threshold']
    else:
        print("Warning: No baseline found. Assuming all runs are 'usable' relative to themselves.")
        df['platform_usable'] = True

    # 4. Separate DataFrame for "Good/Fair" Tables
    # This df is ONLY used for the summary tables to show "Best" performance
    reasonable_acc_df = df[df['platform_usable']].copy()

    # --- TABLES ---
    # 1. Tuning Summary
    agg_metrics = {
        'hp_suffix': 'nunique',
        'acc': ['min', 'max', 'mean'],
        'asr': ['min', 'max', 'mean']
    }
    if not reasonable_acc_df.empty:
        df_summary = reasonable_acc_df.groupby(['defense', 'dataset', 'attack']).agg(agg_metrics)
        df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
        df_summary = df_summary.reset_index()

        def rng(row, m):
            return f"{row[f'{m}_min'] * 100:.1f}-{row[f'{m}_max'] * 100:.1f}" if f'{m}_min' in row else "N/A"

        df_summary['ACC Range %'] = df_summary.apply(lambda r: rng(r, 'acc'), axis=1)
        df_summary['ASR Range %'] = df_summary.apply(lambda r: rng(r, 'asr'), axis=1)
        df_summary = df_summary.rename(columns={'hp_suffix_nunique': 'N_HPs'})

        cols = ['defense', 'dataset', 'attack', 'N_HPs', 'ACC Range %', 'ASR Range %']
        final_cols = [c for c in cols if c in df_summary.columns]
        df_summary[final_cols].to_latex(output_dir / "step3_tuning_summary_range.tex", index=False)

    # 2. Best HP Table
    if 'benign_selection_rate' in reasonable_acc_df.columns:
        df_final = reasonable_acc_df[reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD].copy()
    else:
        df_final = reasonable_acc_df.copy()

    if not df_final.empty:
        df_final['sort_metric'] = np.where(
            df_final['attack'] == 'backdoor',
            df_final['asr'], 1.0 - df_final['acc']
        )
        best_rows = df_final.sort_values(['scenario', 'sort_metric']).groupby('scenario').head(1)
        cols = ['defense', 'dataset', 'attack', 'acc', 'asr']
        best_rows[[c for c in cols if c in best_rows.columns]].to_latex(output_dir / "step3_best_hps_table.tex",
                                                                        index=False)

    # --- PLOTTING ---
    print("\n--- Generating Visualizations ---")

    # 1. Trade-off Scatter (Shows ALL data, with 'X' for unfair/unusable points)
    plot_tradeoff_scatter(df, output_dir)

    # # 2. Composite Summary
    # plot_step3_composite_summary(df, output_dir)
    #
    # # 3. Deep Dive
    # plot_skymask_deep_dive(df, output_dir)
    #
    # # 4. Details
    # for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
    #     plot_defense_comparison(df, scenario, defense, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()