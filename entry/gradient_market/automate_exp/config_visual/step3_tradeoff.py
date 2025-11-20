import pandas as pd
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
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
    """Parses the 'step2.5_find_hps_*' scenario name."""
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
    """Loads only 'acc' from a step2.5 'final_metrics.json'."""
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
        print("Error: No 'step2.5_find_hps_*' directories found.")
        return {}

    for scenario_path in scenario_folders:
        scenario_info = parse_scenario_name_step2_5(scenario_path.name)
        if not scenario_info:
            continue

        for metrics_file in scenario_path.rglob("final_metrics.json"):
            run_metrics = load_run_data_step2_5(metrics_file)
            if run_metrics:
                all_runs.append({
                    "dataset": scenario_info['dataset'],
                    "defense": scenario_info['defense'],
                    "acc": run_metrics['acc']
                })

    if not all_runs:
        print("Error: No 'final_metrics.json' files found in step2.5 directories.")
        return {}

    df_step2_5 = pd.DataFrame(all_runs)
    dataset_max_acc = df_step2_5.groupby('dataset')['acc'].max().to_dict()
    print(f"Found max accuracies: {dataset_max_acc}")
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
            raise ValueError(f"Pattern not matched for: {scenario_name}")
    except Exception as e:
        print(f"Warning: Could not parse scenario name '{scenario_name}': {e}")
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
        print(f"Error loading data from {metrics_file.parent}: {e}")
        return {}


def collect_all_results(base_dir: str) -> pd.DataFrame:
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for 'step3_tune_*_new' results in {base_path.resolve()}...")

    scenario_folders = [f for f in base_path.glob("step3_tune_*_new") if f.is_dir()]

    if not scenario_folders:
        print(f"Error: No 'step3_tune_*_new' directories found.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} '_new' scenario base directories.")
    for scenario_path in scenario_folders:
        if not scenario_path.is_dir(): continue
        scenario_name = scenario_path.name
        run_scenario = parse_scenario_name(scenario_name)
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
                    print(f"Error processing file {metrics_file}: {e}")
    if not all_runs:
        print("Error: No 'final_metrics.json' files were successfully processed.")
        return pd.DataFrame()
    df = pd.DataFrame(all_runs)
    df['defense_score'] = df['acc'] - df['asr']
    if 'mask_clip' in df.columns:
        df['mask_clip'] = df['mask_clip'].fillna(1.0)
    if 'benign_selection_rate' in df.columns and 'adv_selection_rate' in df.columns:
        df['filter_failure'] = (df['benign_selection_rate'] >= 0.99) & \
                               (df['adv_selection_rate'] >= 0.99)
    else:
        df['filter_failure'] = np.nan
    return df


# --- Plotting Functions ---

def plot_step3_composite_summary(df: pd.DataFrame, output_dir: Path):
    """
    Generates the 4-panel composite row (Usability, Acc, Rounds, Selection)
    for every (dataset, attack) combination found in the dataframe.
    """
    print("\n--- Generating Composite Summary Plots (4-in-1) ---")
    set_plot_style()

    # Group by dataset and attack to generate one plot per scenario type
    groups = df.groupby(['dataset', 'attack'])

    for (dataset, attack), group_df in groups:
        print(f"Processing Composite Plot for: {dataset} - {attack}")

        # Defense Order & Labels
        defense_order = ['fedavg', 'fltrust', 'martfl', 'skymask']
        defense_order = [d for d in defense_order if d in group_df['defense'].unique()]
        formatted_labels = [d.capitalize() if d != 'martfl' else 'MARTFL' for d in defense_order]
        formatted_labels = [d.replace('Skymask', 'SkyMask').replace('Fltrust', 'FLTrust').replace('Fedavg', 'FedAvg')
                            for d in formatted_labels]

        if not defense_order:
            continue

        # --- Data Prep ---
        # 1. Usability Rate (%)
        d1 = group_df.groupby('defense')['platform_usable'].mean().reindex(defense_order).reset_index()
        d1['Value'] = d1['platform_usable'] * 100

        # 2. Usable Accuracy (avg of only the useful runs)
        usable_runs = group_df[group_df['platform_usable'] == True]
        if not usable_runs.empty:
            d2 = usable_runs.groupby('defense')['acc'].mean().reindex(defense_order).reset_index()
            d2['Value'] = d2['acc'] * 100

            # 3. Usable Rounds
            d3 = usable_runs.groupby('defense')['rounds'].mean().reindex(defense_order).reset_index()
            d3['Value'] = d3['rounds']
        else:
            d2 = pd.DataFrame({'defense': defense_order, 'Value': 0})
            d3 = pd.DataFrame({'defense': defense_order, 'Value': 0})

        # 4. Selection Rates
        d4 = group_df.groupby('defense')[['benign_selection_rate', 'adv_selection_rate']].mean().reindex(
            defense_order).reset_index()
        d4 = d4.melt(id_vars='defense', var_name='Type', value_name='Rate')
        d4['Rate'] = d4['Rate'] * 100
        d4['Type'] = d4['Type'].replace({'benign_selection_rate': 'Benign', 'adv_selection_rate': 'Adversary'})

        # --- Plotting ---
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

        # Plot 1: Usability
        sns.barplot(ax=axes[0], data=d1, x='defense', y='Value', order=defense_order, palette='viridis',
                    edgecolor='black')
        axes[0].set_title(f"Usability Rate (%)", fontweight='bold')
        axes[0].set_ylabel("Percentage", fontweight='bold')
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 105)

        # Plot 2: Accuracy
        sns.barplot(ax=axes[1], data=d2, x='defense', y='Value', order=defense_order, palette='viridis',
                    edgecolor='black')
        axes[1].set_title("Avg. Usable Acc (%)", fontweight='bold')
        axes[1].set_ylabel("")
        axes[1].set_xlabel("")
        axes[1].set_ylim(0, 105)

        # Plot 3: Rounds
        sns.barplot(ax=axes[2], data=d3, x='defense', y='Value', order=defense_order, palette='viridis',
                    edgecolor='black')
        axes[2].set_title("Avg. Cost (Rounds)", fontweight='bold')
        axes[2].set_ylabel("Rounds", fontweight='bold')
        axes[2].set_xlabel("")

        # Plot 4: Selection
        sns.barplot(ax=axes[3], data=d4, x='defense', y='Rate', hue='Type', order=defense_order,
                    palette={'Benign': '#2ecc71', 'Adversary': '#e74c3c'}, edgecolor='black')
        axes[3].set_title("Avg. Selection Rates", fontweight='bold')
        axes[3].set_ylabel("")
        axes[3].set_xlabel("")
        axes[3].set_ylim(0, 105)
        # Legend above plot to save space
        axes[3].legend(title=None, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

        # --- Formatting ---
        for ax in axes:
            ax.set_xticklabels(formatted_labels, fontsize=12, fontweight='bold', rotation=15)
            ax.grid(axis='y', alpha=0.5)
            # Annotate Bars
            for p in ax.patches:
                h = p.get_height()
                if not np.isnan(h) and h > 0:
                    ax.annotate(f'{h:.0f}', (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 2),
                                textcoords='offset points')

        fig.suptitle(f"Parameter Tuning Summary: {dataset} ({attack} attack)", fontsize=18, fontweight='bold', y=1.1)

        filename = f"plot_composite_{dataset}_{attack}.pdf"
        plt.savefig(output_dir / filename, bbox_inches='tight', format='pdf', dpi=300)
        plt.close('all')

    print("Composite plots saved.")


def plot_skymask_deep_dive(df_all: pd.DataFrame, output_dir: Path):
    print("\n--- Generating SkyMask Deep-Dive Analysis Plot ---")
    set_plot_style()

    df_sky = df_all[df_all['defense'] == 'skymask'].copy()
    if df_sky.empty:
        print("No SkyMask data found for deep-dive plot.")
        return

    if 'mask_clip' not in df_sky.columns: df_sky['mask_clip'] = 1.0
    if 'mask_lr' not in df_sky.columns: df_sky['mask_lr'] = 0.01
    if 'mask_epochs' not in df_sky.columns: df_sky['mask_epochs'] = 20
    if 'mask_threshold' not in df_sky.columns: df_sky['mask_threshold'] = 0.5

    df_sky['mask_lr'] = df_sky['mask_lr'].astype(str)
    df_sky['mask_clip'] = df_sky['mask_clip'].astype(str)

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
    plot_df = df_sky.melt(
        id_vars=['dataset', 'attack', 'hp_type', 'mask_threshold'],
        value_vars=['defense_score', 'filter_failure'],
        var_name='Metric', value_name='Value'
    )

    g = sns.catplot(
        data=plot_df, x='hp_type', y='Value', hue='mask_threshold',
        col='attack', row='Metric', kind='bar', palette='viridis',
        height=4, aspect=1.5, sharey='row', legend_out=True,
        edgecolor='black'
    )
    g.fig.suptitle("SkyMask Analysis: HP Trick Impact", y=1.05, fontweight='bold')
    g.set_axis_labels("", "Value")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

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
            lambda row: ','.join([f"{col.split('.')[-1]}:{row[col]}" for col in hp_cols_to_plot]),
            axis=1
        )

    scenario_df = scenario_df.sort_values(by='defense_score', ascending=False)

    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    metrics_to_plot = [m for m in metrics_to_plot if m in scenario_df.columns]
    plot_df = scenario_df.melt(id_vars=['hp_label'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')

    metric_map = {
        'acc': 'Accuracy', 'asr': 'ASR',
        'adv_selection_rate': 'Adv. Select', 'benign_selection_rate': 'Benign Select'
    }
    plot_df['Metric'] = plot_df['Metric'].replace(metric_map)

    plt.figure(figsize=(max(12, 0.4 * scenario_df['hp_label'].nunique()), 6))
    sns.barplot(data=plot_df, x='hp_label', y='Value', hue='Metric', palette='deep', edgecolor='black')

    plt.title(f'{defense.upper()}: Performance vs HPs ({scenario})', fontweight='bold')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination (Sorted by Defense Score)')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title=None, bbox_to_anchor=(1.0, 1.02), loc='lower right', ncol=4)
    plt.tight_layout()

    plot_file = output_dir / f"plot_{scenario}_performance.pdf"
    plt.savefig(plot_file, bbox_inches='tight', format='pdf')
    plt.close('all')


# --- MAIN FUNCTION ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # --- Step 1: Load Baseline Max Accuracy ---
    dataset_max_acc_lookup = get_step2_5_max_acc_lookup(BASE_RESULTS_DIR)
    if not dataset_max_acc_lookup:
        print("Could not generate max accuracy lookup. Exiting.")
        return

    # --- Step 2: Load Step 3 Data ---
    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No 'step3' results data was loaded. Exiting.")
        return

    # Save Raw Data
    df.to_csv(output_dir / "step3_full_results_summary.csv", index=False, float_format="%.4f")

    # --- Step 3: Calculate Usability Metrics ---
    print("\n" + "=" * 80)
    print(f" Calculating Usability (>{RELATIVE_ACC_THRESHOLD * 100}%)")
    print("=" * 80)

    # Map max accuracy to rows
    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc_lookup)
    if df['dataset_max_acc'].isnull().any():
        step3_max_acc = df.groupby('dataset')['acc'].max().to_dict()
        df['dataset_max_acc'] = df['dataset_max_acc'].fillna(df['dataset'].map(step3_max_acc))

    df['usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD

    # KEY ADDITION: Create the boolean flag for the plots
    df['platform_usable'] = df['acc'] >= df['usable_threshold']

    # Create filtered DF for the tables (which only care about good runs)
    reasonable_acc_df = df[df['platform_usable']].copy()

    # --- Tables (Objective 0 & 1) ---
    # (Logic remains mostly the same as your original script for tables)

    # 1. Tuning Summary Range Table
    agg_metrics = {
        'hp_suffix': 'nunique',
        'acc': ['min', 'max', 'mean'],
        'asr': ['min', 'max', 'mean'],
        'benign_selection_rate': ['min', 'max', 'mean'],
        'adv_selection_rate': ['min', 'max', 'mean']
    }
    agg_metrics = {k: v for k, v in agg_metrics.items() if k in reasonable_acc_df.columns}

    if agg_metrics:
        df_summary = reasonable_acc_df.groupby(['defense', 'dataset', 'attack']).agg(agg_metrics)
        df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
        df_summary = df_summary.reset_index()

        # Helper for range strings
        def rng(row, m):
            mn_col, mx_col = f'{m}_min', f'{m}_max'
            if mn_col not in row or mx_col not in row: return "N/A"
            mn, mx = row[mn_col] * 100, row[mx_col] * 100
            return "N/A" if pd.isna(mn) else f"{mn:.1f}-{mx:.1f}"

        if 'acc_min' in df_summary.columns:
            df_summary['ACC Range %'] = df_summary.apply(lambda r: rng(r, 'acc'), axis=1)
        if 'asr_min' in df_summary.columns:
            df_summary['ASR Range %'] = df_summary.apply(lambda r: rng(r, 'asr'), axis=1)

        df_summary = df_summary.rename(columns={'hp_suffix_nunique': 'N_HPs'})

        # Robust column selection
        target_cols = ['defense', 'dataset', 'attack', 'N_HPs', 'ACC Range %', 'ASR Range %']
        final_cols = [c for c in target_cols if c in df_summary.columns]

        df_summary[final_cols].to_latex(output_dir / "step3_tuning_summary_range.tex", index=False)
        print("Saved tuning summary table.")

    # 2. Best HP Table
    if 'benign_selection_rate' in reasonable_acc_df.columns:
        reasonable_final_df = reasonable_acc_df[
            reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD].copy()
    else:
        reasonable_final_df = reasonable_acc_df.copy()

    if not reasonable_final_df.empty:
        reasonable_final_df['sort_metric'] = np.where(
            reasonable_final_df['attack'] == 'backdoor',
            reasonable_final_df['asr'], 1.0 - reasonable_final_df['acc']
        )
        best_rows = reasonable_final_df.sort_values(['scenario', 'sort_metric']).groupby('scenario').head(1)
        cols_to_save = ['defense', 'dataset', 'attack', 'acc', 'asr']
        cols_to_save = [c for c in cols_to_save if c in best_rows.columns]
        best_rows[cols_to_save].to_latex(output_dir / "step3_best_hps_table.tex", index=False)

    # --- PLOTTING (Objective 2) ---
    print("\n" + "=" * 80)
    print("           Generating Visualizations")
    print("=" * 80)

    # 1. The New Composite Plot (Replaces the old generic stability checks)
    plot_step3_composite_summary(df, output_dir)

    # 2. Deep Dive Specifics
    plot_skymask_deep_dive(df, output_dir)

    # 3. Detailed Per-Scenario Breakdown (Optional, but good for debugging)
    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        plot_defense_comparison(df, scenario, defense, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()