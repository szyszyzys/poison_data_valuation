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


# --- End Configuration ---


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
    """
    (NEW)
    Loads all 'step2.5' data to find the max accuracy for each dataset.
    Returns a dictionary like: {'CIFAR10': 0.95, 'CIFAR100': 0.82}
    """
    print("--- Loading Step 2.5 data to find 'best' accuracy... ---")
    all_runs = []
    base_path = Path(base_dir)
    # This correctly searches ALL step2.5 folders to get the true max
    scenario_folders = [f for f in base_path.glob("step2.5_find_hps_*") if f.is_dir()]

    if not scenario_folders:
        print("Error: No 'step2.5_find_hps_*' directories found.")
        print("Cannot determine relative accuracy thresholds. Exiting.")
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

    # --- FIND THE MAX ACC PER DATASET ---
    dataset_max_acc = df_step2_5.groupby('dataset')['acc'].max().to_dict()

    print(f"Found max accuracies: {dataset_max_acc}")
    print("--- Done loading Step 2.5 data. ---")
    return dataset_max_acc


# --- End of Step 2.5 Functions ---


# --- Step 3 Functions (Unchanged) ---

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
        # This regex already correctly captures the optional _new suffix
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
    """
    (UPDATED)
    Walks the results directory and aggregates all 'step3' run data,
    but ONLY for folders that end in '_new'.
    """
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for 'step3_tune_*_new' results in {base_path.resolve()}...")

    # --- THIS IS THE FIX ---
    # We now glob for folders ending in '_new'
    scenario_folders = [f for f in base_path.glob("step3_tune_*_new") if f.is_dir()]
    # --- END FIX ---

    if not scenario_folders:
        # --- Updated error message ---
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


# --- Plotting Functions (Unchanged) ---

def plot_skymask_deep_dive(df_all: pd.DataFrame, output_dir: Path):
    print("\n--- Generating SkyMask Deep-Dive Analysis Plot ---")
    df_sky = df_all[df_all['defense'] == 'skymask'].copy()
    if df_sky.empty:
        print("No SkyMask data found for deep-dive plot.")
        return
    # Fill defaults
    if 'mask_clip' not in df_sky.columns: df_sky['mask_clip'] = 1.0
    if 'mask_lr' not in df_sky.columns: df_sky['mask_lr'] = 0.01
    if 'mask_epochs' not in df_sky.columns: df_sky['mask_epochs'] = 20
    if 'mask_threshold' not in df_sky.columns: df_sky['mask_threshold'] = 0.5
    if 'clip_norm' not in df_sky.columns: df_sky['clip_norm'] = 'None'
    # Cast to string
    df_sky['mask_lr'] = df_sky['mask_lr'].astype(str)
    df_sky['mask_clip'] = df_sky['mask_clip'].astype(str)
    df_sky['mask_epochs'] = df_sky['mask_epochs'].astype(str)
    df_sky['mask_threshold'] = df_sky['mask_threshold'].astype(str)

    # Categorize
    def categorize_hps(row):
        is_official_lr = row['mask_lr'] in ['10000000.0', '100000000.0']
        is_official_clip = row['mask_clip'] == '1e-07'
        if is_official_lr and is_official_clip:
            return "Official (Trick HPs)"
        elif is_official_lr:
            return "Partial (High LR, Normal Clip)"
        elif is_official_clip:
            return "Partial (Normal LR, Trick Clip)"
        else:
            return "Normal HPs (No Trick)"

    df_sky['hp_type'] = df_sky.apply(categorize_hps, axis=1)
    plot_df = df_sky.melt(
        id_vars=['dataset', 'attack', 'hp_type', 'mask_threshold'],
        value_vars=['defense_score', 'filter_failure'],
        var_name='Metric', value_name='Value'
    )
    g = sns.catplot(
        data=plot_df, x='hp_type', y='Value', hue='mask_threshold',
        col='attack', row='Metric', kind='bar', palette='viridis',
        height=4, aspect=1.5, sharey='row', legend_out=True
    )
    g.fig.suptitle("SkyMask Deep-Dive: Official HPs vs. Normal HPs", y=1.03)
    g.set_axis_labels("HP Category", "Value")
    g.set_titles(col_template="{col_name} Attack", row_template="{row_name}")
    g.add_legend(title='Mask Threshold')
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    plot_file = output_dir / "plot_skymask_deep_dive_analysis.pdf"
    g.fig.savefig(plot_file, bbox_inches='tight', format='pdf')
    print(f"Saved SkyMask deep-dive plot: {plot_file}")
    plt.clf();
    plt.close('all')


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str, output_dir: Path):
    scenario_df = df[df['scenario'] == scenario].copy()
    if scenario_df.empty:
        print(f"No data found for scenario '{scenario}'")
        return
    print(f"\n--- Visualizing Scenario: {scenario} ---")
    hp_cols_present = [c for c in ['clip_norm', 'max_k', 'mask_epochs', 'mask_lr', 'mask_threshold', 'mask_clip'] if
                       c in scenario_df.columns]
    hp_cols_to_plot = [c for c in hp_cols_present if scenario_df[c].nunique() > 1]
    if not hp_cols_to_plot:
        scenario_df['hp_label'] = 'default'
    else:
        scenario_df['hp_label'] = scenario_df[hp_cols_to_plot].apply(
            lambda row: '_'.join([f"{col.split('.')[-1]}:{row[col]}" for col in hp_cols_to_plot]),
            axis=1
        )
    scenario_df = scenario_df.sort_values(by='defense_score', ascending=False)
    metrics_to_plot = ['acc', 'asr', 'adv_selection_rate', 'benign_selection_rate']
    metrics_to_plot = [m for m in metrics_to_plot if m in scenario_df.columns]
    plot_df = scenario_df.melt(id_vars=['hp_label'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')
    plt.figure(figsize=(max(16, 0.5 * scenario_df['hp_label'].nunique()), 7))
    sns.barplot(data=plot_df, x='hp_label', y='Value', hue='Metric', order=scenario_df['hp_label'].unique())
    plt.title(f'Performance & Selection vs. HPs for {defense.upper()} (Scenario: {scenario})')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination (Sorted by Defense Score)')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.legend(title='Metric')
    plt.tight_layout()
    plot_file = output_dir / f"plot_{scenario}_performance.pdf"
    plt.savefig(plot_file, bbox_inches='tight', format='pdf')
    print(f"Saved plot: {plot_file}")
    plt.clf();
    plt.close('all')
    if len(hp_cols_to_plot) >= 2:
        x_hp = hp_cols_to_plot[0]
        scenario_df[x_hp] = scenario_df[x_hp].astype(str)
        y_hp = hp_cols_to_plot[1]
        col_hp = hp_cols_to_plot[2] if len(hp_cols_to_plot) > 2 else None
        metrics_to_plot_grid = ['defense_score', 'acc', 'asr', 'filter_failure']
        plot_df_melted = scenario_df.melt(id_vars=[c for c in [x_hp, y_hp, col_hp] if c is not None],
                                          value_vars=metrics_to_plot_grid, var_name='Metric', value_name='Value')
        g = sns.catplot(
            data=plot_df_melted, x=x_hp, y='Value', hue=y_hp, col=col_hp,
            row='Metric', kind='bar', palette='viridis', height=3,
            aspect=1.2, sharey=False, legend_out=True
        )
        g.fig.suptitle(f'HP Stability Analysis for {defense.upper()} ({scenario})', y=1.03)
        g.set_axis_labels(x_hp, 'Metric Value')
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        plot_file = output_dir / f"plot_{scenario}_stability_grid.pdf"
        g.fig.savefig(plot_file, bbox_inches='tight', format='pdf')
        print(f"Saved stability grid plot: {plot_file}")
        plt.clf()
        plt.close('all')
    else:
        print(f"Skipping stability grid plot for {defense} (not enough HPs to compare).")


# --- MODIFIED MAIN FUNCTION ---

def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # --- Step 1 ---
    # Load 'step2.5' data to get the 'best' accuracy for each dataset
    dataset_max_acc_lookup = get_step2_5_max_acc_lookup(BASE_RESULTS_DIR)
    if not dataset_max_acc_lookup:
        print("Could not generate max accuracy lookup. Exiting.")
        return

    # --- Step 2 ---
    # Load the main 'step3' data (now filtered to * _new)
    df = collect_all_results(BASE_RESULTS_DIR)
    if df.empty:
        print("No 'step3' results data was loaded. Exiting.")
        return

    # --- Save Full Raw Data (for supplementary material) ---
    csv_output_path = output_dir / "step3_full_results_summary.csv"
    df.to_csv(csv_output_path, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_path}\n")

    # --- NEW: Save "Full Detail" LaTeX Table (for Appendix) ---
    try:
        # Create a copy for LaTeX, format percentages
        df_latex_full = df.copy()
        if 'benign_selection_rate' in df_latex_full.columns:
            df_latex_full['benign_selection_rate'] *= 100
        if 'adv_selection_rate' in df_latex_full.columns:
            df_latex_full['adv_selection_rate'] *= 100

        # Select and rename columns for a readable "full" table
        hp_cols = ['clip_norm', 'max_k', 'mask_epochs', 'mask_lr', 'mask_threshold', 'mask_clip']
        metrics_cols = ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate']
        full_table_cols = ['defense', 'dataset', 'attack'] + \
                          [c for c in hp_cols if c in df_latex_full.columns] + \
                          [c for c in metrics_cols if c in df_latex_full.columns]

        df_latex_full = df_latex_full[full_table_cols]

        latex_full_table_str = df_latex_full.to_latex(
            index=False, escape=False, float_format="%.2f",
            caption="Full results of all hyperparameter combinations (for '_new' runs).",
            label="tab:step3_full_results",
            longtable=True  # Use longtable for multi-page tables
        )
        table_full_path = output_dir / "step3_full_results_summary.tex"
        with open(table_full_path, 'w') as f:
            f.write(latex_full_table_str)
        print(f"\n✅ Successfully saved 'full detail' LaTeX table to: {table_full_path}\n")
    except Exception as e:
        print(f"Error generating full LaTeX table: {e}")
    # --- END NEW BLOCK ---

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Apply Relative Accuracy Filter ---
    print("\n" + "=" * 80)
    print(f" Filtering by Relative Accuracy (>{RELATIVE_ACC_THRESHOLD * 100}%)")
    print("=" * 80)

    df['dataset_max_acc'] = df['dataset'].map(dataset_max_acc_lookup)
    if df['dataset_max_acc'].isnull().any():
        print("Warning: Some step3 datasets not in step2.5. Using step3's max as fallback.")
        step3_max_acc = df.groupby('dataset')['acc'].max().to_dict()
        df['dataset_max_acc'] = df['dataset_max_acc'].fillna(
            df['dataset'].map(step3_max_acc)
        )
    df['usable_threshold'] = df['dataset_max_acc'] * RELATIVE_ACC_THRESHOLD

    # This is the "usable" dataframe
    reasonable_acc_df = df[df['acc'] >= df['usable_threshold']].copy()
    print(f"Total runs: {len(df)}. Runs passing relative ACC filter: {len(reasonable_acc_df)}")

    if reasonable_acc_df.empty:
        print(f"\n!WARNING: No runs met the relative accuracy threshold. Using all runs for summary tables.")
        # Fallback to the full dataframe for the summary tables
        reasonable_acc_df = df.copy()

    # --- Analysis for "Sweep Summary" Table (Objective 0) ---
    print("\n" + "=" * 80)
    print("           Objective 0: Generating Sweep Summary Table (from usable runs)")
    print("=" * 80)

    agg_metrics = {
        'hp_suffix': 'nunique',
        'acc': ['min', 'max', 'mean'],
        'asr': ['min', 'max', 'mean'],
        'benign_selection_rate': ['min', 'max', 'mean'],
        'adv_selection_rate': ['min', 'max', 'mean']
    }
    agg_metrics = {k: v for k, v in agg_metrics.items() if k in reasonable_acc_df.columns}

    if agg_metrics:
        # --- THIS IS THE FIX ---
        # We now use 'reasonable_acc_df' as the source
        df_tuning_summary = reasonable_acc_df.groupby(['defense', 'dataset', 'attack']).agg(agg_metrics)
        # --- END FIX ---

        df_tuning_summary.columns = ['_'.join(col).strip() for col in df_tuning_summary.columns.values]
        df_tuning_summary = df_tuning_summary.reset_index()

        def create_range_str(row, metric):
            min_val = row.get(f'{metric}_min', np.nan) * 100
            max_val = row.get(f'{metric}_max', np.nan) * 100
            if pd.isna(min_val): return "N/A"
            return f"{min_val:.1f} - {max_val:.1f}"

        df_tuning_summary['ACC Range (%)'] = df_tuning_summary.apply(lambda r: create_range_str(r, 'acc'), axis=1)
        df_tuning_summary['ASR Range (%)'] = df_tuning_summary.apply(lambda r: create_range_str(r, 'asr'), axis=1)
        df_tuning_summary['Benign Select. Range (%)'] = df_tuning_summary.apply(
            lambda r: create_range_str(r, 'benign_selection_rate'), axis=1)

        # --- FIX 1: Add Adv. Select. Range ---
        df_tuning_summary['Adv. Select. Range (%)'] = df_tuning_summary.apply(
            lambda r: create_range_str(r, 'adv_selection_rate'), axis=1)

        # --- FIX 2: Add Adv. Select. Range to the column list ---
        final_summary_cols = [
            'defense', 'dataset', 'attack', 'hp_suffix_nunique',
            'ACC Range (%)', 'ASR Range (%)', 'Benign Select. Range (%)', 'Adv. Select. Range (%)'
        ]
        df_tuning_summary = df_tuning_summary.rename(columns={'hp_suffix_nunique': 'Num. HPs'})
        final_summary_cols = [c for c in final_summary_cols if c in df_tuning_summary.columns]
        df_tuning_summary_final = df_tuning_summary[final_summary_cols]

        latex_summary_table_str = df_tuning_summary_final.to_latex(
            index=False, escape=False, float_format="%.1f",
            caption="Summary of defense performance across all *usable* tuned hyperparameters (for '_new' runs).",
            label="tab:step3_tuning_summary", position="H"
        )
        table_summary_path = output_dir / "step3_tuning_summary_range.tex"
        with open(table_summary_path, 'w') as f:
            f.write(latex_summary_table_str)
        print(f"\n✅ Successfully saved tuning summary range table to: {table_summary_path}\n")
        print("--- Tuning Summary Table (for console) ---")
        print(df_tuning_summary_final.to_string(index=False))

    else:
        print("No metrics found to generate a summary table.")

    # --- Analysis for "Best HP" Table (Objective 1) ---
    print("\n" + "=" * 80)
    print(f" Objective 1: Finding Best HPs (Filtered by Relative Acc & BSR)")
    print("=" * 80)

    # Apply Benign Selection Rate (Fairness) filter
    if 'benign_selection_rate' in reasonable_acc_df.columns:
        reasonable_final_df = reasonable_acc_df[
            reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD
            ].copy()
        print(f"Runs passing BSR filter: {len(reasonable_final_df)}")
        if reasonable_final_df.empty:
            print(f"\n!WARNING: No runs passed the BSR threshold (after passing accuracy).")
            print("  Falling back to only accuracy-filtered runs.")
            reasonable_final_df = reasonable_acc_df.copy()
    else:
        print("\n!WARNING: 'benign_selection_rate' not found. Skipping Fairness filter.")
        reasonable_final_df = reasonable_acc_df.copy()

    # Create and apply sort metric
    reasonable_final_df['sort_metric'] = np.where(
        reasonable_final_df['attack'] == 'backdoor',
        reasonable_final_df['asr'], 1.0 - reasonable_final_df['acc']
    )
    sort_columns = ['scenario', 'sort_metric']
    sort_ascending = [True, True]
    if 'adv_selection_rate' in reasonable_final_df.columns:
        sort_columns.append('adv_selection_rate')
        sort_ascending.append(True)
    df_sorted = reasonable_final_df.sort_values(by=sort_columns, ascending=sort_ascending)

    # --- Generate "Best HP" Table ---
    print(f"\n--- Generating Best HP Table (Filtered by Relative Acc & BSR) ---")
    grouped = df_sorted.groupby('scenario')
    best_rows_list = []

    for name, group in grouped:
        if group.empty:
            continue
        best_row = group.iloc[0:1].copy()
        hp_cols = ['clip_norm', 'max_k', 'mask_epochs', 'mask_lr', 'mask_threshold', 'mask_clip']
        hp_cols_present = [c for c in hp_cols if c in group.columns and group[c].nunique() > 1]
        if hp_cols_present:
            def get_hp_str(row):
                return ', '.join([f"{col.split('.')[-1]}: {row[col]}" for col in hp_cols_present])

            best_row['Tuned HPs'] = best_row.apply(get_hp_str, axis=1)
        else:
            best_row['Tuned HPs'] = 'default'
        best_rows_list.append(best_row)

    if best_rows_list:
        df_best_hps = pd.concat(best_rows_list, ignore_index=True)
        cols_from_scenario = ['defense', 'attack', 'dataset']
        cols_from_hps = ['Tuned HPs']
        cols_from_metrics = ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate', 'defense_score']
        final_table_cols = [c for c in cols_from_scenario + cols_from_hps + cols_from_metrics if
                            c in df_best_hps.columns]
        df_final_table = df_best_hps[final_table_cols]

        clean_col_names = {
            'defense': 'Defense', 'attack': 'Attack', 'dataset': 'Dataset',
            'acc': 'ACC', 'asr': 'ASR',
            'benign_selection_rate': 'Benign Select (%)', 'adv_selection_rate': 'Adv. Select (%)',
            'defense_score': 'Defense Score'
        }
        if 'benign_selection_rate' in df_final_table.columns:
            df_final_table['benign_selection_rate'] *= 100
        if 'adv_selection_rate' in df_final_table.columns:
            df_final_table['adv_selection_rate'] *= 100
        df_final_table = df_final_table.rename(columns=clean_col_names)
        final_col_order = [
            'Defense', 'Attack', 'Dataset', 'Tuned HPs', 'ACC', 'ASR',
            'Benign Select (%)', 'Adv. Select (%)', 'Defense Score'
        ]
        final_col_order = [c for c in final_col_order if c in df_final_table.columns]
        df_final_table = df_final_table[final_col_order]

        latex_table_str = df_final_table.to_latex(
            index=False, escape=False, float_format="%.2f",
            caption="Best-case defense performance after hyperparameter tuning, filtered by relative usability thresholds (for '_new' runs).",
            label="tab:step3_best_ps", position="H"
        )
        table_output_path = output_dir / "step3_best_hps_table.tex"
        with open(table_output_path, 'w') as f:
            f.write(latex_table_str)
        print(f"\n✅ Successfully saved LaTeX table of best HPs to: {table_output_path}\n")
        print("--- Best HP Summary Table (for console) ---")
        print(df_final_table.to_string(index=False, float_format="%.2f"))
    else:
        print("\nNo rows met the filtering criteria to generate a 'best HPs' table.")

    # --- Analysis for Objective 2: Stableness (Visualization) ---
    print("\n" + "=" * 80)
    print("           Objective 2: Assessing Defense Stability (Plots)")
    print("=" * 80)

    plot_skymask_deep_dive(df, output_dir)
    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        plot_defense_comparison(df, scenario, defense, output_dir)

    print("\nAnalysis complete. Check 'step3_figures' folder for plots and tables.")


if __name__ == "__main__":
    main()