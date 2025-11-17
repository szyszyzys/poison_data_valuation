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

# Set your minimum acceptable accuracy threshold (e.g., 0.70 for 70%)
REASONABLE_ACC_THRESHOLD = 0.70

# Set your minimum acceptable Benign Selection Rate (e.g., 0.50 for 50%)
REASONABLE_BSR_THRESHOLD = 0.50


# --- End Configuration ---


def parse_hp_suffix(hp_folder_name: str) -> Dict[str, Any]:
    """
    (FIXED) Parses the HP suffix folder name using regex.
    This is robust to underscores in parameter names and scientific notation.
    """
    hps = {}

    # Define the regex patterns for each HP we expect
    patterns = {
        'martfl.max_k': r'martfl\.max_k_([0-9]+)',
        'clip_norm': r'clip_norm_([0-9\.]+|None)',  # Looks for 'clip_norm_...'
        'mask_epochs': r'mask_epochs_([0-9]+)',
        'mask_lr': r'mask_lr_([0-9e\.\+]+)',  # Handles 0.01, 0.5, and 1e7
        'mask_threshold': r'mask_threshold_([0-9\.]+)',
        'mask_clip': r'mask_clip_([0-9e\.\-]+)'  # Handles 10.0 and 1e-7
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, hp_folder_name)
        if match:
            value_str = match.group(1)

            # Clean up key name
            clean_key = key.split('.')[-1]  # e.g., 'skymask.mask_epochs' -> 'mask_epochs'

            if value_str == "None":
                hps[clean_key] = 'None'  # Use string 'None' for grouping
            else:
                # Try to cast to float (which covers int and sci-notation)
                try:
                    val_float = float(value_str)
                    # If it's a whole number, store as int for cleaner grouping
                    if val_float == int(val_float):
                        hps[clean_key] = int(val_float)
                    else:
                        hps[clean_key] = val_float
                except ValueError:
                    hps[clean_key] = value_str  # Fallback
    return hps


def parse_scenario_name(scenario_name: str) -> Dict[str, str]:
    """
    (FIXED) Parses the base scenario name using regex to handle
    underscores in dataset and model names.
    e.g., 'step3_tune_martfl_labelflip_tabular_Texas100_mlp_texas100_baseline_new'
    """
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
    """
    Loads key data from final_metrics.json and marketplace_report.json
    """
    run_data = {}
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        run_data['acc'] = metrics.get('acc', 0)
        run_data['asr'] = metrics.get('asr', 0)
        run_data['loss'] = metrics.get('loss', 0)
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
    """Walks the results directory and aggregates all run data."""
    all_runs = []
    base_path = Path(base_dir)
    print(f"Searching for results in {base_path.resolve()}...")

    scenario_folders = list(base_path.glob("step3_tune_*"))
    if not scenario_folders:
        print(f"Error: No 'step3_tune_*' directories found directly inside {base_path}.")
        return pd.DataFrame()

    print(f"Found {len(scenario_folders)} scenario base directories.")

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


def plot_skymask_deep_dive(df_all: pd.DataFrame, output_dir: Path):
    """
    Generates a deep-dive plot for SkyMask to compare official vs. normal HPs.
    """
    print("\n--- Generating SkyMask Deep-Dive Analysis Plot ---")

    df_sky = df_all[df_all['defense'] == 'skymask'].copy()
    if df_sky.empty:
        print("No SkyMask data found for deep-dive plot.")
        return

    # Fill defaults for any missing HPs
    if 'mask_clip' not in df_sky.columns: df_sky['mask_clip'] = 1.0
    if 'mask_lr' not in df_sky.columns: df_sky['mask_lr'] = 0.01  # Assume a default
    if 'mask_epochs' not in df_sky.columns: df_sky['mask_epochs'] = 20  # Assume a default
    if 'mask_threshold' not in df_sky.columns: df_sky['mask_threshold'] = 0.5  # Assume a default
    if 'clip_norm' not in df_sky.columns: df_sky['clip_norm'] = 'None'

    # Ensure all HPs are strings for categorical plotting
    df_sky['mask_lr'] = df_sky['mask_lr'].astype(str)
    df_sky['mask_clip'] = df_sky['mask_clip'].astype(str)
    df_sky['mask_epochs'] = df_sky['mask_epochs'].astype(str)
    df_sky['mask_threshold'] = df_sky['mask_threshold'].astype(str)

    # Create a new categorical column to identify the "trick" HPs
    def categorize_hps(row):
        is_official_lr = row['mask_lr'] in ['10000000.0', '100000000.0']
        is_official_clip = row['mask_clip'] == '1e-07'

        if is_official_lr and is_official_clip:
            return "Official (Trick HPs)"
        elif is_official_lr and not is_official_clip:
            return "Partial (High LR, Normal Clip)"
        elif not is_official_lr and is_official_clip:
            return "Partial (Normal LR, Trick Clip)"
        else:
            return "Normal HPs (No Trick)"

    df_sky['hp_type'] = df_sky.apply(categorize_hps, axis=1)

    # Melt data for plotting
    plot_df = df_sky.melt(
        id_vars=['dataset', 'attack', 'hp_type', 'mask_threshold'],
        value_vars=['defense_score', 'filter_failure'],
        var_name='Metric',
        value_name='Value'
    )

    g = sns.catplot(
        data=plot_df,
        x='hp_type',
        y='Value',
        hue='mask_threshold',
        col='attack',
        row='Metric',
        kind='bar',
        palette='viridis',
        height=4,
        aspect=1.5,
        sharey='row',
        legend_out=True
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
    plt.clf()
    plt.close('all')


def plot_defense_comparison(df: pd.DataFrame, scenario: str, defense: str, output_dir: Path):
    """Generates plots for a specific scenario to compare HP settings."""
    scenario_df = df[df['scenario'] == scenario].copy()

    if scenario_df.empty:
        print(f"No data found for scenario '{scenario}'")
        return

    print(f"\n--- Visualizing Scenario: {scenario} ---")

    # --- 1. Plot for Objective 1: Best Performance Trade-off ---
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

    plot_df = scenario_df.melt(
        id_vars=['hp_label'],
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(max(16, 0.5 * scenario_df['hp_label'].nunique()), 7))
    sns.barplot(
        data=plot_df,
        x='hp_label',
        y='Value',
        hue='Metric',
        order=scenario_df['hp_label'].unique()
    )
    plt.title(f'Performance & Selection vs. HPs for {defense.upper()} (Scenario: {scenario})')
    plt.ylabel('Rate')
    plt.xlabel('Hyperparameter Combination (Sorted by Defense Score)')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.legend(title='Metric')
    plt.tight_layout()

    plot_file = output_dir / f"plot_{scenario}_performance.pdf"
    plt.savefig(plot_file, bbox_inches='tight', format='pdf')
    print(f"Saved plot: {plot_file}")
    plt.clf()
    plt.close('all')

    # --- 2. Plot for Objective 2: Stableness (Parameter Sensitivity) ---
    if len(hp_cols_to_plot) >= 2:
        x_hp = hp_cols_to_plot[0]
        scenario_df[x_hp] = scenario_df[x_hp].astype(str)
        y_hp = hp_cols_to_plot[1]
        col_hp = hp_cols_to_plot[2] if len(hp_cols_to_plot) > 2 else None
        metrics_to_plot_grid = ['defense_score', 'acc', 'asr', 'filter_failure']

        plot_df_melted = scenario_df.melt(
            id_vars=[c for c in [x_hp, y_hp, col_hp] if c is not None],
            value_vars=metrics_to_plot_grid,
            var_name='Metric',
            value_name='Value'
        )

        g = sns.catplot(
            data=plot_df_melted,
            x=x_hp,
            y='Value',
            hue=y_hp,
            col=col_hp,
            row='Metric',
            kind='bar',
            palette='viridis',
            height=3,
            aspect=1.2,
            sharey=False,
            legend_out=True
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


def main():
    output_dir = Path(FIGURE_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    df = collect_all_results(BASE_RESULTS_DIR)

    if df.empty:
        print("No results data was loaded. Exiting.")
        return

    csv_output_path = output_dir / "step3_full_results_summary.csv"
    df.to_csv(csv_output_path, index=False, float_format="%.4f")
    print(f"\n✅ Successfully saved full analysis data to: {csv_output_path}\n")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Analysis for Objective 1: Best Performance with Filtering ---
    print("\n" + "=" * 80)
    print(f" Objective 1: Finding Best HPs (Filtered by acc >= {REASONABLE_ACC_THRESHOLD})")
    print("=" * 80)

    reasonable_acc_df = df[df['acc'] >= REASONABLE_ACC_THRESHOLD].copy()

    if reasonable_acc_df.empty:
        print(f"\n!WARNING: No runs met the accuracy threshold of {REASONABLE_ACC_THRESHOLD}.")
        print("  Falling back to all runs for analysis.")
        reasonable_acc_df = df.copy()

    if 'benign_selection_rate' in reasonable_acc_df.columns:
        reasonable_final_df = reasonable_acc_df[
            reasonable_acc_df['benign_selection_rate'] >= REASONABLE_BSR_THRESHOLD
            ].copy()

        if reasonable_final_df.empty:
            print(
                f"\n!WARNING: No runs passed the BSR threshold of {REASONABLE_BSR_THRESHOLD} (after passing accuracy).")
            print("  Falling back to only accuracy-filtered runs.")
            reasonable_final_df = reasonable_acc_df.copy()
    else:
        print("\n!WARNING: 'benign_selection_rate' not found. Skipping Fairness filter.")
        reasonable_final_df = reasonable_acc_df.copy()

    reasonable_final_df['sort_metric'] = np.where(
        reasonable_final_df['attack'] == 'backdoor',
        reasonable_final_df['asr'],  # Low is good
        1.0 - reasonable_final_df['acc']  # Low is good (high acc)
    )

    sort_columns = ['scenario', 'sort_metric']
    sort_ascending = [True, True]

    if 'adv_selection_rate' in reasonable_final_df.columns:
        sort_columns.append('adv_selection_rate')
        sort_ascending.append(True)

    df_sorted = reasonable_final_df.sort_values(
        by=sort_columns,
        ascending=sort_ascending
    )

    # --- THIS IS THE START OF THE FIX ---
    print(
        f"\n--- Generating Best HP Table (acc >= {REASONABLE_ACC_THRESHOLD}, benign_select_rate >= {REASONABLE_BSR_THRESHOLD}) ---")
    print(f"--- Sorted by: 1. Best Defense (Low ASR or High ACC), 2. Low Adv. Selection ---")

    grouped = df_sorted.groupby('scenario')

    best_rows_list = []

    for name, group in grouped:
        if group.empty:
            continue

        best_row = group.iloc[0:1].copy()  # Get first row as a DataFrame

        hp_cols = ['clip_norm', 'max_k', 'mask_epochs', 'mask_lr', 'mask_threshold', 'mask_clip']
        # Find HPs that were actually tuned (more than 1 unique value in the *group*)
        hp_cols_present = [c for c in hp_cols if c in group.columns and group[c].nunique() > 1]

        if hp_cols_present:
            # Create a readable string from the HPs of the best row
            def get_hp_str(row):
                return ', '.join([f"{col.split('.')[-1]}: {row[col]}" for col in hp_cols_present])

            best_row['Tuned HPs'] = best_row.apply(get_hp_str, axis=1)
        else:
            best_row['Tuned HPs'] = 'default'

        best_rows_list.append(best_row)

    # --- Generate and save the LaTeX table ---
    if best_rows_list:
        df_best_hps = pd.concat(best_rows_list, ignore_index=True)

        cols_from_scenario = ['defense', 'attack', 'dataset']
        cols_from_hps = ['Tuned HPs']
        cols_from_metrics = ['acc', 'asr', 'benign_selection_rate', 'adv_selection_rate', 'defense_score']

        final_table_cols = cols_from_scenario + cols_from_hps + cols_from_metrics
        final_table_cols = [c for c in final_table_cols if c in df_best_hps.columns]

        df_final_table = df_best_hps[final_table_cols]

        # --- Clean up column names for LaTeX ---
        clean_col_names = {
            'defense': 'Defense',
            'attack': 'Attack',
            'dataset': 'Dataset',
            'acc': 'ACC',
            'asr': 'ASR',
            'benign_selection_rate': 'Benign Select (%)',
            'adv_selection_rate': 'Adv. Select (%)',
            'defense_score': 'Defense Score'
        }

        # Convert rates to percentage for the table
        if 'benign_selection_rate' in df_final_table.columns:
            df_final_table['benign_selection_rate'] *= 100
        if 'adv_selection_rate' in df_final_table.columns:
            df_final_table['adv_selection_rate'] *= 100

        df_final_table = df_final_table.rename(columns=clean_col_names)

        # Define the new column order for the table
        final_col_order = [
            'Defense', 'Attack', 'Dataset', 'Tuned HPs', 'ACC', 'ASR',
            'Benign Select (%)', 'Adv. Select (%)', 'Defense Score'
        ]
        # Filter to only columns that exist
        final_col_order = [c for c in final_col_order if c in df_final_table.columns]
        df_final_table = df_final_table[final_col_order]

        # Generate LaTeX string
        latex_table_str = df_final_table.to_latex(
            index=False,
            escape=False,
            float_format="%.2f",  # Use 2 decimal places for percentages
            caption="Best-case defense performance after hyperparameter tuning, filtered by usability thresholds.",
            label="tab:step3_best_hps",
            position="H"
        )

        table_output_path = output_dir / "step3_best_hps_table.tex"
        with open(table_output_path, 'w') as f:
            f.write(latex_table_str)

        print(f"\n✅ Successfully saved LaTeX table of best HPs to: {table_output_path}\n")

        print("--- Best HP Summary Table (for console) ---")
        print(df_final_table.to_string(index=False, float_format="%.2f"))

    else:
        print("\nNo rows met the filtering criteria to generate a table.")
    # --- THIS IS THE END OF THE FIX ---

    # --- Analysis for Objective 2: Stableness (Visualization) ---
    print("\n" + "=" * 80)
    print("           Objective 2: Assessing Defense Stability (Plots)")
    print("=" * 80)

    # --- Call the NEW SkyMask Plot ---
    plot_skymask_deep_dive(df, output_dir)

    # --- THIS IS THE SECOND FIX ---
    # Plot the other defenses (simplified) - now includes SkyMask
    for scenario, defense in df[['scenario', 'defense']].drop_duplicates().values:
        # The 'if defense != 'skymask'' check is now GONE.
        plot_defense_comparison(df, scenario, defense, output_dir)
    # --- END OF SECOND FIX ---

    print("\nAnalysis complete. Check 'step3_figures' folder for plots and table.")


if __name__ == "__main__":
    main()