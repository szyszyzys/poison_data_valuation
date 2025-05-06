import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any # Added Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Plotting Style ---
sns.set_theme(style="whitegrid", palette="viridis")

# --- Utility Functions (Keep as before) ---
def safe_literal_eval(val: Any) -> Any:
    if pd.isna(val) or val is None: return None
    if isinstance(val, (list, dict, tuple, int, float)): return val
    if isinstance(val, str):
        s = val.strip()
        if not s: return None
        if (s.startswith('{') and s.endswith('}')) or \
           (s.startswith('[') and s.endswith(']')) or \
           (s.startswith('(') and s.endswith(')')):
            try:
                evaluated = ast.literal_eval(s)
                return list(evaluated) if isinstance(evaluated, tuple) else evaluated
            except: return s
        try: return float(s)
        except ValueError: return s
    return val

def get_experiment_parameter(exp_name: str, pattern: str, default: Any = None, cast_type: type = str) -> Any:
    import re # Ensure re is imported if not globally
    match = re.search(pattern, exp_name)
    if match:
        try: return cast_type(match.group(1))
        except (ValueError, TypeError): return default
    return default

# --- Data Loading (Keep the version tailored to your structure) ---
def load_all_results(
    base_dir: str,
    csv_filename: str = "round_results.csv",
    objectives: Optional[List[str]] = None
) -> Dict[str, List[pd.DataFrame]]:
    # ... (Keep the implementation from the previous response) ...
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.error(f"Base results directory not found: '{base_dir}'")
        return {}

    all_results: Dict[str, List[pd.DataFrame]] = {}
    logging.info(f"Scanning for results in: {base_path}")
    if objectives:
        logging.info(f"Filtering for objectives: {objectives}")

    # 1. Iterate through objective folders directly under base_dir
    for objective_path in sorted(base_path.iterdir()):
        if not objective_path.is_dir():
            continue # Skip files, etc.

        objective_name = objective_path.name

        # 2. Filter by the provided objectives list (if any)
        if objectives and objective_name not in objectives:
            logging.debug(f"Skipping objective folder (not in filter list): {objective_name}")
            continue

        logging.info(f"Processing objective: {objective_name}")

        # 3. Iterate through experiment_id folders within the objective folder
        for exp_path in sorted(objective_path.iterdir()):
            if not exp_path.is_dir():
                continue # Skip files, etc.

            experiment_id_part = exp_path.name

            # 4. Construct the unique key for the results dictionary
            combined_exp_key = f"{objective_name}_{experiment_id_part}"

            run_dfs: List[pd.DataFrame] = []
            logging.info(f"--> Processing experiment: {experiment_id_part} (Key: {combined_exp_key})")

            # 5. Find and process run_* directories within the experiment folder
            run_dirs_found = False
            for run_dir in sorted(exp_path.glob('run_*')): # Use glob to find run directories
                if not run_dir.is_dir():
                    continue
                run_dirs_found = True

                csv_file = run_dir / csv_filename
                if not csv_file.is_file():
                    # logging.warning(f"      CSV file '{csv_filename}' not found in {run_dir}") # Less verbose
                    continue

                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        # logging.warning(f"      CSV file is empty: {csv_file}") # Less verbose
                        continue

                    # Add identifying information to the DataFrame
                    df['run_id'] = run_dir.name
                    df['experiment_setup'] = combined_exp_key # Use the combined key
                    # Optional: Store original parts if needed later
                    df['objective_name'] = objective_name
                    df['experiment_id_part'] = experiment_id_part

                    run_dfs.append(df)

                except pd.errors.EmptyDataError:
                     logging.warning(f"      CSV file is empty (pandas error): {csv_file}")
                except Exception as e:
                    logging.error(f"      Failed to load or process {csv_file}: {e}", exc_info=False) # Less verbose error

            if not run_dirs_found:
                 logging.warning(f"--> No 'run_*' directories found within experiment path: {exp_path}")


            # 6. Add the collected run DataFrames to the main dictionary
            if run_dfs:
                # Handle potential (though unlikely) key collisions by extending list
                if combined_exp_key in all_results:
                     all_results[combined_exp_key].extend(run_dfs)
                     logging.warning(f"--> Appended {len(run_dfs)} runs to existing key '{combined_exp_key}'. Check structure.")
                else:
                     all_results[combined_exp_key] = run_dfs
                logging.info(f"--> Loaded {len(run_dfs)} runs for {combined_exp_key}")
            elif run_dirs_found: # Only log if runs were expected but failed to load
                 logging.warning(f"--> No valid CSVs loaded for experiment {combined_exp_key} despite finding run directories.")


    if not all_results:
        logging.warning(f"No experiment results loaded from {base_dir}. Check structure and objective filters.")
    else:
        logging.info(f"Finished loading data for {len(all_results)} unique experiment setups.")

    return all_results


# --- Modified Preprocessing ---

def preprocess_data_simple_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing focused on selection composition metrics.
    Extracts acc, asr, and calculates benign/malicious PROPORTIONS within selected set per round.
    """
    if df.empty: return df

    logging.debug(f"Preprocessing run {df['run_id'].iloc[0]} for exp {df['experiment_setup'].iloc[0]}")

    # --- Evaluate Literal Columns (Essential Ones) ---
    literal_cols = ['selected_sellers', 'seller_ids_all', 'perf_global']
    list_cols = ['selected_sellers', 'seller_ids_all']
    dict_cols = ['perf_global']

    for col in literal_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(safe_literal_eval)

    for col in list_cols:
        if col in df.columns: df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        else: df[col] = [[] for _ in range(len(df))]

    for col in dict_cols:
        if col in df.columns: df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
        else: df[col] = [{} for _ in range(len(df))]

    # --- Extract Performance & ASR ---
    if 'perf_global' in df.columns:
        df['global_acc'] = df['perf_global'].apply(lambda d: d.get('accuracy', np.nan))
        asr_key = next((k for k in df['perf_global'].iloc[0].keys() if 'asr' in k or 'attack_success' in k), None)
        if asr_key: df['global_asr'] = df['perf_global'].apply(lambda d: d.get(asr_key, np.nan))
        else: df['global_asr'] = np.nan
    else:
        df['global_acc'] = np.nan
        df['global_asr'] = np.nan

    # --- Convert Core Numerics ---
    if 'round_number' in df.columns:
        df['round_number'] = pd.to_numeric(df['round_number'], errors='coerce').astype(pd.Int64Dtype())
    if 'num_sellers_selected' in df.columns:
         df['num_sellers_selected'] = pd.to_numeric(df['num_sellers_selected'], errors='coerce')
    else: # Add column if missing, needed for rate calculation
         if 'selected_sellers' in df.columns:
             df['num_sellers_selected'] = df['selected_sellers'].apply(len)
         else:
             df['num_sellers_selected'] = 0


    # --- Extract Parameters ---
    exp_name = df['experiment_setup'].iloc[0] if 'experiment_setup' in df.columns and len(df) > 0 else ""
    df['exp_aggregator'] = get_experiment_parameter(exp_name, r'_(fedavg|fltrust|martfl|skymask)', default='unknown', cast_type=str)
    df['exp_dataset'] = get_experiment_parameter(exp_name, r'_(cifar|fmnist|agnews|trec)', default='unknown', cast_type=str)
    df['exp_attack'] = get_experiment_parameter(exp_name, r'_(backdoor|label.?flip|sybil|mimicry|baseline|none)', default='none', cast_type=str)
    adv_rate_pct = get_experiment_parameter(exp_name, r'_adv(\d+)pct', default=None, cast_type=int)
    adv_rate_frac = get_experiment_parameter(exp_name, r'_adv(0\.\d+)', default=None, cast_type=float)
    df['exp_adv_rate'] = (adv_rate_pct / 100.0) if adv_rate_pct is not None else adv_rate_frac if adv_rate_frac is not None else 0.0

    # --- Calculate Selection Counts & **Composition Rates** ---
    seller_ids = df['seller_ids_all'].iloc[0] if len(df) > 0 and isinstance(df['seller_ids_all'].iloc[0], list) else []
    benign_ids = {s for s in seller_ids if isinstance(s, str) and s.startswith('bn_')}
    adv_ids = {s for s in seller_ids if isinstance(s, str) and s.startswith('adv_')}
    # Store these for potential reference, but not strictly needed for composition plot
    df['num_benign'] = len(benign_ids)
    df['num_malicious'] = len(adv_ids)

    if 'selected_sellers' in df.columns:
        df['benign_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in benign_ids))
        df['malicious_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in adv_ids))

        # Calculate COMPOSITION rates per round (handle division by zero if num_sellers_selected is 0)
        # This is: fraction OF THE SELECTED SET that are benign/malicious
        df['selected_comp_benign'] = (df['benign_selected_count'] / df['num_sellers_selected']).fillna(0)
        df['selected_comp_malicious'] = (df['malicious_selected_count'] / df['num_sellers_selected']).fillna(0)

    else: # Ensure columns exist
        df['benign_selected_count'] = 0
        df['malicious_selected_count'] = 0
        df['selected_comp_benign'] = 0.0
        df['selected_comp_malicious'] = 0.0

    return df

# --- Plotting Functions ---

def plot_final_metric_bar(
    summary_df: pd.DataFrame,
    metric: str, error_metric: str,
    group_title: str, plot_title: str, ylabel: str,
    higher_is_better: bool, save_path: Path
) -> None:
    # ... (Keep implementation from previous response) ...
    if summary_df.empty or metric not in summary_df.columns:
        logging.warning(f"No data or metric '{metric}' for bar plot: {plot_title}")
        return

    df_plot = summary_df.dropna(subset=[metric]).sort_values('exp_aggregator').copy()
    if df_plot.empty:
        logging.warning(f"No non-NaN data for metric '{metric}' in bar plot: {plot_title}")
        return

    errors = df_plot[error_metric] if error_metric in df_plot.columns else 0
    exp_names = df_plot['exp_aggregator'].tolist()
    means = df_plot[metric].values

    best_idx = -1
    if means.size > 0:
        try: best_idx = int(np.nanargmax(means) if higher_is_better else np.nanargmin(means))
        except ValueError: pass

    plt.figure(figsize=(max(6, len(exp_names) * 1.0), 5))
    try:
        bar_container = plt.bar(exp_names, means, yerr=errors, capsize=5,
                                color=sns.color_palette("viridis", len(exp_names)))
        if best_idx != -1 and best_idx < len(bar_container):
            bar_container[best_idx].set_color('red')
    except Exception as e:
        logging.error(f"Error creating bar plot '{plot_title}': {e}")
        plt.close()
        return

    plt.xticks(rotation=15, ha='right')
    plt.title(f"{group_title}\n{plot_title}")
    plt.ylabel(ylabel)
    plt.xlabel("Aggregation Method")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot: {save_path}")
    plt.close()


# --- MODIFIED PLOTTING FUNCTION for SELECTION COMPOSITION ---
def plot_selected_composition(
    summary_df: pd.DataFrame,
    group_title: str,
    save_path: Path
) -> None:
    """
    Generates a grouped bar plot showing the average composition
    (proportion benign vs. malicious) WITHIN the selected set of sellers.
    """
    # Use the metrics calculated in preprocessing: selected_comp_benign / selected_comp_malicious
    required_metrics = ['avg_comp_benign_mean', 'avg_comp_malicious_mean']
    if summary_df.empty or not all(m in summary_df.columns for m in required_metrics):
        logging.warning(f"Missing required composition columns for selection plot: {group_title}")
        return

    df_plot = summary_df.dropna(subset=required_metrics).sort_values('exp_aggregator').copy()
    if df_plot.empty:
        logging.warning(f"No non-NaN data for selected composition plot: {group_title}")
        return

    # --- Prepare data for grouped bar plot ---
    df_melt = pd.melt(
        df_plot,
        id_vars=['experiment_setup', 'exp_aggregator'],
        value_vars={ # Map internal name to display name
            'avg_comp_benign_mean': 'Benign Proportion',
            'avg_comp_malicious_mean': 'Malicious Proportion'
        },
        var_name='Seller Type Proportion',
        value_name='Average Proportion in Selected Set'
    )
    # Add error bars (SEM) if available
    sem_benign_col = 'avg_comp_benign_sem'
    sem_malicious_col = 'avg_comp_malicious_sem'
    if sem_benign_col in df_plot.columns and sem_malicious_col in df_plot.columns:
         df_sem_melt = pd.melt(
            df_plot,
            id_vars=['experiment_setup', 'exp_aggregator'],
            value_vars={
                 sem_benign_col: 'Benign Proportion',
                 sem_malicious_col: 'Malicious Proportion'
            },
            var_name='Seller Type Proportion', # Must match the value_vars map above
            value_name='SEM'
         )
         # Merge SEM back based on experiment and the type
         df_melt = pd.merge(df_melt, df_sem_melt, on=['experiment_setup', 'exp_aggregator', 'Seller Type Proportion'])
         use_error_bars = True
    else:
         use_error_bars = False
         df_melt['SEM'] = 0 # Add dummy column if SEM not available


    plt.figure(figsize=(max(7, len(df_plot['exp_aggregator'].unique()) * 1.2), 5))
    try:
        ax = sns.barplot(
            data=df_melt,
            x='exp_aggregator',
            y='Average Proportion in Selected Set',
            hue='Seller Type Proportion',
            palette={'Benign Proportion': 'tab:blue', 'Malicious Proportion': 'tab:red'},
            errorbar=None # Add error bars manually below if available
        )

        # --- Add error bars manually if SEM is available ---
        if use_error_bars:
            num_bars = len(ax.patches)
            num_groups = len(df_plot['exp_aggregator'].unique())
            num_hues = df_melt['Seller Type Proportion'].nunique()
            width = ax.patches[0].get_width() # Get bar width from plot
            positions = [p.get_x() + width / 2. for p in ax.patches] # Center of each bar

            # Reshape df_melt to align error bars correctly
            df_pivot = df_melt.pivot_table(index='exp_aggregator', columns='Seller Type Proportion', values='SEM').reindex(df_plot['exp_aggregator'].unique()) # Preserve order
            errors_ordered = df_pivot.values.flatten('F') # Flatten column-wise ('F') to match bar order

            ax.errorbar(positions, ax.containers[0].datavalues + ax.containers[1].datavalues, # Use bar heights
                        yerr=errors_ordered, fmt='none', c='black', capsize=4)


        plt.title(f"{group_title}\nAverage Composition of Selected Sellers")
        plt.ylabel("Avg. Proportion within Selected Set")
        plt.xlabel("Aggregation Method")
        plt.xticks(rotation=15, ha='right')
        plt.ylim(0, 1.05) # Proportion sums to 1
        plt.legend(title='Seller Type', loc='upper right')
        plt.tight_layout()

    except Exception as e:
        logging.error(f"Error creating selected composition plot '{group_title}': {e}", exc_info=True)
        plt.close()
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot: {save_path}")
    plt.close()


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    RESULTS_BASE_DIR = "./experiment_results" # MODIFY AS NEEDED
    OUTPUT_DIR = Path("./analysis_plots_composition") # MODIFY AS NEEDED
    OBJECTIVES_TO_LOAD = None # Set to None to load all, or e.g., ['attack_comparison', 'sybil_comparison']

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    raw_results = load_all_results(RESULTS_BASE_DIR, objectives=OBJECTIVES_TO_LOAD)
    if not raw_results: exit()

    # --- 2. Preprocess Data ---
    processed_results: Dict[str, List[pd.DataFrame]] = {}
    for exp_name, run_dfs in raw_results.items():
        # Use the NEW preprocessing function
        processed_dfs = [preprocess_data_simple_composition(df.copy()) for df in run_dfs if not df.empty]
        if processed_dfs: processed_results[exp_name] = processed_dfs

    if not processed_results:
        logging.error("Preprocessing failed for all loaded data.")
        exit()

    # --- 3. Calculate Run Summaries & Aggregate ---
    run_summaries = []
    for exp_name, run_list in processed_results.items():
        for i, df_run in enumerate(run_list):
            if df_run.empty: continue
            final_acc = df_run.loc[df_run['round_number'].idxmax(), 'global_acc'] if 'global_acc' in df_run.columns and not df_run['global_acc'].isnull().all() else np.nan
            final_asr = df_run.loc[df_run['round_number'].idxmax(), 'global_asr'] if 'global_asr' in df_run.columns and not df_run['global_asr'].isnull().all() else np.nan

            # Calculate average COMPOSITION rates over the run
            avg_comp_benign = df_run['selected_comp_benign'].mean() if 'selected_comp_benign' in df_run.columns else np.nan
            avg_comp_malicious = df_run['selected_comp_malicious'].mean() if 'selected_comp_malicious' in df_run.columns else np.nan

            run_summary = {
                'experiment_setup': exp_name,
                'run_id': df_run['run_id'].iloc[0],
                # Include parameters for grouping
                'objective_name': df_run['objective_name'].iloc[0],
                'experiment_id_part': df_run['experiment_id_part'].iloc[0],
                'exp_aggregator': df_run['exp_aggregator'].iloc[0],
                'exp_dataset': df_run['exp_dataset'].iloc[0],
                'exp_attack': df_run['exp_attack'].iloc[0],
                'exp_adv_rate': df_run['exp_adv_rate'].iloc[0],
                # Metrics
                'final_acc': final_acc,
                'final_asr': final_asr,
                'avg_comp_benign': avg_comp_benign,     # Average proportion benign in selected set
                'avg_comp_malicious': avg_comp_malicious, # Average proportion malicious in selected set
            }
            run_summaries.append(run_summary)

    if not run_summaries:
        logging.error("No valid run summaries could be calculated.")
        exit()

    # Aggregate results across runs
    summary_df = pd.DataFrame(run_summaries)
    agg_metrics = {
        'final_acc': ['mean', 'std', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0],
        'final_asr': ['mean', 'std', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0],
        'avg_comp_benign': ['mean', 'std', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0],
        'avg_comp_malicious': ['mean', 'std', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0],
    }
    # Rename SEM function for clarity
    for key in agg_metrics: agg_metrics[key][2].__name__ = 'sem'

    final_summary_df = summary_df.groupby('experiment_setup').agg(agg_metrics).reset_index()
    final_summary_df.columns = ['_'.join(col).strip('_') for col in final_summary_df.columns.values]
    params_df = summary_df[['experiment_setup', 'objective_name', 'experiment_id_part', 'exp_aggregator', 'exp_dataset', 'exp_attack', 'exp_adv_rate']].drop_duplicates()
    final_summary_df = pd.merge(final_summary_df, params_df, on='experiment_setup', how='left')

    summary_csv_path = OUTPUT_DIR / "composition_metrics_summary.csv"
    final_summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Composition summary saved to: {summary_csv_path}")

    # --- 4. Define Plotting Groups ---
    grouping_cols = ['exp_dataset', 'exp_attack', 'exp_adv_rate']
    plot_data = final_summary_df

    # --- 5. Generate Focused Plots for each Group ---
    for name, group_df in plot_data.groupby(grouping_cols):
        dataset, attack, adv_rate = name
        if dataset == 'unknown' or attack == 'unknown': continue

        adv_rate_pct = int(adv_rate * 100)
        group_title = f"{dataset.upper()} / {attack.capitalize()} / {adv_rate_pct}% Adversaries"
        group_save_prefix = f"{dataset.lower()}_{attack.lower()}_adv{adv_rate_pct}pct"
        group_output_dir = OUTPUT_DIR / group_save_prefix
        group_output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"--- Generating plots for group: {group_title} ---")

        # Plot 1: Final Accuracy
        plot_final_metric_bar(
            summary_df=group_df,
            metric='final_acc_mean', error_metric='final_acc_sem',
            group_title=group_title, plot_title="Final Global Accuracy",
            ylabel="Accuracy", higher_is_better=True,
            save_path=group_output_dir / "final_accuracy.png"
        )

        # Plot 2: Final ASR (if relevant)
        if attack not in ['none', 'baseline'] and not group_df['final_asr_mean'].isnull().all():
            plot_final_metric_bar(
                summary_df=group_df,
                metric='final_asr_mean', error_metric='final_asr_sem',
                group_title=group_title, plot_title="Final Attack Success Rate (ASR)",
                ylabel="ASR", higher_is_better=False,
                save_path=group_output_dir / "final_asr.png"
            )

        # Plot 3: Selection COMPOSITION (only if malicious sellers might exist)
        # Note: This plot is meaningful even if adv_rate is 0, showing 100% benign composition.
        # However, it's most *interesting* when adv_rate > 0.
        # Let's plot it unless both composition means are missing.
        if not group_df['avg_comp_benign_mean'].isnull().all() or not group_df['avg_comp_malicious_mean'].isnull().all():
             plot_selected_composition( # Use the NEW plot function
                 summary_df=group_df,
                 group_title=group_title,
                 save_path=group_output_dir / "avg_selected_composition.png" # New filename
             )

        # --- Optional Scatter Plots (Keep as before, using appropriate metrics) ---
        # Acc vs ASR
        if attack not in ['none', 'baseline'] and not group_df['final_asr_mean'].isnull().all() and not group_df['final_acc_mean'].isnull().all():
             plt.figure(figsize=(7, 7))
             try:
                sns.scatterplot(data=group_df, x='final_acc_mean', y='final_asr_mean', hue='exp_aggregator', s=100)
                plt.title(f"{group_title}\nTrade-off: Accuracy vs ASR")
                plt.xlabel("Final Accuracy (Mean)")
                plt.ylabel("Final ASR (Mean)")
                plt.legend(title="Aggregator", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                s_path = group_output_dir / "scatter_acc_vs_asr.png"
                plt.savefig(s_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved plot: {s_path}")
             except Exception as e: logging.error(f"Failed scatter plot Acc vs ASR for {group_title}: {e}")
             finally: plt.close()

        # Acc vs Malicious Composition
        if not group_df['avg_comp_malicious_mean'].isnull().all() and not group_df['final_acc_mean'].isnull().all():
             plt.figure(figsize=(7, 7))
             try:
                 sns.scatterplot(data=group_df, x='final_acc_mean', y='avg_comp_malicious_mean', hue='exp_aggregator', s=100)
                 plt.title(f"{group_title}\nTrade-off: Accuracy vs Malicious Proportion Selected")
                 plt.xlabel("Final Accuracy (Mean)")
                 plt.ylabel("Avg. Malicious Proportion in Selected Set")
                 plt.legend(title="Aggregator", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                 plt.tight_layout(rect=[0, 0, 0.85, 1])
                 s_path = group_output_dir / "scatter_acc_vs_mal_comp.png"
                 plt.savefig(s_path, dpi=300, bbox_inches='tight')
                 logging.info(f"Saved plot: {s_path}")
             except Exception as e: logging.error(f"Failed scatter plot Acc vs MalComp for {group_title}: {e}")
             finally: plt.close()


    logging.info(f"--- Composition analysis complete. Plots saved in: {OUTPUT_DIR} ---")