import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

# --- Utility Functions ---

def safe_literal_eval(val: Any) -> Any:
    """Safely evaluate strings that look like Python literals."""
    # Simplified version for brevity, assuming the previous one works
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
            except: return s # Fallback to original string on error
        try: return float(s)
        except ValueError: return s
    return val

def get_experiment_parameter(exp_name: str, pattern: str, default: Any = None, cast_type: type = str) -> Any:
    """Extracts a parameter value from an experiment name string using regex."""
    match = re.search(pattern, exp_name)
    if match:
        try: return cast_type(match.group(1))
        except (ValueError, TypeError): return default
    return default

# --- Data Loading (Using the function you provided) ---

def load_all_results(
    base_dir: str,
    csv_filename: str = "round_results.csv",
    objectives: Optional[List[str]] = None # Optional list of objective folders to load
) -> Dict[str, List[pd.DataFrame]]:
    """
    Load results structured as base_dir/<objective>/<experiment_id>/run_*/csv_filename.
    (Code identical to the version you provided in the previous turn)
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.error(f"Base results directory not found: '{base_dir}'")
        return {}

    all_results: Dict[str, List[pd.DataFrame]] = {}
    logging.info(f"Scanning for results in: {base_path}")
    if objectives:
        logging.info(f"Filtering for objectives: {objectives}")

    for objective_path in sorted(base_path.iterdir()):
        if not objective_path.is_dir(): continue
        objective_name = objective_path.name
        if objectives and objective_name not in objectives: continue

        logging.info(f"Processing objective: {objective_name}")
        for exp_path in sorted(objective_path.iterdir()):
            if not exp_path.is_dir(): continue
            experiment_id_part = exp_path.name
            combined_exp_key = f"{objective_name}_{experiment_id_part}"
            run_dfs: List[pd.DataFrame] = []
            logging.info(f"--> Processing experiment: {experiment_id_part} (Key: {combined_exp_key})")
            run_dirs_found = False
            for run_dir in sorted(exp_path.glob('run_*')):
                if not run_dir.is_dir(): continue
                run_dirs_found = True
                csv_file = run_dir / csv_filename
                if not csv_file.is_file(): continue
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty: continue
                    df['run_id'] = run_dir.name
                    df['experiment_setup'] = combined_exp_key
                    df['objective_name'] = objective_name
                    df['experiment_id_part'] = experiment_id_part
                    run_dfs.append(df)
                except Exception as e:
                    logging.error(f"      Failed to load or process {csv_file}: {e}", exc_info=False) # Less verbose error

            if not run_dirs_found:
                 logging.warning(f"--> No 'run_*' directories found within experiment path: {exp_path}")

            if run_dfs:
                if combined_exp_key in all_results:
                     all_results[combined_exp_key].extend(run_dfs)
                else:
                     all_results[combined_exp_key] = run_dfs
                logging.info(f"--> Loaded {len(run_dfs)} runs for {combined_exp_key}")
            elif run_dirs_found:
                 logging.warning(f"--> No valid CSVs loaded for experiment {combined_exp_key}")

    if not all_results:
        logging.warning(f"No experiment results loaded from {base_dir}. Check structure and objective filters.")
    else:
        logging.info(f"Finished loading data for {len(all_results)} unique experiment setups.")
    return all_results


# --- Simplified Preprocessing ---

def preprocess_data_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preprocessing to extract key metrics and parameters for focused analysis.
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
        # Find ASR key flexibly
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


    # --- Extract Parameters from Experiment Name ---
    exp_name = df['experiment_setup'].iloc[0] if 'experiment_setup' in df.columns and len(df) > 0 else ""
    df['exp_aggregator'] = get_experiment_parameter(exp_name, r'_(fedavg|fltrust|martfl|skymask)', default='unknown', cast_type=str)
    df['exp_dataset'] = get_experiment_parameter(exp_name, r'_(cifar|fmnist|agnews|trec)', default='unknown', cast_type=str)
    df['exp_attack'] = get_experiment_parameter(exp_name, r'_(backdoor|label.?flip|sybil|mimicry|baseline|none)', default='none', cast_type=str) # Include baseline/none
     # Handle different adv rate formats (_adv10pct or _adv0.1)
    adv_rate_pct = get_experiment_parameter(exp_name, r'_adv(\d+)pct', default=None, cast_type=int)
    adv_rate_frac = get_experiment_parameter(exp_name, r'_adv(0\.\d+)', default=None, cast_type=float)
    df['exp_adv_rate'] = (adv_rate_pct / 100.0) if adv_rate_pct is not None else adv_rate_frac if adv_rate_frac is not None else 0.0


    # --- Calculate Seller Counts/Rates ---
    seller_ids = df['seller_ids_all'].iloc[0] if len(df) > 0 and isinstance(df['seller_ids_all'].iloc[0], list) else []
    benign_ids = {s for s in seller_ids if isinstance(s, str) and s.startswith('bn_')}
    adv_ids = {s for s in seller_ids if isinstance(s, str) and s.startswith('adv_')}
    df['num_benign'] = len(benign_ids)
    df['num_malicious'] = len(adv_ids)

    if 'selected_sellers' in df.columns:
        if 'num_sellers_selected' not in df.columns:
             df['num_sellers_selected'] = df['selected_sellers'].apply(len)

        df['benign_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in benign_ids))
        df['malicious_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in adv_ids))

        # Calculate rates per round, handle division by zero
        # Rate relative to *total selected* in that round
        df['benign_selection_rate_round'] = (df['benign_selected_count'] / df['num_sellers_selected']).fillna(0)
        df['malicious_selection_rate_round'] = (df['malicious_selected_count'] / df['num_sellers_selected']).fillna(0)
        # Rate relative to *total available* of that type (might be more informative for fairness)
        df['benign_selection_rate_available'] = (df['benign_selected_count'] / df['num_benign']).fillna(0) if df['num_benign'].iloc[0] > 0 else 0.0
        df['malicious_selection_rate_available'] = (df['malicious_selected_count'] / df['num_malicious']).fillna(0) if df['num_malicious'].iloc[0] > 0 else 0.0

    else: # Ensure columns exist
        df['benign_selected_count'] = 0
        df['malicious_selected_count'] = 0
        df['benign_selection_rate_round'] = 0.0
        df['malicious_selection_rate_round'] = 0.0
        df['benign_selection_rate_available'] = 0.0
        df['malicious_selection_rate_available'] = 0.0
        if 'num_sellers_selected' not in df.columns: df['num_sellers_selected'] = 0

    return df


# --- Plotting Functions ---

def plot_final_metric_bar(
    summary_df: pd.DataFrame,
    metric: str, # e.g., 'final_acc_mean'
    error_metric: str, # e.g., 'final_acc_sem'
    group_title: str,
    plot_title: str,
    ylabel: str,
    higher_is_better: bool,
    save_path: Path
) -> None:
    """Generates a bar plot for a final metric from the summary DataFrame."""
    if summary_df.empty or metric not in summary_df.columns:
        logging.warning(f"No data or metric '{metric}' for bar plot: {plot_title}")
        return

    df_plot = summary_df.dropna(subset=[metric]).sort_values('exp_aggregator').copy()
    if df_plot.empty:
        logging.warning(f"No non-NaN data for metric '{metric}' in bar plot: {plot_title}")
        return

    # Use pre-calculated SEM or default to 0 if error metric doesn't exist
    errors = df_plot[error_metric] if error_metric in df_plot.columns else 0

    exp_names = df_plot['exp_aggregator'].tolist() # X-axis is the aggregator
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
            # Simple annotation
            # plt.text(best_idx, means[best_idx] + errors.iloc[best_idx] * 1.1, 'Best',
            #          ha='center', va='bottom', color='red', fontsize=9)

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


def plot_selection_rates_comparison(
    summary_df: pd.DataFrame,
    group_title: str,
    save_path: Path
) -> None:
    """
    Generates a grouped bar plot comparing average benign vs malicious selection rates.
    Uses the rates calculated relative to the number available of each type.
    """
    required_metrics = ['avg_benign_rate_avail_mean', 'avg_malicious_rate_avail_mean']
    if summary_df.empty or not all(m in summary_df.columns for m in required_metrics):
        logging.warning(f"Missing required rate columns for selection plot: {group_title}")
        return

    df_plot = summary_df.dropna(subset=required_metrics).sort_values('exp_aggregator').copy()
    if df_plot.empty:
        logging.warning(f"No non-NaN data for selection rate plot: {group_title}")
        return

    # --- Prepare data for grouped bar plot ---
    df_melt = pd.melt(
        df_plot,
        id_vars=['experiment_setup', 'exp_aggregator'],
        value_vars={ # Map internal name to display name
            'avg_benign_rate_avail_mean': 'Benign',
            'avg_malicious_rate_avail_mean': 'Malicious'
        },
        var_name='Selection Type',
        value_name='Average Selection Rate'
    )
    # Include error bars (SEM) - requires melting std/sem columns similarly if available
    # For simplicity, we omit error bars here, but they can be added
    # Example: Add avg_benign_rate_avail_sem, avg_malicious_rate_avail_sem to summary
    # Melt them too, then use errorbar argument or ax.errorbar in sns.barplot

    plt.figure(figsize=(max(7, len(df_plot['exp_aggregator'].unique()) * 1.2), 5))
    try:
        ax = sns.barplot(
            data=df_melt,
            x='exp_aggregator',
            y='Average Selection Rate',
            hue='Selection Type',
            palette={'Benign': 'tab:blue', 'Malicious': 'tab:red'},
            # errorbar=None # Add error bars here if SEM is available
        )
        plt.title(f"{group_title}\nAverage Selection Rate (Relative to Available Sellers)")
        plt.ylabel("Avg. Fraction Selected per Round")
        plt.xlabel("Aggregation Method")
        plt.xticks(rotation=15, ha='right')
        plt.ylim(0, 1.05) # Rate should be between 0 and 1
        plt.legend(title='Seller Type', loc='upper right')
        plt.tight_layout()

    except Exception as e:
        logging.error(f"Error creating selection rate plot '{group_title}': {e}")
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
    OUTPUT_DIR = Path("./analysis_plots_simplified") # MODIFY AS NEEDED
    OBJECTIVES_TO_LOAD = None # Set to None to load all, or e.g., ['attack_comparison', 'sybil_comparison']

    # --- 1. Load Data ---
    raw_results = load_all_results(RESULTS_BASE_DIR, objectives=OBJECTIVES_TO_LOAD)
    if not raw_results: exit()

    # --- 2. Preprocess Data ---
    processed_results: Dict[str, List[pd.DataFrame]] = {}
    for exp_name, run_dfs in raw_results.items():
        processed_dfs = [preprocess_data_simple(df.copy()) for df in run_dfs if not df.empty]
        if processed_dfs: processed_results[exp_name] = processed_dfs

    if not processed_results:
        logging.error("Preprocessing failed for all loaded data.")
        exit()

    # --- 3. Calculate Run Summaries & Aggregate ---
    run_summaries = []
    for exp_name, run_list in processed_results.items():
        for i, df_run in enumerate(run_list):
            if df_run.empty: continue
            # Calculate final metrics
            final_acc = df_run.loc[df_run['round_number'].idxmax(), 'global_acc'] if 'global_acc' in df_run.columns and not df_run['global_acc'].isnull().all() else np.nan
            final_asr = df_run.loc[df_run['round_number'].idxmax(), 'global_asr'] if 'global_asr' in df_run.columns and not df_run['global_asr'].isnull().all() else np.nan

            # Calculate average selection rates over the run (relative to available)
            avg_benign_rate_avail = df_run['benign_selection_rate_available'].mean() if 'benign_selection_rate_available' in df_run.columns else np.nan
            avg_malicious_rate_avail = df_run['malicious_selection_rate_available'].mean() if 'malicious_selection_rate_available' in df_run.columns else np.nan

            run_summary = {
                'experiment_setup': exp_name,
                'run_id': df_run['run_id'].iloc[0],
                'objective_name': df_run['objective_name'].iloc[0],
                'experiment_id_part': df_run['experiment_id_part'].iloc[0],
                'exp_aggregator': df_run['exp_aggregator'].iloc[0],
                'exp_dataset': df_run['exp_dataset'].iloc[0],
                'exp_attack': df_run['exp_attack'].iloc[0],
                'exp_adv_rate': df_run['exp_adv_rate'].iloc[0],
                'final_acc': final_acc,
                'final_asr': final_asr,
                'avg_benign_rate_avail': avg_benign_rate_avail,
                'avg_malicious_rate_avail': avg_malicious_rate_avail,
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
        'avg_benign_rate_avail': ['mean', 'std', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0],
        'avg_malicious_rate_avail': ['mean', 'std', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0],
    }
    # Rename SEM function for clarity in column names
    agg_metrics['final_acc'][2].__name__ = 'sem'
    agg_metrics['final_asr'][2].__name__ = 'sem'
    agg_metrics['avg_benign_rate_avail'][2].__name__ = 'sem'
    agg_metrics['avg_malicious_rate_avail'][2].__name__ = 'sem'

    final_summary_df = summary_df.groupby('experiment_setup').agg(agg_metrics).reset_index()
    # Flatten multi-index columns
    final_summary_df.columns = ['_'.join(col).strip('_') for col in final_summary_df.columns.values]
    # Merge back parameters for filtering/grouping plots
    params_df = summary_df[['experiment_setup', 'objective_name', 'experiment_id_part', 'exp_aggregator', 'exp_dataset', 'exp_attack', 'exp_adv_rate']].drop_duplicates()
    final_summary_df = pd.merge(final_summary_df, params_df, on='experiment_setup', how='left')

    # Save final summary
    summary_csv_path = OUTPUT_DIR / "focused_metrics_summary.csv"
    final_summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Focused summary saved to: {summary_csv_path}")

    # --- 4. Define Plotting Groups ---
    # Group by Dataset, Attack Type, and Adversary Rate for direct comparison of aggregators
    grouping_cols = ['exp_dataset', 'exp_attack', 'exp_adv_rate']
    # Filter out baseline comparisons where adv_rate is 0 if we only want attack scenarios
    # plot_data = final_summary_df[final_summary_df['exp_adv_rate'] > 0]
    plot_data = final_summary_df # Include baselines if desired

    # --- 5. Generate Focused Plots for each Group ---
    for name, group_df in plot_data.groupby(grouping_cols):
        dataset, attack, adv_rate = name
        if dataset == 'unknown' or attack == 'unknown': continue # Skip poorly parsed experiments

        # Create a readable title for the group
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

        # Plot 2: Final ASR (only if attack is not 'none'/'baseline' and ASR data exists)
        if attack not in ['none', 'baseline'] and not group_df['final_asr_mean'].isnull().all():
            plot_final_metric_bar(
                summary_df=group_df,
                metric='final_asr_mean', error_metric='final_asr_sem',
                group_title=group_title, plot_title="Final Attack Success Rate (ASR)",
                ylabel="ASR", higher_is_better=False,
                save_path=group_output_dir / "final_asr.png"
            )

        # Plot 3: Selection Rates (only if malicious sellers exist)
        if adv_rate > 0 and not group_df['avg_malicious_rate_avail_mean'].isnull().all():
             plot_selection_rates_comparison(
                 summary_df=group_df,
                 group_title=group_title,
                 save_path=group_output_dir / "avg_selection_rates.png"
             )
        elif adv_rate == 0:
             logging.info(f"Skipping selection rate plot for baseline group: {group_title}")


        # Optional: Scatter Plots within the group (Acc vs ASR)
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
             except Exception as e:
                 logging.error(f"Failed scatter plot Acc vs ASR for {group_title}: {e}")
             finally:
                 plt.close()

        # Optional: Scatter plot Acc vs Malicious Selection Rate
        if adv_rate > 0 and not group_df['avg_malicious_rate_avail_mean'].isnull().all() and not group_df['final_acc_mean'].isnull().all():
             plt.figure(figsize=(7, 7))
             try:
                 sns.scatterplot(data=group_df, x='final_acc_mean', y='avg_malicious_rate_avail_mean', hue='exp_aggregator', s=100)
                 plt.title(f"{group_title}\nTrade-off: Accuracy vs Malicious Selection Rate")
                 plt.xlabel("Final Accuracy (Mean)")
                 plt.ylabel("Avg. Malicious Selection Rate")
                 plt.legend(title="Aggregator", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                 plt.tight_layout(rect=[0, 0, 0.85, 1])
                 s_path = group_output_dir / "scatter_acc_vs_mal_rate.png"
                 plt.savefig(s_path, dpi=300, bbox_inches='tight')
                 logging.info(f"Saved plot: {s_path}")
             except Exception as e:
                 logging.error(f"Failed scatter plot Acc vs MalRate for {group_title}: {e}")
             finally:
                 plt.close()


    logging.info(f"--- Simplified analysis complete. Plots saved in: {OUTPUT_DIR} ---")