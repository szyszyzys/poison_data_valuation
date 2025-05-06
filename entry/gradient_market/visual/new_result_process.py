import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt # Keep for potential type hints if needed later
import numpy as np
import pandas as pd
import seaborn as sns # Keep for potential type hints if needed later
import yaml # For loading YAML configs

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Utility Functions ---

def safe_literal_eval(val: Any) -> Any:
    """Safely evaluate strings that look like Python literals."""
    if pd.isna(val) or val is None: return None
    if isinstance(val, (list, dict, tuple, int, float)): return val
    if isinstance(val, str):
        s = val.strip()
        if not s: return None
        # Basic check for list/dict strings
        if (s.startswith('[') and s.endswith(']')) or \
           (s.startswith('{') and s.endswith('}')):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                logging.warning(f"Failed to literal_eval string: {s[:50]}...")
                return s # Fallback to original string on error
        # Try float conversion for simple strings
        try: return float(s)
        except ValueError: return s # Return original string if not float/literal
    return val

# --- Data Loading Functions ---

def load_configs(config_base_dir: str) -> Dict[str, Dict]:
    """
    Loads all experiment configurations from YAML files structured as
    config_base_dir/<objective>/<experiment_id>.yaml.

    Returns:
        A dictionary mapping combined keys '<objective>_<experiment_id>'
        to their corresponding configuration dictionaries.
    """
    config_base_path = Path(config_base_dir)
    if not config_base_path.is_dir():
        logging.error(f"Configuration base directory not found: '{config_base_dir}'")
        return {}

    config_data: Dict[str, Dict] = {}
    logging.info(f"Scanning for configuration YAML files in: {config_base_path}")

    # Iterate through objective folders
    for objective_path in sorted(config_base_path.iterdir()):
        if not objective_path.is_dir(): continue
        objective_name = objective_path.name

        # Iterate through yaml files (experiment configs) in the objective folder
        for config_file_path in sorted(objective_path.glob('*.yaml')):
            if not config_file_path.is_file(): continue

            experiment_id_part = config_file_path.stem # Get filename without extension
            combined_exp_key = f"{objective_name}_{experiment_id_part}"

            try:
                with open(config_file_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                if config_dict: # Ensure config is not empty
                    # Add identifiers back into the dict for easier access later if needed
                    config_dict['_objective_name'] = objective_name
                    config_dict['_experiment_id_part'] = experiment_id_part
                    config_dict['_combined_exp_key'] = combined_exp_key
                    config_data[combined_exp_key] = config_dict
                else:
                    logging.warning(f"Loaded empty configuration from: {config_file_path}")

            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file {config_file_path}: {e}")
            except Exception as e:
                logging.error(f"Error reading config file {config_file_path}: {e}")

    if not config_data:
        logging.warning(f"No configuration files found or loaded from {config_base_dir}")
    else:
        logging.info(f"Loaded {len(config_data)} configurations.")

    return config_data


def load_results_data(
    results_base_dir: str,
    csv_filename: str = "round_results.csv",
    objectives: Optional[List[str]] = None
) -> Dict[str, List[pd.DataFrame]]:
    """
    Loads raw experiment results from CSV files structured as
    results_base_dir/<objective>/<experiment_id>/run_*/csv_filename.
    (Using the previously validated loading logic)
    """
    base_path = Path(results_base_dir)
    if not base_path.is_dir():
        logging.error(f"Results base directory not found: '{results_base_dir}'")
        return {}

    all_results: Dict[str, List[pd.DataFrame]] = {}
    logging.info(f"Scanning for results CSV files in: {base_path}")
    if objectives: logging.info(f"Filtering for objectives: {objectives}")

    for objective_path in sorted(base_path.iterdir()):
        if not objective_path.is_dir(): continue
        objective_name = objective_path.name
        if objectives and objective_name not in objectives: continue

        # logging.debug(f"Processing objective results: {objective_name}") # Make less verbose
        for exp_path in sorted(objective_path.iterdir()):
            if not exp_path.is_dir(): continue
            experiment_id_part = exp_path.name
            combined_exp_key = f"{objective_name}_{experiment_id_part}"
            run_dfs: List[pd.DataFrame] = []
            # logging.debug(f"--> Processing experiment results: {experiment_id_part} (Key: {combined_exp_key})")

            run_dirs = list(exp_path.glob('run_*'))
            if not run_dirs:
                 # logging.warning(f"--> No 'run_*' directories found in results path: {exp_path}") # Less verbose
                 continue

            for run_dir in sorted(run_dirs):
                if not run_dir.is_dir(): continue
                csv_file = run_dir / csv_filename
                if not csv_file.is_file(): continue
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty: continue
                    df['run_id'] = run_dir.name
                    # Store the key used to link with config
                    df['_combined_exp_key'] = combined_exp_key
                    run_dfs.append(df)
                except Exception as e:
                    logging.error(f"      Failed to load/process results {csv_file}: {e}", exc_info=False)

            if run_dfs:
                if combined_exp_key in all_results:
                     all_results[combined_exp_key].extend(run_dfs)
                else:
                     all_results[combined_exp_key] = run_dfs
                # logging.info(f"--> Loaded {len(run_dfs)} runs results for {combined_exp_key}") # Less verbose


    if not all_results:
        logging.warning(f"No results data loaded from {results_base_dir}. Check structure/filters.")
    else:
        logging.info(f"Loaded results data for {len(all_results)} unique experiment setups.")
    return all_results


# --- Preprocessing & Integration ---

def preprocess_and_integrate(df: pd.DataFrame, config: Dict) -> Optional[pd.DataFrame]:
    """
    Preprocesses a single run's DataFrame and integrates key parameters from its config.

    Args:
        df: The DataFrame for one run, loaded from round_results.csv.
        config: The corresponding configuration dictionary for this experiment.

    Returns:
        The preprocessed DataFrame with added config parameters, or None if processing fails.
    """
    if df.empty or not config:
        return None

    df = df.copy() # Avoid modifying original DataFrame

    # --- 1. Rename CSV columns ---
    rename_map = {
        'perf_global_accuracy': 'global_acc',
        'perf_global_loss': 'global_loss',
        'perf_global_attack_success_rate': 'global_asr',
        'selection_rate_info_detection_rate (TPR)': 'tpr',
        'selection_rate_info_false_positive_rate (FPR)': 'fpr',
        'num_sellers_selected': 'num_selected',
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # --- 2. Ensure Core Columns Exist and Have Correct Types ---
    numeric_cols = ['global_acc', 'global_loss', 'global_asr', 'tpr', 'fpr', 'num_selected', 'round_number']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    if 'round_number' in df.columns:
        df['round_number'] = df['round_number'].astype(pd.Int64Dtype())

    # --- 3. Parse List Columns ---
    if 'selected_sellers' in df.columns and df['selected_sellers'].dtype == 'object':
        df['selected_sellers'] = df['selected_sellers'].apply(safe_literal_eval)
        df['selected_sellers'] = df['selected_sellers'].apply(lambda x: x if isinstance(x, list) else [])
    elif 'selected_sellers' not in df.columns:
        df['selected_sellers'] = [[] for _ in range(len(df))]

    # --- 4. Calculate Selection Composition ---
    def get_composition(selected_list):
        if not isinstance(selected_list, list) or not selected_list:
            return 0, 0
        benign_count = sum(1 for s in selected_list if isinstance(s, str) and s.startswith('bn_'))
        malicious_count = sum(1 for s in selected_list if isinstance(s, str) and s.startswith('adv_'))
        return benign_count, malicious_count

    counts = df['selected_sellers'].apply(get_composition)
    df['benign_selected_count'] = counts.apply(lambda x: x[0])
    df['malicious_selected_count'] = counts.apply(lambda x: x[1])

    # Recalculate num_selected if it wasn't present or reliable
    if 'num_selected' not in df.columns or df['num_selected'].isnull().any():
         df['num_selected'] = df['benign_selected_count'] + df['malicious_selected_count']
         # Ensure it's numeric after calculation
         df['num_selected'] = pd.to_numeric(df['num_selected'], errors='coerce')


    # Calculate composition proportions (handle division by zero)
    df['selected_comp_benign'] = (df['benign_selected_count'] / df['num_selected']).fillna(0)
    df['selected_comp_malicious'] = (df['malicious_selected_count'] / df['num_selected']).fillna(0)


    # --- 5. Integrate Key Config Parameters ---
    df['exp_aggregator'] = config.get('aggregation_method', 'unknown')
    df['exp_dataset'] = config.get('dataset_name', 'unknown')
    df['exp_adv_rate'] = config.get('data_split', {}).get('adv_rate', 0.0)

    attack_config = config.get('attack', {})
    df['exp_attack_enabled'] = attack_config.get('enabled', False)
    # Get attack type, prioritize 'attack_type' if present
    attack_type = attack_config.get('attack_type')
    # If 'attack_type' is missing or None, check 'scenario' as a fallback
    if attack_type is None:
        attack_type = attack_config.get('scenario', 'none') # Default to 'none' if neither is present
    df['exp_attack_type'] = attack_type if attack_type else 'none' # Ensure it's not None

    # *** FIXED LINE HERE ***
    # Check the value from the *first row* of the Series
    if not df.empty and df['exp_attack_type'].iloc[0] == 'none' and attack_config.get('scenario') == 'backdoor':
        # If the primary check resulted in 'none' but the older 'scenario' key exists and is 'backdoor',
        # update the 'exp_attack_type' for all rows. This handles potential inconsistencies
        # in older config files where only 'scenario' might have been set.
        logging.debug(f"Correcting attack type based on fallback 'scenario' key for {config.get('_combined_exp_key')}")
        df['exp_attack_type'] = 'backdoor'
    # *** END FIX ***

    data_split_config = config.get('data_split', {})
    df['exp_split_mode'] = data_split_config.get('data_split_mode', 'unknown')
    if not df.empty and df['exp_split_mode'].iloc[0] == 'discovery':
        dm_params = data_split_config.get('dm_params', {})
        df['exp_discovery_quality'] = dm_params.get('discovery_quality', np.nan)
        df['exp_buyer_mode'] = dm_params.get('buyer_data_mode', 'unknown')
    else:
        df['exp_discovery_quality'] = np.nan
        df['exp_buyer_mode'] = 'N/A'

    sybil_config = config.get('sybil', {})
    df['exp_sybil_enabled'] = sybil_config.get('is_sybil', False)

    df['objective_name'] = config.get('_objective_name', 'unknown')
    df['experiment_id_part'] = config.get('_experiment_id_part', 'unknown')
    df['combined_exp_key'] = config.get('_combined_exp_key', 'unknown')

    return df

# --- Statistics Aggregation ---

def calculate_aggregated_stats(processed_run_dfs: List[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Calculates aggregated statistics (mean, std, sem) across multiple runs
    of a single experiment setup.

    Args:
        processed_run_dfs: A list of preprocessed DataFrames, one for each run.

    Returns:
        A dictionary containing the aggregated statistics for the experiment setup,
        or None if input is empty or invalid.
    """
    if not processed_run_dfs:
        return None

    run_summaries = []
    # --- 1. Calculate summary metrics for EACH run ---
    for df_run in processed_run_dfs:
        if df_run.empty: continue

        # Get metrics from the last round
        last_round = df_run.loc[df_run['round_number'].idxmax()] if not df_run['round_number'].isnull().all() else pd.Series(dtype=float)

        # Average metrics over the entire run
        avg_metrics = df_run.mean(numeric_only=True) # Calculate mean for all numeric columns

        run_summary = {
            'final_acc': last_round.get('global_acc', np.nan),
            'final_asr': last_round.get('global_asr', np.nan),
            'avg_comp_benign': avg_metrics.get('selected_comp_benign', np.nan),
            'avg_comp_malicious': avg_metrics.get('selected_comp_malicious', np.nan),
            'avg_tpr': avg_metrics.get('tpr', np.nan),
            'avg_fpr': avg_metrics.get('fpr', np.nan),
            'avg_num_selected': avg_metrics.get('num_selected', np.nan),
            # Add other final/average metrics if needed
        }
        run_summaries.append(run_summary)

    if not run_summaries:
        logging.warning("No valid run summaries generated during aggregation.")
        return None

    # --- 2. Aggregate across runs ---
    summary_across_runs_df = pd.DataFrame(run_summaries)
    aggregated_stats = {}
    num_runs = len(summary_across_runs_df)
    aggregated_stats['num_runs'] = num_runs

    for metric in summary_across_runs_df.columns:
        values = summary_across_runs_df[metric].dropna() # Drop NaNs for calculations
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std(ddof=1) if len(values) > 1 else 0.0
            sem_val = std_val / np.sqrt(len(values)) if len(values) > 1 else 0.0
            aggregated_stats[f"{metric}_mean"] = mean_val
            aggregated_stats[f"{metric}_std"] = std_val
            aggregated_stats[f"{metric}_sem"] = sem_val
        else:
            # If all values were NaN, store NaN for aggregates
            aggregated_stats[f"{metric}_mean"] = np.nan
            aggregated_stats[f"{metric}_std"] = np.nan
            aggregated_stats[f"{metric}_sem"] = np.nan

    return aggregated_stats


# --- Main Workflow ---
if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_BASE_DIR = "./configs_generated" # Directory with YAML config files
    RESULTS_BASE_DIR = "./experiment_results" # Directory with CSV results files
    OUTPUT_STATS_FILE = Path("./analysis_outputs/aggregated_statistics.csv") # Output CSV path

    # --- Setup ---
    OUTPUT_STATS_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # --- 1. Load Configs ---
    all_configs = load_configs(CONFIG_BASE_DIR)
    if not all_configs:
        logging.error("Failed to load any configurations. Exiting.")
        exit()

    # --- 2. Load Results ---
    # Extract objective names from loaded configs to potentially filter results loading
    # objectives_found = {cfg.get('_objective_name') for cfg in all_configs.values() if cfg.get('_objective_name')}
    # objectives_to_load = list(objectives_found) if objectives_found else None
    objectives_to_load = None # Or specify manually: ['baselines', 'attack_comparison']

    raw_results_data = load_results_data(RESULTS_BASE_DIR, objectives=objectives_to_load)
    if not raw_results_data:
        logging.error("Failed to load any results data. Exiting.")
        exit()

    # --- 3. Process and Aggregate ---
    all_aggregated_stats = []
    processed_count = 0
    skipped_count = 0

    logging.info("Starting preprocessing, integration, and aggregation...")
    # Iterate through the *results* data, as we need results to exist
    for combined_exp_key, run_dfs_list in raw_results_data.items():
        # Find the corresponding configuration
        if combined_exp_key not in all_configs:
            logging.warning(f"Skipping results for '{combined_exp_key}': Corresponding configuration not found.")
            skipped_count += len(run_dfs_list)
            continue

        config = all_configs[combined_exp_key]

        # Preprocess each run's DataFrame for this experiment setup
        processed_dfs_for_exp = []
        for df_run in run_dfs_list:
            processed_df = preprocess_and_integrate(df_run, config)
            if processed_df is not None:
                processed_dfs_for_exp.append(processed_df)
            else:
                logging.warning(f"Failed to preprocess run {df_run.get('run_id', 'unknown')} for {combined_exp_key}")
                skipped_count += 1


        # Calculate aggregated statistics if any runs were successfully processed
        if processed_dfs_for_exp:
            aggregated_stats = calculate_aggregated_stats(processed_dfs_for_exp)

            if aggregated_stats:
                # Add key identifying parameters from the config to the stats dict
                aggregated_stats['combined_exp_key'] = combined_exp_key
                aggregated_stats['objective_name'] = config.get('_objective_name', 'unknown')
                aggregated_stats['experiment_id_part'] = config.get('_experiment_id_part', 'unknown')
                # Add main experimental parameters used for grouping
                aggregated_stats['exp_aggregator'] = config.get('aggregation_method', 'unknown')
                aggregated_stats['exp_dataset'] = config.get('dataset_name', 'unknown')
                aggregated_stats['exp_adv_rate'] = config.get('data_split', {}).get('adv_rate', 0.0)
                attack_config = config.get('attack', {})
                aggregated_stats['exp_attack_type'] = attack_config.get('attack_type', 'none')
                if aggregated_stats['exp_attack_type'] == 'none' and attack_config.get('scenario') == 'backdoor':
                     aggregated_stats['exp_attack_type'] = 'backdoor' # Handle fallback
                aggregated_stats['exp_attack_enabled'] = attack_config.get('enabled', False)
                aggregated_stats['exp_sybil_enabled'] = config.get('sybil', {}).get('is_sybil', False)
                # Add discovery params if relevant
                if config.get('data_split', {}).get('data_split_mode') == 'discovery':
                    aggregated_stats['exp_discovery_quality'] = config.get('data_split',{}).get('dm_params',{}).get('discovery_quality', np.nan)
                    aggregated_stats['exp_buyer_mode'] = config.get('data_split',{}).get('dm_params',{}).get('buyer_data_mode', 'unknown')
                else:
                    aggregated_stats['exp_discovery_quality'] = np.nan
                    aggregated_stats['exp_buyer_mode'] = 'N/A'


                all_aggregated_stats.append(aggregated_stats)
                processed_count += aggregated_stats.get('num_runs', 0) # Add number of runs processed for this key
            else:
                 logging.warning(f"Aggregation failed for {combined_exp_key} despite having processed runs.")
                 skipped_count += len(processed_dfs_for_exp) # Count runs that failed aggregation
        else:
            logging.warning(f"No runs could be successfully preprocessed for {combined_exp_key}.")
            # skipped_count already incremented during preprocess loop


    # --- 4. Create and Save Final Statistics DataFrame ---
    if not all_aggregated_stats:
        logging.error("No experiments could be processed and aggregated. No statistics generated.")
    else:
        final_summary_df = pd.DataFrame(all_aggregated_stats)

        # Optional: Reorder columns for better readability
        id_cols = ['combined_exp_key', 'objective_name', 'experiment_id_part', 'num_runs']
        param_cols = [c for c in final_summary_df.columns if c.startswith('exp_')]
        metric_cols = [c for c in final_summary_df.columns if c not in id_cols and not c.startswith('exp_')]
        final_summary_df = final_summary_df[id_cols + sorted(param_cols) + sorted(metric_cols)]

        try:
            final_summary_df.to_csv(OUTPUT_STATS_FILE, index=False)
            logging.info(f"Successfully processed {processed_count} runs, skipped {skipped_count}.")
            logging.info(f"Aggregated statistics saved to: {OUTPUT_STATS_FILE}")
        except Exception as e:
            logging.error(f"Failed to save aggregated statistics CSV: {e}")

    logging.info("Statistics generation finished.")