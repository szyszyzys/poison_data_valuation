import ast
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, norm

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Plotting Style ---
sns.set_theme(style="whitegrid", palette="viridis")

# Define objectives if needed for filtering during loading (optional)
# OBJECTIVES = [
#     "attack_comparison",
#     "label_flip_attack_comparison",
#     "sybil_comparison",
#     "discovery_split",
#     "privacy",
#     "baselines"
# ]

# --- Utility Functions ---

def safe_literal_eval(val: Any) -> Any:
    """
    Safely evaluate strings that look like Python literals (list/dict/tuple)
    or attempt to convert to float, otherwise return original.
    Handles None/NaN gracefully.
    """
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (list, dict, tuple, int, float)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if (s.startswith('{') and s.endswith('}')) or \
           (s.startswith('[') and s.endswith(']')) or \
           (s.startswith('(') and s.endswith(')')):
            try:
                evaluated = ast.literal_eval(s)
                # Ensure lists/tuples are lists for consistency later
                return list(evaluated) if isinstance(evaluated, tuple) else evaluated
            except (ValueError, SyntaxError, MemoryError):
                logging.warning(f"Could not literal_eval string: '{s[:100]}...'")
                # Decide fallback: return original string or None/empty?
                # Returning empty list/dict might be safer for downstream processing
                if s.startswith('['): return []
                if s.startswith('{'): return {}
                return s # Fallback to original string if unsure
        try:
            # Attempt float conversion for simple strings
            return float(s)
        except ValueError:
            # Return the original string if it's not a literal or float
            return s
    # Return original value if it's some other type
    return val


def get_experiment_parameter(exp_name: str, pattern: str, default: Any = None, cast_type: type = str) -> Any:
    """Extracts a parameter value from an experiment name string using regex."""
    match = re.search(pattern, exp_name)
    if match:
        try:
            return cast_type(match.group(1))
        except (ValueError, TypeError):
            logging.warning(f"Could not cast matched value '{match.group(1)}' to {cast_type} for pattern '{pattern}' in '{exp_name}'.")
            return default
    return default

# --- Metric Calculation Functions ---

def calculate_cost_of_convergence(df_run: pd.DataFrame, target_acc: float, acc_col: str = 'global_acc') -> Optional[float]:
    """
    Return cumulative #selected sellers until global_acc >= target_acc, else NaN.
    Assumes 'round_number' and 'num_sellers_selected' columns exist.
    """
    if acc_col not in df_run.columns or 'num_sellers_selected' not in df_run.columns or 'round_number' not in df_run.columns:
        logging.debug(f"Missing columns for CoC calculation ({acc_col}, num_sellers_selected, round_number).")
        return np.nan
    df_valid = df_run.dropna(subset=[acc_col, 'num_sellers_selected', 'round_number'])
    if df_valid.empty:
        return np.nan

    meets = df_valid[df_valid[acc_col] >= target_acc]
    if meets.empty:
        # If target never met, return total cost over all rounds as a potential upper bound (or NaN)
        # return df_valid['num_sellers_selected'].sum() # Option 1: Total cost
        return np.nan # Option 2: Indicate failure to converge

    first_round_met = meets['round_number'].min()
    # Sum selections up to and *including* the round where target was met
    cost = df_valid.loc[df_valid['round_number'] <= first_round_met, 'num_sellers_selected'].sum()
    return cost


def calculate_fairness_differential(df_run: pd.DataFrame) -> Optional[float]:
    """
    Avg per-seller benign selection rate MINUS avg per-seller malicious selection rate.
    Positive value means benign sellers are selected proportionally more often.
    """
    required_cols = ['round_number', 'benign_selected_count', 'malicious_selected_count', 'num_benign', 'num_malicious']
    if not all(col in df_run.columns for col in required_cols):
        logging.debug("Missing columns for fairness differential calculation.")
        return np.nan

    df_valid = df_run.dropna(subset=required_cols)
    if df_valid.empty: return np.nan

    n_rounds = df_valid['round_number'].max() + 1 # Assuming round numbers start from 0
    if n_rounds <= 0: return np.nan

    total_benign_selections = df_valid['benign_selected_count'].sum()
    total_malicious_selections = df_valid['malicious_selected_count'].sum()

    # Use the number of benign/malicious sellers from the last available round
    # (assuming it's constant throughout the run)
    num_benign = df_valid['num_benign'].iloc[-1] if not df_valid['num_benign'].empty else 0
    num_malicious = df_valid['num_malicious'].iloc[-1] if not df_valid['num_malicious'].empty else 0

    avg_benign_rate = (total_benign_selections / num_benign / n_rounds) if num_benign > 0 else 0
    avg_malicious_rate = (total_malicious_selections / num_malicious / n_rounds) if num_malicious > 0 else 0

    # If only one type of seller exists, the differential might not be meaningful, return NaN or 0?
    if num_benign == 0 or num_malicious == 0:
        # Let's return NaN as the comparison isn't really possible
        return np.nan

    return avg_benign_rate - avg_malicious_rate


def get_seller_divergence(sid: str) -> Optional[float]:
    """
    Placeholder function to extract a 'divergence' or 'quality' proxy from seller ID.
    Example: Assumes seller ID like 'bn_f0.1_id123' or 'adv_f0.9_id456'.
    Adjust the regex pattern based on your actual seller ID naming convention.
    """
    # Try matching pattern like '_f<float>_' indicating discovery noise level 'f'
    match = re.search(r'_f([\d\.]+)_', sid)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Add other patterns if needed, e.g., quality score _q<float>_
    # match = re.search(r'_q([\d\.]+)_', sid)
    # if match:
    #     try:
    #         return float(match.group(1))
    #     except ValueError:
    #         pass

    # If no pattern matches, return None or a default value (e.g., 0 or np.nan)
    return np.nan


def calculate_fairness_correlation(df_run: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Calculate Spearman correlation between benign sellers' selection frequency
    and their divergence/quality proxy (extracted via `get_seller_divergence`).
    Returns (correlation_coefficient, p_value) or None if calculation fails.
    """
    required_cols = ['selected_sellers', 'seller_ids_all', 'experiment_setup', 'round_number']
    if not all(col in df_run.columns for col in required_cols):
        logging.debug("Missing columns for fairness correlation calculation.")
        return None

    df_valid = df_run.dropna(subset=['selected_sellers', 'round_number'])
    if df_valid.empty: return None

    # Ensure 'seller_ids_all' is available and is a list
    seller_ids_all = df_run['seller_ids_all'].iloc[0] if 'seller_ids_all' in df_run and len(df_run) > 0 else []
    if not isinstance(seller_ids_all, list) or not seller_ids_all:
         logging.debug("seller_ids_all is missing or not a list.")
         return None

    benign_ids = [s for s in seller_ids_all if isinstance(s, str) and s.startswith('bn_')]
    if not benign_ids:
        logging.debug("No benign seller IDs found.")
        return None

    # Flatten the list of selected sellers across all rounds
    all_selections = [item for sublist in df_valid['selected_sellers'] if isinstance(sublist, list) for item in sublist]
    selection_counts = Counter(all_selections)

    n_rounds = df_valid['round_number'].max() + 1
    if n_rounds <= 0: return None

    frequencies: List[float] = []
    divergences: List[float] = []

    for bid in benign_ids:
        divergence = get_seller_divergence(bid)
        # Only include sellers for whom we can get a divergence value
        if divergence is not None and not np.isnan(divergence):
            freq = selection_counts.get(bid, 0) / n_rounds
            frequencies.append(freq)
            divergences.append(divergence)

    # Need at least two data points to calculate correlation
    if len(frequencies) < 2:
        logging.debug(f"Insufficient data points ({len(frequencies)}) for correlation calculation.")
        return None

    try:
        rho, p_value = spearmanr(frequencies, divergences)
        # Handle potential NaN result from spearmanr if input arrays have issues (e.g., zero variance)
        if np.isnan(rho):
            logging.warning("Spearman correlation resulted in NaN.")
            return None
        return rho, p_value
    except Exception as e:
        logging.error(f"Error calculating Spearman correlation: {e}")
        return None


def gini(x: np.ndarray) -> float:
    """Calculate the Gini coefficient of a numpy array."""
    x = np.asarray(x, dtype=float)
    # Filter out NaNs
    x = x[~np.isnan(x)]
    if x.size == 0: return np.nan # Or 0.0 if preferred for empty input
    # Ensure non-negative values (Gini is usually defined for non-negative quantities)
    if np.any(x < 0):
         # Shift values to be non-negative if context allows (e.g., selection counts)
         # If negative values have meaning, Gini might not be appropriate
         x = x - np.min(x)
         # logging.warning("Input array to Gini contained negative values. Shifted to non-negative.")

    # Handle the case of all zeros
    if np.all(x == 0):
        return 0.0

    # Sort the array
    x_sorted = np.sort(x)
    n = x_sorted.size
    # Cumulative sum
    cumx = np.cumsum(x_sorted)
    if cumx[-1] == 0: # Avoid division by zero if sum is zero after filtering/shifting
        return 0.0
    # Calculate Gini using the formula: (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    gini_coeff = (2.0 * np.sum((np.arange(1, n + 1) * x_sorted)) / (n * cumx[-1])) - (n + 1.0) / n
    return gini_coeff


def calculate_gini_coefficient(df_run: pd.DataFrame) -> Optional[float]:
    """Calculate Gini coefficient over total selections per seller."""
    required_cols = ['selected_sellers', 'seller_ids_all', 'round_number']
    if not all(col in df_run.columns for col in required_cols):
        logging.debug("Missing columns for Gini calculation.")
        return np.nan

    df_valid = df_run.dropna(subset=['selected_sellers', 'round_number'])
    if df_valid.empty: return np.nan

    seller_ids_all = df_run['seller_ids_all'].iloc[0] if 'seller_ids_all' in df_run and len(df_run) > 0 else []
    if not isinstance(seller_ids_all, list) or not seller_ids_all:
         logging.debug("seller_ids_all is missing or not a list for Gini calculation.")
         return np.nan

    # Flatten the list of selected sellers
    all_selections = [item for sublist in df_valid['selected_sellers'] if isinstance(sublist, list) for item in sublist]
    selection_counts = Counter(all_selections)

    # Get counts for *all* potential sellers, including those never selected (count = 0)
    selection_values = [selection_counts.get(sid, 0) for sid in seller_ids_all]

    if len(selection_values) < 1: # Need at least one seller
        return 0.0 if len(seller_ids_all) > 0 else np.nan # 0 if sellers exist but none selected? Nan if no sellers?

    gini_coeff = gini(np.array(selection_values))
    return gini_coeff


def calculate_selection_entropy(df_run: pd.DataFrame) -> Optional[float]:
    """Calculate normalized Shannon entropy of selection frequencies across all sellers."""
    required_cols = ['selected_sellers', 'seller_ids_all', 'round_number']
    if not all(col in df_run.columns for col in required_cols):
        logging.debug("Missing columns for entropy calculation.")
        return np.nan

    df_valid = df_run.dropna(subset=['selected_sellers', 'round_number'])
    if df_valid.empty: return np.nan

    seller_ids_all = df_run['seller_ids_all'].iloc[0] if 'seller_ids_all' in df_run and len(df_run) > 0 else []
    if not isinstance(seller_ids_all, list) or not seller_ids_all:
         logging.debug("seller_ids_all is missing or not a list for entropy calculation.")
         return np.nan

    num_sellers = len(seller_ids_all)
    if num_sellers < 1:
        return np.nan # Cannot calculate entropy with no sellers

    # Flatten the list of selected sellers
    all_selections = [item for sublist in df_valid['selected_sellers'] if isinstance(sublist, list) for item in sublist]
    selection_counts = Counter(all_selections)

    # Get counts for all sellers
    counts_array = np.array([selection_counts.get(sid, 0) for sid in seller_ids_all], dtype=float)
    total_selections = counts_array.sum()

    if total_selections == 0:
        # If no sellers were ever selected, entropy is arguably 0 (or undefined/NaN)
        return 0.0 # Let's define it as 0 diversity

    # Calculate probabilities (frequencies)
    probabilities = counts_array / total_selections
    # Filter out probabilities of 0 to avoid log2(0)
    probabilities = probabilities[probabilities > 0]

    if len(probabilities) == 0: # Should not happen if total_selections > 0, but as safeguard
        return 0.0

    # Calculate Shannon entropy: H = -sum(p_i * log2(p_i))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Normalize by maximum possible entropy (log2(N)), where N is number of sellers
    # Avoid division by zero if only 1 seller
    max_entropy = np.log2(num_sellers) if num_sellers > 1 else 1.0 # Normalize to 1 if only 1 seller? Or entropy is 0?
    if num_sellers <= 1:
         return 0.0 # Entropy is 0 if only one choice exists

    normalized_entropy = entropy / max_entropy
    # Ensure result is within [0, 1] due to potential float inaccuracies
    return np.clip(normalized_entropy, 0.0, 1.0)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0  # Similarity is 1 if both sets are empty
    return intersection / union


def calculate_selection_stability(df_run: pd.DataFrame) -> Optional[float]:
    """Calculate the average Jaccard similarity between selected seller sets in consecutive rounds."""
    required_cols = ['selected_sellers', 'round_number']
    if not all(col in df_run.columns for col in required_cols):
        logging.debug("Missing columns for stability calculation.")
        return np.nan

    # Sort by round number and drop rounds with missing selections
    df_sorted = df_run.sort_values('round_number').dropna(subset=['selected_sellers'])
    # Ensure 'selected_sellers' contains lists/sets
    selections_per_round = [set(s) if isinstance(s, list) else set() for s in df_sorted['selected_sellers']]

    if len(selections_per_round) < 2:
        # Stability is undefined or trivially 1/0 with fewer than 2 rounds
        return np.nan # Or 1.0? Let's use NaN

    jaccard_indices: List[float] = []
    for i in range(len(selections_per_round) - 1):
        set1 = selections_per_round[i]
        set2 = selections_per_round[i+1]
        jaccard_indices.append(jaccard_similarity(set1, set2))

    if not jaccard_indices: # Should not happen if len > 1, but safeguard
        return np.nan

    return float(np.mean(jaccard_indices))


# --- Data Loading and Preprocessing ---

def load_all_results(
    base_dir: str,
    csv_filename: str = "round_results.csv",
    objectives: Optional[List[str]] = None # Now used for filtering
) -> Dict[str, List[pd.DataFrame]]:
    """
    Load results from base_dir/<objective>/<experiment>/run_*/csv_filename.
    Only loads from objective folders listed in `objectives` if provided.
    Returns a dictionary mapping a combined 'objective_experiment' name
    to a list of run DataFrames.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        logging.error(f"Base directory '{base_dir}' not found.")
        return {}

    all_results: Dict[str, List[pd.DataFrame]] = {}
    logging.info(f"Scanning for results under objectives in: {base_path}")

    # 1. Iterate through directories directly under base_dir (these are objectives)
    for objective_path in sorted(base_path.iterdir()):
        if not objective_path.is_dir():
            continue

        objective_name = objective_path.name
        print(objective_name)
        # 2. Filter by objectives list if provided
        if objectives and objective_name not in objectives:
            logging.debug(f"Skipping objective folder (not in requested list): {objective_name}")
            continue

        logging.info(f"Processing objective: {objective_name}")

        # 3. Iterate through subdirectories of the objective (these are experiments)
        for exp_path in sorted(objective_path.iterdir()):
            if not exp_path.is_dir():
                continue

            experiment_name_part = exp_path.name
            # 6. Construct the combined experiment name
            combined_exp_name = f"{objective_name}_{experiment_name_part}"

            run_dfs: List[pd.DataFrame] = []
            logging.info(f"--> Processing experiment: {experiment_name_part} (Combined Key: {combined_exp_name})")

            # 4. Look for run_* directories within the experiment path
            run_dirs = list(exp_path.glob('run_*'))
            if not run_dirs:
                logging.warning(f"No 'run_*' directories found in {exp_path}")
                continue

            # 5. Load data from each run directory
            for run_dir in sorted(run_dirs):
                if not run_dir.is_dir(): continue
                csv_file = run_dir / csv_filename
                if not csv_file.is_file():
                    # Optional: log only if verbose logging is enabled
                    # logging.debug(f"CSV file '{csv_filename}' not found in {run_dir}")
                    continue
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        logging.warning(f"CSV file is empty: {csv_file}")
                        continue
                    # Store run ID and the combined experiment name
                    df['run_id'] = run_dir.name
                    df['experiment_setup'] = combined_exp_name # Use the combined name
                    # Store original parts if needed for later filtering
                    df['objective_name'] = objective_name
                    df['experiment_name_part'] = experiment_name_part
                    run_dfs.append(df)
                except pd.errors.EmptyDataError:
                     logging.warning(f"CSV file is empty: {csv_file}")
                except Exception as e:
                    logging.error(f"Failed to load or process {csv_file}: {e}")

            # 7. Store results keyed by the combined name
            if run_dfs:
                # If combined name already exists (shouldn't with this structure but safer), append
                if combined_exp_name in all_results:
                     all_results[combined_exp_name].extend(run_dfs)
                     logging.warning(f"Appended {len(run_dfs)} runs to existing key '{combined_exp_name}'. Check for duplicate experiment structures.")
                else:
                     all_results[combined_exp_name] = run_dfs
                logging.info(f"--> Loaded {len(run_dfs)} runs for {combined_exp_name}")
            # else: (already warned about no runs found)

    if not all_results:
        logging.error(f"No experiments loaded matching the structure base_dir/objective/experiment/run_* from {base_path}.")
        if objectives:
             logging.error(f"Filtering was active for objectives: {objectives}")
    else:
        logging.info(f"Finished loading. Found data for {len(all_results)} unique experiment setups.")
    return all_results


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps to a single run's DataFrame.
    - Safely evaluate list/dict string columns.
    - Extract key metrics (acc, loss, asr, privacy).
    - Extract experiment parameters (adv rate, discovery, buyer mode).
    - Calculate derived counts (benign/malicious selected).
    """
    if df.empty:
        return df

    logging.debug(f"Preprocessing run {df['run_id'].iloc[0]} for exp {df['experiment_setup'].iloc[0]} with columns: {df.columns.tolist()}")

    # Columns expected to contain list/dict/tuple strings
    literal_cols = ['selected_sellers', 'seller_ids_all', 'perf_global', 'perf_local',
                    'selection_rate_info', 'gradient_inversion_log',
                    'seller_scores', 'seller_contributions'] # Add others as needed
    list_cols = ['selected_sellers', 'seller_ids_all'] # Ensure these are lists
    dict_cols = ['perf_global', 'perf_local', 'selection_rate_info',
                 'gradient_inversion_log'] # Ensure these are dicts

    for col in literal_cols:
        if col in df.columns:
            # Apply safe_literal_eval only if column type is object (likely string)
            if df[col].dtype == 'object':
                df[col] = df[col].apply(safe_literal_eval)
            # Handle cases where data might already be loaded correctly (e.g., from parquet)
            elif pd.api.types.is_list_like(df[col].iloc[0]) or isinstance(df[col].iloc[0], dict):
                 pass # Already seems parsed
            else:
                 # Attempt conversion if it's not object but might contain representable strings
                 try:
                      df[col] = df[col].apply(safe_literal_eval)
                 except Exception as e:
                      logging.warning(f"Failed safe_literal_eval on non-object column '{col}': {e}")


    # Ensure specific columns are lists or dicts after eval, fillna with appropriate empty type
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        else:
            df[col] = [[] for _ in range(len(df))] # Add empty list column if missing

    for col in dict_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
        else:
            df[col] = [{} for _ in range(len(df))] # Add empty dict column if missing

    # --- Extract Performance Metrics ---
    if 'perf_global' in df.columns:
        df['global_acc'] = df['perf_global'].apply(lambda d: d.get('accuracy', np.nan))
        df['global_loss'] = df['perf_global'].apply(lambda d: d.get('loss', np.nan))
        # Assumes ASR is logged under 'attack_success_rate' or similar
        asr_key = next((k for k in df['perf_global'].iloc[0].keys() if 'asr' in k or 'attack_success' in k), None)
        if asr_key:
             df['global_asr'] = df['perf_global'].apply(lambda d: d.get(asr_key, np.nan))
        else:
             df['global_asr'] = np.nan # Ensure column exists even if ASR wasn't measured

    # --- Extract Privacy Metrics ---
    if 'gradient_inversion_log' in df.columns:
        df['attack_psnr'] = df['gradient_inversion_log'].apply(lambda d: d.get('metrics', {}).get('psnr', np.nan))
        df['attack_ssim'] = df['gradient_inversion_log'].apply(lambda d: d.get('metrics', {}).get('ssim', np.nan))
        df['attack_label_acc'] = df['gradient_inversion_log'].apply(lambda d: d.get('metrics', {}).get('label_accuracy', np.nan)) # Adjust key if needed
    else:
        # Ensure columns exist even if no privacy attack was run
        df['attack_psnr'] = np.nan
        df['attack_ssim'] = np.nan
        df['attack_label_acc'] = np.nan

    # --- Convert Core Numerics ---
    numeric_cols = ['round_number', 'num_sellers_selected', 'total_sellers_available'] # Add others if present
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure round_number is integer if present and valid
    if 'round_number' in df.columns and not df['round_number'].isnull().all():
        # Use Int64 (capital I) to handle potential NaNs during conversion
        df['round_number'] = df['round_number'].astype(pd.Int64Dtype())


    # --- Extract Parameters from Experiment Name ---
    # Assume first row's experiment_setup is representative for the whole run
    exp_name = df['experiment_setup'].iloc[0] if 'experiment_setup' in df.columns and len(df) > 0 else ""

    # Adv Rate (e.g., _adv10pct_, _adv0.3_)
    adv_rate_pct = get_experiment_parameter(exp_name, r'_adv(\d+)pct', default=None, cast_type=int)
    adv_rate_frac = get_experiment_parameter(exp_name, r'_adv(0\.\d+)', default=None, cast_type=float)
    df['exp_adv_rate'] = (adv_rate_pct / 100.0) if adv_rate_pct is not None else adv_rate_frac if adv_rate_frac is not None else 0.0

    # Discovery Quality (e.g., _discoveryF0.1_, _f0.1_)
    df['exp_discovery_f'] = get_experiment_parameter(exp_name, r'(?:discoveryF|f)([\d\.]+)', default=np.nan, cast_type=float)

    # Buyer Data Mode (e.g., _buyerUnbiased_, _buyerBiased_)
    df['exp_buyer_mode'] = get_experiment_parameter(exp_name, r'_buyer(unbiased|biased)', default='unknown', cast_type=str)

    # Dataset (e.g., _cifar_, _fmnist_) - useful for grouping later
    df['exp_dataset'] = get_experiment_parameter(exp_name, r'_(cifar|fmnist|agnews|trec)', default='unknown', cast_type=str)

    # Aggregator (e.g., _fedavg_, _fltrust_)
    df['exp_aggregator'] = get_experiment_parameter(exp_name, r'_(fedavg|fltrust|martfl|skymask)', default='unknown', cast_type=str)

    # Attack Type (e.g., _backdoor_, _labelflip_, _sybil_)
    df['exp_attack'] = get_experiment_parameter(exp_name, r'_(backdoor|label.?flip|sybil|mimicry)', default='none', cast_type=str)


    # --- Calculate Derived Seller Counts/Rates ---
    # Use first row's seller_ids_all (assuming it's constant for the run)
    seller_ids = df['seller_ids_all'].iloc[0] if 'seller_ids_all' in df.columns and len(df) > 0 and isinstance(df['seller_ids_all'].iloc[0], list) else []

    benign_ids = {s for s in seller_ids if isinstance(s, str) and s.startswith('bn_')}
    adv_ids = {s for s in seller_ids if isinstance(s, str) and s.startswith('adv_')}
    df['num_benign'] = len(benign_ids)
    df['num_malicious'] = len(adv_ids)

    if 'selected_sellers' in df.columns:
        # Ensure 'num_sellers_selected' is calculated or present if needed for rates
        if 'num_sellers_selected' not in df.columns:
             df['num_sellers_selected'] = df['selected_sellers'].apply(len)

        df['benign_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in benign_ids))
        df['malicious_selected_count'] = df['selected_sellers'].apply(lambda sel: sum(1 for s in sel if s in adv_ids))

        # Calculate rates, handle division by zero
        df['benign_selection_rate'] = (df['benign_selected_count'] / df['num_sellers_selected']).fillna(0)
        df['malicious_selection_rate'] = (df['malicious_selected_count'] / df['num_sellers_selected']).fillna(0)
    else:
        # Ensure columns exist even if selections weren't logged properly
        df['benign_selected_count'] = 0
        df['malicious_selected_count'] = 0
        df['benign_selection_rate'] = 0.0
        df['malicious_selection_rate'] = 0.0

    # Log columns after preprocessing for debugging
    logging.debug(f"Columns after preprocessing: {df.columns.tolist()}")
    # Log unique values of extracted parameters for verification
    logging.debug(f"Extracted params: adv_rate={df['exp_adv_rate'].unique()}, "
                  f"discovery_f={df['exp_discovery_f'].unique()}, "
                  f"buyer_mode={df['exp_buyer_mode'].unique()}, "
                  f"dataset={df['exp_dataset'].unique()}, "
                  f"aggregator={df['exp_aggregator'].unique()}, "
                  f"attack={df['exp_attack'].unique()}")


    return df


# --- Plotting Functions ---
# (Keep all plotting functions as they were in the previous version)
# plot_metric_comparison, plot_final_round_comparison, plot_scatter_comparison,
# plot_aggregated_metric_bar, plot_discovery_effect, plot_buyer_mode_effect,
# plot_fairness_scatter

def plot_metric_comparison(
        results: Dict[str, List[pd.DataFrame]],
        metric: str,
        title: str,
        xlabel: str = "Round Number",
        ylabel: Optional[str] = None,
        ci: Union[str, int, None] = 'sd', # 'sd' for std dev, 95 for 95% CI, None for no error bars
        save_path: Optional[Path] = None
) -> None:
    """
    Generate a line plot comparing a metric over rounds across different experiments.
    Handles multiple runs per experiment by plotting mean ± confidence interval.
    """
    all_dfs = []
    for exp_name, run_list in results.items():
        for i, df_run in enumerate(run_list):
            # Ensure required columns exist and metric has non-NaN values
            if metric in df_run.columns and 'round_number' in df_run.columns and not df_run[metric].isnull().all():
                # Select relevant columns and add identifiers
                df_plot = df_run[['round_number', metric]].copy()
                df_plot['experiment'] = exp_name # Use the main experiment name for grouping
                df_plot['run_id'] = df_run['run_id'].iloc[0] if 'run_id' in df_run else f"run_{i}"
                all_dfs.append(df_plot)
            # else:
            #     logging.warning(f"Metric '{metric}' or 'round_number' missing/empty in run {i} of {exp_name}")


    if not all_dfs:
        logging.warning(f"No valid data found for metric '{metric}' in the provided results. Skipping plot: '{title}'")
        return
    df_all = pd.concat(all_dfs, ignore_index=True).dropna(subset=[metric, 'round_number']) # Drop rows where metric or round is NaN

    if df_all.empty:
        logging.warning(f"No non-NaN data points found for metric '{metric}' after concatenation. Skipping plot: '{title}'")
        return

    plt.figure(figsize=(10, 6))
    try:
        sns.lineplot(
            data=df_all,
            x='round_number',
            y=metric,
            hue='experiment',
            estimator=np.mean, # Plot the mean across runs
            errorbar=ci,       # Show confidence interval (e.g., standard deviation)
            legend='full'      # Show legend
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric.replace('_', ' ').title()) # Auto-generate Y label if not provided
        plt.legend(title='Experiment Setup', loc='best', fontsize='small') # Adjust legend location and size
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plot: {save_path}")
        else:
            plt.show() # Display plot if not saving

    except Exception as e:
        logging.error(f"Failed to generate line plot '{title}' for metric '{metric}': {e}")
    finally:
        plt.close() # Close the figure to free memory


def plot_final_round_comparison(
        results: Dict[str, List[pd.DataFrame]],
        metric: str,
        title: str,
        higher_is_better: bool = True,
        ylabel: Optional[str] = None,
        save_path: Optional[Path] = None
) -> None:
    """
    Generate a bar plot comparing the final round's metric value across experiments.
    Error bars represent the standard error of the mean across runs.
    Highlights the best performing experiment.
    """
    final_values = {} # {exp_name: [final_val_run1, final_val_run2, ...]}

    for exp_name, run_list in results.items():
        run_final_vals = []
        for i, df_run in enumerate(run_list):
            if metric in df_run.columns and 'round_number' in df_run.columns:
                # Find the row corresponding to the maximum round number
                df_valid = df_run.dropna(subset=[metric, 'round_number'])
                if not df_valid.empty:
                    last_round_idx = df_valid['round_number'].idxmax()
                    final_val = df_valid.loc[last_round_idx, metric]
                    if not pd.isna(final_val):
                        run_final_vals.append(final_val)
                    # else:
                    #     logging.debug(f"NaN final value for metric '{metric}' in run {i} of {exp_name}")
                # else:
                #     logging.debug(f"No valid rows found for metric '{metric}' in run {i} of {exp_name}")
            # else:
            #     logging.warning(f"Metric '{metric}' or 'round_number' missing in run {i} of {exp_name}")

        if run_final_vals:
            final_values[exp_name] = run_final_vals
        # else:
        #     logging.warning(f"No valid final values found for metric '{metric}' in experiment {exp_name}")


    if not final_values:
        logging.warning(f"No final round data available for metric '{metric}'. Skipping bar plot: '{title}'")
        return

    exp_names = list(final_values.keys())
    means = [np.mean(vals) for vals in final_values.values()]
    # Calculate standard error of the mean (SEM = std_dev / sqrt(n))
    errors = [np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0 for vals in final_values.values()]

    # Create DataFrame for easier plotting and handling
    df_plot = pd.DataFrame({'experiment': exp_names, 'mean': means, 'sem': errors})

    # Find the best performing experiment
    best_idx = -1
    if means: # Check if means list is not empty
        try:
            best_idx = int(np.argmax(means) if higher_is_better else np.argmin(means))
        except ValueError: # Handle cases where means might contain NaNs after processing
            logging.warning(f"Could not determine best experiment for {metric} in '{title}' due to invalid values.")


    plt.figure(figsize=(max(8, len(exp_names) * 0.8), 6)) # Adjust width based on number of bars

    # Create bar plot
    try:
        bar_container = plt.bar(df_plot['experiment'], df_plot['mean'], yerr=df_plot['sem'], capsize=5,
                                color=sns.color_palette("viridis", len(exp_names))) # Use a color palette

        # Highlight the best bar if found
        if best_idx != -1 and best_idx < len(bar_container):
            best_exp_name = exp_names[best_idx]
            bar_container[best_idx].set_color('red')
            # Add annotation for the best experiment
            plt.text(best_idx, df_plot['mean'].iloc[best_idx] + df_plot['sem'].iloc[best_idx], ' Best',
                     ha='center', va='bottom', color='red', fontweight='bold')
        elif best_idx != -1:
             logging.warning(f"Best index {best_idx} out of range for highlighting in '{title}'.")


    except Exception as e:
        logging.error(f"Error creating bar plot for '{title}': {e}")
        plt.close()
        return

    plt.xticks(rotation=45, ha='right', fontsize='small') # Rotate labels for readability
    plt.title(title)
    plt.ylabel(ylabel or metric.replace('_', ' ').title())
    plt.tight_layout()


    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_scatter_comparison(
        results: Dict[str, List[pd.DataFrame]],
        x_metric: str,
        y_metric: str,
        title: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[Path] = None
) -> None:
    """
    Generate a scatter plot comparing the final round values of two metrics across experiments.
    Each point represents the mean final value for an experiment setup.
    """
    scatter_data = []

    for exp_name, run_list in results.items():
        x_vals, y_vals = [], []
        for i, df_run in enumerate(run_list):
            # Check if both metrics and round_number are present
            if x_metric in df_run.columns and y_metric in df_run.columns and 'round_number' in df_run.columns:
                df_valid = df_run.dropna(subset=[x_metric, y_metric, 'round_number'])
                if not df_valid.empty:
                    last_round_idx = df_valid['round_number'].idxmax()
                    final_x = df_valid.loc[last_round_idx, x_metric]
                    final_y = df_valid.loc[last_round_idx, y_metric]
                    # Only include if both values are valid
                    if not pd.isna(final_x) and not pd.isna(final_y):
                        x_vals.append(final_x)
                        y_vals.append(final_y)

        # If we have valid data for this experiment, calculate the mean
        if x_vals and y_vals:
            scatter_data.append({
                'experiment': exp_name,
                x_metric: np.mean(x_vals),
                y_metric: np.mean(y_vals),
                'x_std': np.std(x_vals), # Store std dev for potential error bars later
                'y_std': np.std(y_vals)
            })
        # else:
        #     logging.warning(f"No valid final pairs found for metrics '{x_metric}' vs '{y_metric}' in experiment {exp_name}")


    if not scatter_data:
        logging.warning(f"No data available for scatter plot '{title}' ({x_metric} vs {y_metric}). Skipping.")
        return

    df_scatter = pd.DataFrame(scatter_data)

    plt.figure(figsize=(8, 8))
    try:
        scatter_plot = sns.scatterplot(
            data=df_scatter,
            x=x_metric,
            y=y_metric,
            hue='experiment',
            s=100, # Size of points
            legend='full'
        )
        plt.title(title)
        plt.xlabel(xlabel or x_metric.replace('_', ' ').title())
        plt.ylabel(ylabel or y_metric.replace('_', ' ').title())
        # Position legend outside plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    except Exception as e:
         logging.error(f"Error creating scatter plot '{title}': {e}")
         plt.close()
         return


    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_aggregated_metric_bar(
        summary_df: pd.DataFrame,
        metric_base: str, # e.g., 'coc', 'gini', 'fairness_diff'
        title: str,
        ylabel: str,
        higher_is_better: bool,
        save_path: Path
) -> None:
    """
    Generate a bar plot for aggregated metrics (like CoC, Gini) from the summary DataFrame.
    Uses pre-calculated mean and std columns (e.g., 'coc_mean', 'coc_std').
    Highlights the best performer.
    """
    mean_col = f"{metric_base}_mean"
    std_col = f"{metric_base}_std"

    if mean_col not in summary_df.columns:
        logging.warning(f"Mean column '{mean_col}' not found in summary DataFrame. Skipping plot: '{title}'")
        return
    # Std column is optional for error bars
    use_error_bars = std_col in summary_df.columns

    # Filter out rows where the mean is NaN
    df_plot = summary_df.dropna(subset=[mean_col]).copy()
    if df_plot.empty:
        logging.warning(f"No valid data found for metric '{metric_base}' after dropping NaNs. Skipping plot: '{title}'")
        return

    # If using error bars, fill NaN std dev with 0
    if use_error_bars:
        df_plot[std_col] = df_plot[std_col].fillna(0)
    else:
        # If std_col is missing, create a dummy column of zeros for yerr parameter
        df_plot[std_col] = 0.0

    # Sort by experiment name for consistent order? Optional.
    df_plot = df_plot.sort_values('experiment_setup')

    exp_names = df_plot['experiment_setup'].tolist()
    means = df_plot[mean_col].values
    errors = df_plot[std_col].values if use_error_bars else np.zeros_like(means)

    # Find the best performing experiment based on the mean value
    best_idx = -1
    if means.size > 0: # Check if array is not empty
        try:
            # Use nanargmax/nanargmin to handle potential NaNs if any slipped through
            best_idx = int(np.nanargmax(means) if higher_is_better else np.nanargmin(means))
        except ValueError:
            logging.warning(f"Could not determine best experiment for {metric_base} in '{title}' due to invalid values.")


    plt.figure(figsize=(max(8, len(exp_names) * 0.8), 6))

    # Create bar plot
    try:
        bar_container = plt.bar(exp_names, means, yerr=errors, capsize=5,
                                color=sns.color_palette("viridis", len(exp_names)))

        # Highlight the best bar
        if best_idx != -1 and best_idx < len(bar_container):
            try:
                bar_container[best_idx].set_color('red')
                # Add annotation for the best experiment
                plt.text(best_idx, means[best_idx] + errors[best_idx], ' Best',
                         ha='center', va='bottom', color='red', fontweight='bold')
            except IndexError:
                 logging.warning(f"Could not highlight best bar for '{title}'. Index {best_idx} out of bounds for {len(bar_container)} bars.")
            except ValueError: # Handle cases where mean/error might be NaN
                 logging.warning(f"Could not place annotation for best bar in '{title}' due to NaN value.")

        elif best_idx != -1:
             logging.warning(f"Best index {best_idx} out of range for highlighting in '{title}'.")

    except Exception as e:
        logging.error(f"Error creating aggregated bar plot '{title}': {e}")
        plt.close()
        return


    plt.xticks(rotation=45, ha='right', fontsize='small')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot: {save_path}")
    plt.close()


def plot_discovery_effect(
        processed_results: Dict[str, List[pd.DataFrame]],
        metric: str = 'global_acc',
        title_prefix: str = "Effect of Discovery Noise",
        output_dir: Path = Path("."),
        discovery_param: str = 'exp_discovery_f'
) -> None:
    """
    Plots the final performance metric against the discovery noise level 'f'.
    Generates one plot per dataset, showing how performance changes with f for each aggregator.
    Assumes experiment names encode the 'f' value and aggregator type.
    """
    plot_data = []
    for exp_name, run_list in processed_results.items():
        # Extract aggregator and discovery factor from name (using preprocessed columns)
        if not run_list: continue
        df_first_run = run_list[0] # Assume params are same across runs
        aggregator = df_first_run['exp_aggregator'].iloc[0]
        discovery_f = df_first_run[discovery_param].iloc[0]
        dataset = df_first_run['exp_dataset'].iloc[0]
        # Optionally filter by attack type etc. if discovery effect is attack-specific
        attack = df_first_run['exp_attack'].iloc[0]
        adv_rate = df_first_run['exp_adv_rate'].iloc[0]


        if aggregator == 'unknown' or pd.isna(discovery_f) or dataset == 'unknown':
            # logging.debug(f"Skipping exp {exp_name} for discovery plot (missing aggregator, f, or dataset).")
            continue

        # Calculate final metric value (mean across runs)
        final_vals = []
        for df_run in run_list:
            if metric in df_run.columns and 'round_number' in df_run.columns:
                df_valid = df_run.dropna(subset=[metric, 'round_number'])
                if not df_valid.empty:
                    last_round_idx = df_valid['round_number'].idxmax()
                    final_val = df_valid.loc[last_round_idx, metric]
                    if not pd.isna(final_val):
                        final_vals.append(final_val)

        if final_vals:
            plot_data.append({
                'aggregator': aggregator,
                'discovery_f': discovery_f,
                'dataset': dataset,
                'attack': attack, # Include for potential facetting
                'adv_rate': adv_rate, # Include for potential facetting
                f'{metric}_mean': np.mean(final_vals),
                f'{metric}_std': np.std(final_vals),
                 'n_runs': len(final_vals) # For calculating SEM if needed
            })

    if not plot_data:
        logging.warning(f"No data found for discovery effect plot (Metric: {metric}).")
        return

    df_plot = pd.DataFrame(plot_data)
    df_plot[f'{metric}_sem'] = df_plot[f'{metric}_std'] / np.sqrt(df_plot['n_runs'])


    # Create a separate plot for each dataset (and optionally attack scenario)
    for (dataset_name, attack_name, adv_rate_val), group_data in df_plot.groupby(['dataset', 'attack', 'adv_rate']):
        if dataset_name == 'unknown' or group_data.empty: continue

        plt.figure(figsize=(10, 6))
        try:
            # Plot with lines connecting points for the same aggregator, and error bars
            sns.lineplot(
                data=group_data,
                x='discovery_f',
                y=f'{metric}_mean',
                hue='aggregator',
                style='aggregator', # Use different styles (lines/markers)
                markers=True,
                err_style="bars", # Show std dev or SEM as error bars
                errorbar=('ci', 95), # Or use errorbar=lambda data: (data[f'{metric}_mean'] - data[f'{metric}_sem'], data[f'{metric}_mean'] + data[f'{metric}_sem'])) for SEM
                legend='full'
            )

            attack_str = f"Attack: {attack_name}" + (f" ({int(adv_rate_val*100)}%)" if adv_rate_val > 0 else "")
            plt.title(f"{title_prefix} on {dataset_name.upper()} ({metric})\n{attack_str}")
            plt.xlabel("Discovery Noise Level (f)")
            plt.ylabel(f"Final {metric.replace('_', ' ').title()} (Mean ± 95% CI)")
            plt.legend(title='Aggregator', loc='best')
            plt.tight_layout()

            save_path = output_dir / f"discovery_effect_{dataset_name}_{attack_name}_{int(adv_rate_val*100)}pct_{metric}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plot: {save_path}")

        except Exception as e:
            logging.error(f"Failed to generate discovery plot for {dataset_name}, {attack_name}, {metric}: {e}")
        finally:
            plt.close()


def plot_buyer_mode_effect(
    processed_results: Dict[str, List[pd.DataFrame]],
    metric: str = 'global_acc',
    title_prefix: str = "Effect of Buyer Root Data",
    output_dir: Path = Path("."),
    higher_is_better: bool = True,
) -> None:
    """
    Compares the final performance metric between 'unbiased' and 'biased' buyer root data settings.
    Generates bar plots grouped by aggregator/dataset/attack scenario. Assumes experiment names encode buyer mode.
    """
    comparison_data = {} # {(dataset, aggregator, attack, adv_rate): {'unbiased': [vals], 'biased': [vals]}}

    for exp_name, run_list in processed_results.items():
        if not run_list: continue
        df_first_run = run_list[0]
        aggregator = df_first_run['exp_aggregator'].iloc[0]
        buyer_mode = df_first_run['exp_buyer_mode'].iloc[0]
        dataset = df_first_run['exp_dataset'].iloc[0]
        # Maybe include attack type / adv rate in the key if comparing within attack scenarios
        attack = df_first_run['exp_attack'].iloc[0]
        adv_rate = df_first_run['exp_adv_rate'].iloc[0]

        if aggregator == 'unknown' or buyer_mode not in ['unbiased', 'biased'] or dataset == 'unknown':
            continue

        # Key to group experiments that are identical except for buyer_mode
        exp_key = (dataset, aggregator, attack, adv_rate)

        if exp_key not in comparison_data:
            comparison_data[exp_key] = {'unbiased': [], 'biased': []}

        # Calculate final metric value for all runs of this specific experiment
        final_vals = []
        for df_run in run_list:
            if metric in df_run.columns and 'round_number' in df_run.columns:
                df_valid = df_run.dropna(subset=[metric, 'round_number'])
                if not df_valid.empty:
                    last_round_idx = df_valid['round_number'].idxmax()
                    final_val = df_valid.loc[last_round_idx, metric]
                    if not pd.isna(final_val):
                        final_vals.append(final_val)

        if final_vals:
             # Check if the specific mode list exists before extending
             if buyer_mode in comparison_data[exp_key]:
                 comparison_data[exp_key][buyer_mode].extend(final_vals) # Use extend to add all run values
             else:
                 logging.warning(f"Unexpected buyer_mode '{buyer_mode}' encountered for key {exp_key}.")


    # --- Now Plot ---
    plot_rows = []
    for key, mode_values in comparison_data.items():
        dataset, aggregator, attack, adv_rate = key
        for mode, values in mode_values.items():
            if values: # Only add if there's data for this mode
                 mean_val = np.mean(values)
                 sem_val = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                 plot_rows.append({
                     'dataset': dataset,
                     'aggregator': aggregator,
                     'attack': attack,
                     'adv_rate': adv_rate,
                     'buyer_mode': mode,
                     'mean_metric': mean_val,
                     'sem_metric': sem_val
                 })

    if not plot_rows:
        logging.warning(f"No data found for buyer mode comparison plot (Metric: {metric}).")
        return

    df_plot = pd.DataFrame(plot_rows)

    # Create a separate plot per dataset (and potentially per attack scenario)
    for (dataset_name, attack_name, adv_rate_val), group_data in df_plot.groupby(['dataset', 'attack', 'adv_rate']):
         if dataset_name == 'unknown' or group_data.empty: continue

         plt.figure(figsize=(10, 6))
         try:
             ax = sns.barplot(
                 data=group_data,
                 x='aggregator',
                 y='mean_metric',
                 hue='buyer_mode',
                 palette={'unbiased': 'tab:blue', 'biased': 'tab:orange'}, # Use distinct colors
                 errorbar=None # We will add error bars manually for clarity
             )

             # Add error bars manually
             num_aggregators = len(group_data['aggregator'].unique())
             x_coords = np.arange(num_aggregators)
             width = 0.4 # Default bar width for hue plots in seaborn

             # Get the aggregator order from the plot axis for correct alignment
             agg_order = [tick.get_text() for tick in ax.get_xticklabels()]

             unbiased_data = group_data[group_data['buyer_mode'] == 'unbiased'].set_index('aggregator').reindex(agg_order)
             biased_data = group_data[group_data['buyer_mode'] == 'biased'].set_index('aggregator').reindex(agg_order)

             # Calculate positions for error bars (centered on the bars)
             unbiased_pos = x_coords - width / 2
             biased_pos = x_coords + width / 2

             # Add error bars, handling potential missing data for a mode/aggregator combination
             if not unbiased_data.empty:
                # Ensure alignment even if some aggregators are missing for this mode
                valid_unbiased_indices = [i for i, agg in enumerate(agg_order) if agg in unbiased_data.index]
                ax.errorbar(x=unbiased_pos[valid_unbiased_indices],
                            y=unbiased_data.loc[agg_order[valid_unbiased_indices], 'mean_metric'],
                            yerr=unbiased_data.loc[agg_order[valid_unbiased_indices], 'sem_metric'],
                            fmt='none', c='black', capsize=5)
             if not biased_data.empty:
                 valid_biased_indices = [i for i, agg in enumerate(agg_order) if agg in biased_data.index]
                 ax.errorbar(x=biased_pos[valid_biased_indices],
                             y=biased_data.loc[agg_order[valid_biased_indices], 'mean_metric'],
                             yerr=biased_data.loc[agg_order[valid_biased_indices], 'sem_metric'],
                             fmt='none', c='black', capsize=5)


             attack_str = f"Attack: {attack_name}" + (f" ({int(adv_rate_val*100)}%)" if adv_rate_val > 0 else "")
             plt.title(f"{title_prefix} on {dataset_name.upper()} ({metric})\n{attack_str}")
             plt.xlabel("Aggregator")
             plt.ylabel(f"Final {metric.replace('_', ' ').title()} (Mean ± SEM)")
             plt.xticks(rotation=15, ha='right')
             # Ensure legend only shows 'unbiased' and 'biased' once
             handles, labels = ax.get_legend_handles_labels()
             if handles: # Check if legend exists
                 ax.legend(handles, labels, title='Buyer Root Data', loc='best')
             else:
                 logging.warning(f"Could not generate legend for buyer mode plot: {dataset_name}, {attack_name}")

             plt.tight_layout()

             save_path = output_dir / f"buyer_mode_effect_{dataset_name}_{attack_name}_{int(adv_rate_val*100)}pct_{metric}.png"
             save_path.parent.mkdir(parents=True, exist_ok=True)
             plt.savefig(save_path, dpi=300, bbox_inches='tight')
             logging.info(f"Saved plot: {save_path}")

         except Exception as e:
             logging.error(f"Failed to generate buyer mode plot for {dataset_name}, {attack_name}, {metric}: {e}")
         finally:
             plt.close()


def plot_fairness_scatter(
        processed_results: Dict[str, List[pd.DataFrame]],
        title_prefix: str = "Fairness: Selection Freq vs Divergence",
        output_dir: Path = Path("."),
        divergence_metric: str = 'Divergence Proxy' # Label for the plot
) -> None:
    """
    Generates scatter plots showing benign seller selection frequency vs. their divergence proxy.
    Creates one plot per Dataset-Attack-AdvRate combo, hue by Aggregator.
    """
    scatter_points = []

    for exp_name, run_list in processed_results.items():
        if not run_list: continue

        # Aggregate frequencies and divergences across runs for this experiment
        run_freq_div_pairs = [] # List of [(freq, div), (freq, div), ...] for each run

        for df_run in run_list:
            required_cols = ['selected_sellers', 'seller_ids_all', 'round_number']
            if not all(col in df_run.columns for col in required_cols): continue
            df_valid = df_run.dropna(subset=['selected_sellers', 'round_number'])
            if df_valid.empty: continue

            seller_ids_all = df_run['seller_ids_all'].iloc[0] if len(df_run) > 0 and isinstance(df_run['seller_ids_all'].iloc[0], list) else []
            benign_ids = [s for s in seller_ids_all if isinstance(s, str) and s.startswith('bn_')]
            if not benign_ids: continue

            all_selections = [item for sublist in df_valid['selected_sellers'] if isinstance(sublist, list) for item in sublist]
            selection_counts = Counter(all_selections)
            n_rounds = df_valid['round_number'].max() + 1
            if n_rounds <= 0: continue

            run_pairs = []
            for bid in benign_ids:
                divergence = get_seller_divergence(bid)
                if divergence is not None and not np.isnan(divergence):
                    freq = selection_counts.get(bid, 0) / n_rounds
                    run_pairs.append((freq, divergence))
            if run_pairs:
                 run_freq_div_pairs.append(run_pairs) # Store pairs for this run

        # If we collected data, add points to the main list for plotting
        if run_freq_div_pairs:
            # Combine pairs from all runs for this experiment setup
            # Each point represents one seller from one run
            # Alternatively, average freq/div per seller across runs? Let's plot all points first.
            all_pairs_for_exp = [pair for run_pairs in run_freq_div_pairs for pair in run_pairs]
            for freq, divergence in all_pairs_for_exp:
                 scatter_points.append({
                     'experiment': exp_name, # Keep original experiment name if needed
                     'frequency': freq,
                     'divergence': divergence,
                     # Add pre-extracted parameters for grouping plots
                     'dataset': run_list[0]['exp_dataset'].iloc[0],
                     'aggregator': run_list[0]['exp_aggregator'].iloc[0],
                     'attack': run_list[0]['exp_attack'].iloc[0],
                     'adv_rate': run_list[0]['exp_adv_rate'].iloc[0]
                 })

    if not scatter_points:
        logging.warning("No data found for fairness scatter plot.")
        return

    df_scatter = pd.DataFrame(scatter_points)

    # Plot: Separate plot per Dataset-Attack-AdvRate combo, hue by Aggregator
    for (dataset_name, attack_name, adv_rate_val), group_data in df_scatter.groupby(['dataset', 'attack', 'adv_rate']):
        if dataset_name == 'unknown' or group_data.empty: continue

        plt.figure(figsize=(10, 7))
        try:
            sns.scatterplot(
                data=group_data,
                x='divergence',
                y='frequency',
                hue='aggregator',
                style='aggregator', # Use different markers as well
                alpha=0.6, # Make points slightly transparent if many points overlap
                s=50 # Adjust point size
            )

            attack_str = f"Attack: {attack_name}" + (f" ({int(adv_rate_val*100)}%)" if adv_rate_val > 0 else "")
            plt.title(f"{title_prefix}\nDataset: {dataset_name.upper()}, {attack_str}")
            plt.xlabel(divergence_metric)
            plt.ylabel("Benign Seller Selection Frequency")
            plt.legend(title='Aggregator', loc='best', fontsize='small')
            # Optional: Add grid lines
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            save_path = output_dir / f"fairness_scatter_{dataset_name}_{attack_name}_{int(adv_rate_val*100)}pct.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plot: {save_path}")

        except Exception as e:
             logging.error(f"Failed to generate fairness scatter plot for {dataset_name}, {attack_name}: {e}")
        finally:
            plt.close()


# --- Main Execution Logic ---

if __name__ == "__main__":
    # --- Configuration ---
    # Base directory where objective folders (like 'baselines', 'backdoor_attack') reside
    RESULTS_BASE_DIR = "./experiment_results" # MODIFY AS NEEDED
    # Directory to save generated plots and summary CSV
    OUTPUT_DIR = Path("./analysis_plots_marketplace_revised") # MODIFY AS NEEDED
    # Target accuracy for Cost of Convergence calculation
    TARGET_ACC_FOR_COC = 0.75 # MODIFY AS NEEDED
    # Optional: Specify exactly which objective folders to load, or None to load all
    # OBJECTIVES_TO_LOAD = ['baselines', 'backdoor_attacks_cifar', 'privacy_analysis']
    OBJECTIVES_TO_LOAD = None # Set to None to load all objectives found

    # --- Setup ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    # Load results using the modified function
    raw_results = load_all_results(
        RESULTS_BASE_DIR,
        csv_filename="round_results.csv",
        objectives=OBJECTIVES_TO_LOAD
    )

    if not raw_results:
        logging.error("No results were loaded. Exiting analysis.")
        exit()

    # --- 2. Preprocess Data ---
    # Apply preprocessing to each run's DataFrame
    processed_results: Dict[str, List[pd.DataFrame]] = {}
    for exp_name, run_dfs in raw_results.items():
        processed_dfs = []
        for df in run_dfs:
            try:
                processed_df = preprocess_data(df.copy()) # Use copy to avoid modifying original
                if not processed_df.empty:
                    processed_dfs.append(processed_df)
            except Exception as e:
                logging.error(f"Error preprocessing run for experiment {exp_name}: {e}", exc_info=True)
        if processed_dfs:
             processed_results[exp_name] = processed_dfs

    if not processed_results:
         logging.error("Preprocessing failed for all loaded data. Exiting analysis.")
         exit()

    # --- 3. Calculate Aggregated Marketplace Metrics ---
    summary_metrics = []
    for exp_name, run_list in processed_results.items():
        # Store calculated metrics for each run
        run_vals = {
            'coc': [], 'fairness_diff': [], 'gini': [],
            'entropy': [], 'stability': [], 'fair_corr_rho': [], 'fair_corr_p': []
        }
        for df_run in run_list:
            # Calculate metrics for this run, append NaN if calculation fails
            run_vals['coc'].append(calculate_cost_of_convergence(df_run, TARGET_ACC_FOR_COC))
            run_vals['fairness_diff'].append(calculate_fairness_differential(df_run))
            run_vals['gini'].append(calculate_gini_coefficient(df_run))
            run_vals['entropy'].append(calculate_selection_entropy(df_run))
            run_vals['stability'].append(calculate_selection_stability(df_run))

            corr_result = calculate_fairness_correlation(df_run)
            if corr_result:
                run_vals['fair_corr_rho'].append(corr_result[0])
                run_vals['fair_corr_p'].append(corr_result[1])
            else:
                run_vals['fair_corr_rho'].append(np.nan)
                run_vals['fair_corr_p'].append(np.nan)

        # Aggregate metrics across runs (mean and std)
        agg_record = {'experiment_setup': exp_name}
        # Add extracted parameters for easier filtering/grouping of summary
        if run_list:
            first_run = run_list[0]
            # Add objective_name and experiment_name_part if they exist from loading
            for orig_param in ['objective_name', 'experiment_name_part']:
                 agg_record[orig_param] = first_run[orig_param].iloc[0] if orig_param in first_run else 'unknown'

            # Add extracted parameters like dataset, aggregator etc.
            for param in ['exp_dataset', 'exp_aggregator', 'exp_attack', 'exp_adv_rate', 'exp_buyer_mode', 'exp_discovery_f']:
                 agg_record[param] = first_run[param].iloc[0] if param in first_run else 'unknown'


        for metric_key, values in run_vals.items():
            values_np = np.array(values, dtype=float) # Ensure float for nan handling
            valid_values = values_np[~np.isnan(values_np)] # Filter NaNs for calculation
            if valid_values.size > 0: # Check if there are any non-NaN values
                agg_record[f"{metric_key}_mean"] = np.mean(valid_values)
                # Use sample std dev (ddof=1), requires at least 2 points
                agg_record[f"{metric_key}_std"] = np.std(valid_values, ddof=1) if valid_values.size > 1 else 0.0
            else:
                agg_record[f"{metric_key}_mean"] = np.nan
                agg_record[f"{metric_key}_std"] = np.nan

        summary_metrics.append(agg_record)

    # Create and save the summary DataFrame
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv_path = OUTPUT_DIR / "marketplace_metrics_summary.csv"
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        logging.info(f"Marketplace metrics summary saved to: {summary_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save summary CSV: {e}")


    # --- 4. Define Experiment Groups for Plotting ---
    # **IMPORTANT**: Customize these groups based on your combined experiment names
    # (e.g., 'objective_experimentname') and the specific comparisons needed.
    all_exp_keys = list(processed_results.keys())

    # Example Groupings (ADAPT THESE based on your combined names):
    experiment_groups = {
        # Baselines (No Attack) - Per Dataset
        "Baseline_CIFAR": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'cifar' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'none'],
        "Baseline_FMNIST": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'fmnist' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'none'],
        "Baseline_AGNEWS": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'agnews' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'none'],
        "Baseline_TREC": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'trec' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'none'],

        # Backdoor Attacks - Compare Aggregators (e.g., CIFAR, 30% Adv)
        "Backdoor_CIFAR_30pct": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'cifar' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'backdoor' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_adv_rate'].iloc[0] == 0.3],
        "Backdoor_FMNIST_30pct": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'fmnist' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'backdoor' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_adv_rate'].iloc[0] == 0.3],
        "Backdoor_AGNEWS_30pct": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'agnews' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'backdoor' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_adv_rate'].iloc[0] == 0.3],

        # Backdoor Attacks - Compare Adv Rates (e.g., CIFAR, FLTrust)
        "Backdoor_CIFAR_FLTrust_vs_AdvRate": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'cifar' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] == 'backdoor' and summary_df.loc[summary_df['experiment_setup']==k, 'exp_aggregator'].iloc[0] == 'fltrust'],

        # Label Flipping Attacks - Compare Aggregators (e.g., CIFAR, 30% Adv)
        "LabelFlip_CIFAR_30pct": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'cifar' and 'label' in summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] and summary_df.loc[summary_df['experiment_setup']==k, 'exp_adv_rate'].iloc[0] == 0.3],
        "LabelFlip_FMNIST_30pct": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'fmnist' and 'label' in summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] and summary_df.loc[summary_df['experiment_setup']==k, 'exp_adv_rate'].iloc[0] == 0.3],

        # Adaptive/Sybil Attacks - Compare Aggregators (e.g., CIFAR, 30% Adv)
        "Sybil_CIFAR_30pct": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'cifar' and ('sybil' in summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0] or 'mimicry' in summary_df.loc[summary_df['experiment_setup']==k, 'exp_attack'].iloc[0]) and summary_df.loc[summary_df['experiment_setup']==k, 'exp_adv_rate'].iloc[0] == 0.3],

         # Privacy Attack Experiments (Group all where privacy was measured - check summary df)
        "Privacy_Analysis_CIFAR": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'exp_dataset'].iloc[0] == 'cifar' and not pd.isna(summary_df.loc[summary_df['experiment_setup']==k, 'attack_psnr_mean'].iloc[0])],

        # Groups based on objective folder name (if loaded)
        # "Objective_Baselines": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'objective_name'].iloc[0] == 'baselines'],
        # "Objective_Backdoor": [k for k in all_exp_keys if summary_df.loc[summary_df['experiment_setup']==k, 'objective_name'].iloc[0].startswith('backdoor')],

        # You might need more specific groups depending on your analysis needs.
    }

    # Filter out empty groups
    experiment_groups = {g_name: g_keys for g_name, g_keys in experiment_groups.items() if g_keys}

    if not experiment_groups:
         logging.warning("No experiment groups were defined or matched any experiments based on current filters. Limited plots will be generated.")

    # --- 5. Generate Plots for each Group ---
    for group_name, exp_keys in experiment_groups.items():
        logging.info(f"--- Generating plots for group: {group_name} ---")
        group_output_dir = OUTPUT_DIR / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)

        # Get the subset of results for this group
        group_results = {k: processed_results[k] for k in exp_keys if k in processed_results}
        if not group_results:
             logging.warning(f"No processed results found for group {group_name}. Skipping.")
             continue

        # Get the subset of the summary DataFrame for this group
        group_summary_df = summary_df[summary_df['experiment_setup'].isin(exp_keys)]
        if group_summary_df.empty:
             logging.warning(f"Summary data is empty for group {group_name}. Skipping aggregated plots.")
             # continue # Or allow non-aggregated plots to proceed

        # --- Plot Standard Performance Metrics (Time Series & Final) ---
        # Fig 1a-X (Accuracy), Fig 2a-X (Acc/ASR), Fig 3a-X (Acc), Fig 4a-X (Acc/ASR)
        plot_metric_comparison(group_results, 'global_acc', f"{group_name} - Global Accuracy vs Round",
                               save_path=group_output_dir / "ts_global_acc.png")
        plot_metric_comparison(group_results, 'global_loss', f"{group_name} - Global Loss vs Round",
                               save_path=group_output_dir / "ts_global_loss.png")
        plot_metric_comparison(group_results, 'global_asr', f"{group_name} - Attack Success Rate (ASR) vs Round",
                               save_path=group_output_dir / "ts_global_asr.png")

        # Fig 1z (Acc), Fig 2y/2z (Acc/ASR), Fig 3z (Acc), Fig 4z (Acc/ASR)
        plot_final_round_comparison(group_results, 'global_acc', f"{group_name} - Final Global Accuracy", higher_is_better=True,
                                    save_path=group_output_dir / "final_global_acc.png")
        plot_final_round_comparison(group_results, 'global_asr', f"{group_name} - Final Attack Success Rate (ASR)", higher_is_better=False,
                                    save_path=group_output_dir / "final_global_asr.png")

        # --- Plot Privacy Metrics (Time Series & Final) ---
        # Fig 6a/b (PSNR/SSIM curves)
        plot_metric_comparison(group_results, 'attack_psnr', f"{group_name} - Gradient Attack PSNR vs Round",
                               save_path=group_output_dir / "ts_attack_psnr.png")
        plot_metric_comparison(group_results, 'attack_ssim', f"{group_name} - Gradient Attack SSIM vs Round",
                               save_path=group_output_dir / "ts_attack_ssim.png")
        plot_metric_comparison(group_results, 'attack_label_acc', f"{group_name} - Gradient Attack Label Accuracy vs Round",
                               save_path=group_output_dir / "ts_attack_label_acc.png")

        # Fig 6z (Final PSNR/SSIM bar) + Optional Label Acc
        plot_final_round_comparison(group_results, 'attack_psnr', f"{group_name} - Final Gradient Attack PSNR", higher_is_better=True, # Higher PSNR often means better reconstruction
                                    save_path=group_output_dir / "final_attack_psnr.png")
        plot_final_round_comparison(group_results, 'attack_ssim', f"{group_name} - Final Gradient Attack SSIM", higher_is_better=True,
                                    save_path=group_output_dir / "final_attack_ssim.png")
        plot_final_round_comparison(group_results, 'attack_label_acc', f"{group_name} - Final Gradient Attack Label Accuracy", higher_is_better=True,
                                    save_path=group_output_dir / "final_attack_label_acc.png")


        # --- Plot Selection Metrics (Time Series & Optional Final) ---
        # Optional Fig 2/3/4 (Malicious selection rate)
        plot_metric_comparison(group_results, 'malicious_selection_rate', f"{group_name} - Malicious Selection Rate vs Round",
                               save_path=group_output_dir / "ts_malicious_selection_rate.png")
        # Optional Fig 6.7 (Num Sellers Selected)
        plot_metric_comparison(group_results, 'num_sellers_selected', f"{group_name} - Number of Sellers Selected vs Round",
                               save_path=group_output_dir / "ts_num_sellers_selected.png")

        # --- Plot Aggregated Marketplace Metrics (Bar plots from summary_df) ---
        if not group_summary_df.empty:
            # Fig 6.7 (CoC, Gini, Entropy, Stability, Fairness Corr/Diff)
            plot_aggregated_metric_bar(group_summary_df, 'coc', f"{group_name} - Cost of Convergence (Acc >= {TARGET_ACC_FOR_COC})",
                                    "Total Selections", higher_is_better=False, save_path=group_output_dir / "agg_coc.png")
            plot_aggregated_metric_bar(group_summary_df, 'gini', f"{group_name} - Gini Coefficient of Selections",
                                    "Gini (0=Equal, 1=Max Ineq.)", higher_is_better=False, save_path=group_output_dir / "agg_gini.png")
            plot_aggregated_metric_bar(group_summary_df, 'entropy', f"{group_name} - Normalized Selection Entropy (Diversity)",
                                    "Norm. Entropy (0=Min, 1=Max Div.)", higher_is_better=True, save_path=group_output_dir / "agg_entropy.png")
            plot_aggregated_metric_bar(group_summary_df, 'stability', f"{group_name} - Selection Stability (Avg. Jaccard)",
                                    "Avg. Jaccard Index", higher_is_better=True, save_path=group_output_dir / "agg_stability.png")
            plot_aggregated_metric_bar(group_summary_df, 'fairness_diff', f"{group_name} - Fairness Differential (Benign Rate - Malicious Rate)",
                                    "Selection Rate Difference", higher_is_better=True, save_path=group_output_dir / "agg_fairness_diff.png")
            plot_aggregated_metric_bar(group_summary_df, 'fair_corr_rho', f"{group_name} - Fairness Correlation (Selection Freq vs Divergence)",
                                    "Spearman Rho", higher_is_better=False, # Often aiming for low correlation (fairness)
                                    save_path=group_output_dir / "agg_fairness_corr_rho.png")
        else:
             logging.warning(f"Skipping aggregated marketplace plots for group {group_name} due to empty summary data.")


        # --- Plot Scatter Plots for Trade-offs ---
        # Fig 6.8 (Trade-offs), e.g., Acc vs ASR, Acc vs Privacy, Acc vs Fairness/Cost
        plot_scatter_comparison(group_results, 'global_acc', 'global_asr', f"{group_name} - Trade-off: Final Accuracy vs ASR",
                                xlabel="Final Global Accuracy", ylabel="Final Global ASR", save_path=group_output_dir / "scatter_acc_vs_asr.png")
        plot_scatter_comparison(group_results, 'global_acc', 'attack_psnr', f"{group_name} - Trade-off: Final Accuracy vs Privacy (PSNR)",
                                xlabel="Final Global Accuracy", ylabel="Final Attack PSNR", save_path=group_output_dir / "scatter_acc_vs_psnr.png")
        # Add more scatter plots as needed, potentially using aggregated data from summary_df if appropriate


    # --- 6. Generate Specific Comparison Plots (across groups or all data) ---

    # --- Plot Discovery Effect (Fig 5a) ---
    # Uses all relevant processed results (those with discovery_f parameter)
    logging.info("--- Generating Discovery Effect plots ---")
    # Filter processed_results to only include experiments where discovery_f is not NaN
    discovery_results = {k: v for k, v in processed_results.items() if not pd.isna(v[0]['exp_discovery_f'].iloc[0])}
    if discovery_results:
        plot_discovery_effect(discovery_results, metric='global_acc', output_dir=OUTPUT_DIR / "Comparison_DiscoveryEffect")
        plot_discovery_effect(discovery_results, metric='global_asr', output_dir=OUTPUT_DIR / "Comparison_DiscoveryEffect") # If relevant
    else:
        logging.warning("No experiments with valid 'exp_discovery_f' found for discovery effect plots.")

    # --- Plot Buyer Mode Effect (Fig 5b) ---
    # Uses all relevant processed results (those with known buyer_mode)
    logging.info("--- Generating Buyer Mode Effect plots ---")
    buyer_mode_results = {k: v for k, v in processed_results.items() if v[0]['exp_buyer_mode'].iloc[0] in ['unbiased', 'biased']}
    if buyer_mode_results:
        plot_buyer_mode_effect(buyer_mode_results, metric='global_acc', output_dir=OUTPUT_DIR / "Comparison_BuyerModeEffect")
        plot_buyer_mode_effect(buyer_mode_results, metric='global_asr', output_dir=OUTPUT_DIR / "Comparison_BuyerModeEffect", higher_is_better=False) # If relevant
    else:
         logging.warning("No experiments with 'unbiased' or 'biased' buyer modes found for buyer mode effect plots.")


    # --- Plot Fairness Scatter (Optional Fig 6.7) ---
    logging.info("--- Generating Fairness Scatter plots ---")
    # Potentially filter results here if only certain scenarios are relevant for this plot
    # e.g., fairness_scatter_results = {k: v for k,v in processed_results.items() if v[0]['exp_discovery_f'] ... }
    plot_fairness_scatter(processed_results, output_dir=OUTPUT_DIR / "Comparison_FairnessScatter")


    logging.info("--- Analysis complete. Plots saved in: %s ---", OUTPUT_DIR)