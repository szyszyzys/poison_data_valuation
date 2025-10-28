import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging

# Set up a simple logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parsers for different folder name patterns ---

HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")
EXP_NAME_REGEX = re.compile(r"step1_tune_fedavg_.*") # Finds the main experiment folder
SEED_REGEX_1 = re.compile(r"run_\d+_seed_(\d+)") # e.g., run_1_seed_43
SEED_REGEX_2 = re.compile(r".*_seed-(\d+)") # e.g., ds-texas100_..._seed-42

def parse_hp_from_name(name: str) -> Dict[str, Any]:
    """
    Parses 'opt_Adam_lr_0.001_epochs_5'
    into {'optimizer': 'Adam', 'lr': 0.001, 'epochs': 5}
    """
    match = HPARAM_REGEX.match(name)
    if not match:
        return {} # Return empty if no match

    data = match.groupdict()
    try:
        # Convert types
        data['lr'] = float(data['lr'])
        data['epochs'] = int(data['epochs'])
        return data
    except ValueError:
        logger.warning(f"Could not convert types in HP folder name: {name}")
        return {} # Return empty on conversion error

def parse_seed_from_name(name: str) -> int:
    """Parses seed from various run folder name formats."""
    match1 = SEED_REGEX_1.match(name)
    if match1:
        try:
            return int(match1.group(1))
        except ValueError:
            pass
    
    match2 = SEED_REGEX_2.match(name)
    if match2:
        try:
            return int(match2.group(1))
        except ValueError:
            pass
            
    # Fallback for old "run_0_seed_42" or similar
    if "seed" in name:
        try:
            seed_str = name.split('_')[-1]
            return int(seed_str)
        except (ValueError, IndexError):
            pass
            
    return -1 # Return -1 if no seed found

def find_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Finds all final_metrics.json files recursively and parses their context
    from the directory structure.
    
    This version is robust to different/inconsistent directory structures.
    """
    logger.info(f"🔍 Scanning recursively for results in: {results_dir}...")

    # Use rglob to find all 'final_metrics.json' files recursively
    metrics_files = list(results_dir.rglob("final_metrics.json"))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {results_dir}.")
        logger.error("Please make sure you are pointing to the correct root results directory.")
        return []

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []

    for metrics_file in metrics_files:
        try:
            run_dir = metrics_file.parent
            
            # 1. Check if the run was successful (using the .success marker)
            if not (run_dir / ".success").exists():
                logger.debug(f"Skipping failed/incomplete run: {run_dir.name}")
                continue

            # 2. Initialize context parts
            parsed_hps = {}
            exp_name = "unknown_exp"
            seed = -1
            run_name_raw = run_dir.name # Default run name

            # 3. Iterate up the parents to find context
            current_path = metrics_file.parent
            # Stop if we reach the root or are outside the results_dir
            while str(current_path).startswith(str(results_dir)) and current_path != results_dir.parent:
                folder_name = current_path.name
                
                # A. Try to parse HPs (only take the first one found)
                if not parsed_hps:
                    parsed_hps = parse_hp_from_name(folder_name)
                
                # B. Try to find the experiment name
                if EXP_NAME_REGEX.match(folder_name):
                    exp_name = folder_name
                
                # C. Try to parse seed (only take the first one found)
                if seed == -1:
                    seed = parse_seed_from_name(folder_name)
                    if seed != -1:
                        run_name_raw = folder_name # Store name we got seed from

                current_path = current_path.parent

            # 4. Final check if seed wasn't found in a parent
            if seed == -1:
                seed = parse_seed_from_name(run_dir.name)
                run_name_raw = run_dir.name

            # 5. Handle missing HPs (e.g., the 'swapped' structure)
            if not parsed_hps:
                # Check if the 'exp_name' was actually the HP folder
                parsed_hps = parse_hp_from_name(exp_name)
                if not parsed_hps:
                    logger.warning(f"Could not find HP folder (opt_...) for {metrics_file}. Skipping.")
                    continue

            # 6. Load the metrics JSON
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            record = {
                "exp_name": exp_name,
                "run_name_raw": run_name_raw,  # For debugging
                **parsed_hps,  # Adds 'optimizer', 'lr', 'epochs'
                "seed": seed,
                "status": "success",
                **metrics
            }
            all_results.append(record)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {metrics_file}: {e}", exc_info=True)
            continue

    return all_results

def analyze_results(results: List[Dict[str, Any]], results_dir: Path):
    """Aggregates results and prints summary table."""

    if not results:
        logger.warning("No successful results to analyze.")
        return

    df = pd.DataFrame(results)

    # --- 1. Identify Config vs. Metric Columns ---
    
    # Config columns are what define a single experiment configuration
    # We group by 'exp_name' + all parsed HP columns
    known_cols = {"exp_name", "run_name_raw", "seed", "status"}
    
    metric_cols_to_agg = []
    # Start with exp_name, as it contains dataset/model/iid info
    config_cols = ["exp_name"] 
    
    for col in df.columns:
        if col in known_cols:
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a known HP config column
            if col in ['lr', 'epochs']:
                config_cols.append(col)
            else:
                # Otherwise, it's a metric to aggregate
                metric_cols_to_agg.append(col)
        elif col == 'optimizer': # String config col
             config_cols.append(col)

    if not metric_cols_to_agg:
        logger.error("❌ No numeric metric columns found to aggregate (e.g., 'test_acc').")
        logger.error(f"DataFrame columns found: {df.columns.tolist()}")
        return

    logger.info(f"📊 Aggregating by: {sorted(list(set(config_cols)))}")
    logger.info(f"📊 Found metrics to aggregate: {metric_cols_to_agg}")

    # --- 2. Build Aggregation Dictionary Dynamically ---
    agg_ops = {}
    mean_cols = []
    std_cols = []

    for col in metric_cols_to_agg:
        agg_ops[f"mean_{col}"] = pd.NamedAgg(column=col, aggfunc="mean")
        agg_ops[f"std_{col}"] = pd.NamedAgg(column=col, aggfunc="std")
        mean_cols.append(f"mean_{col}")
        std_cols.append(f"std_{col}")

    agg_ops["num_success_runs"] = pd.NamedAgg(column="seed", aggfunc="count")

    # --- 3. Perform Aggregation ---
    try:
        # Ensure config_cols are unique
        config_cols = sorted(list(set(config_cols)))
        agg_df = df.groupby(config_cols).agg(**agg_ops).reset_index()
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        logger.error(f"Attempted to group by: {config_cols}")
        logger.error(f"Original DataFrame columns: {df.columns.tolist()}")
        return

    # Fill NaN std (for single-seed runs) with 0 for cleaner sorting/display
    for col in std_cols:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0)

    # --- 4. Sort by a Primary Metric ---
    primary_metric = "mean_test_acc"
    if primary_metric not in agg_df.columns:
        if "mean_acc" in agg_df.columns: # Fallback for tabular (which often just uses 'acc')
             primary_metric = "mean_acc"
        elif mean_cols:
            primary_metric = mean_cols[0] # Fallback to first available metric
            logger.warning(f"'mean_test_acc' and 'mean_acc' not found. Sorting by '{primary_metric}'.")
        else:
            primary_metric = "num_success_runs" # Last resort

    agg_df = agg_df.sort_values(by=primary_metric, ascending=False)

    # --- 5. Display Summary Table ---
    print("\n" + "="*80)
    print(f"📈 Aggregated Experiment Results (sorted by {primary_metric})")
    print("="*80)

    # Dynamically create display columns, prioritizing key metrics
    display_cols = config_cols.copy() # Show all config columns

    # Add primary metric and its std dev
    display_cols.append(primary_metric)
    primary_std = primary_metric.replace("mean_", "std_")
    if primary_std in agg_df.columns:
        display_cols.append(primary_std)

    # Add other common/key metrics if they exist
    for m_key in ['backdoor_asr', 'adv_success_rate', 'test_loss', 'target_class_acc']:
        m_mean = f"mean_{m_key}"
        if m_mean in agg_df.columns and m_mean != primary_metric:
            display_cols.append(m_mean)

    display_cols.append("num_success_runs")

    # Filter display_cols to only those that actually exist in the aggregated DataFrame
    final_display_cols = [c for c in display_cols if c in agg_df.columns]

    print(agg_df[final_display_cols].to_string(index=False, float_format="%.4f"))

    # --- 6. Save Full Results to CSV ---
    output_csv = results_dir / "aggregated_results_ROBUST.csv"
    try:
        agg_df.to_csv(output_csv, index=False, float_format="%.4f")
        logger.info(f"\n✅ Full aggregated results saved to: {output_csv}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save aggregated results to CSV: {e}")

    print("\n" + "="*80)
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL experiment results from the run_parallel.py script."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        default="./results", # Assuming your configs save to a 'results' dir
        help="The root directory where all experiment results are stored (default: ./results)"
    )

    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve() # Use resolve for clean paths

    # Set pandas display options for nice console output
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    try:
        results_list = find_all_results(results_path)
        if results_list:
            analyze_results(results_list, results_path)
    except FileNotFoundError:
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()