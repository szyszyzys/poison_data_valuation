import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Set up a simple logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")


def parse_hp_tuning_folder_name(name: str) -> Dict[str, Any]:
    """
    Parses 'opt_Adam_lr_0.001_epochs_5'
    into {'optimizer': 'Adam', 'lr': 0.001, 'epochs': 5}
    """
    match = HPARAM_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse HP tuning folder name: {name}")
        return {}  # Return empty if no match

    data = match.groupdict()
    try:
        # Convert types
        data['lr'] = float(data['lr'])
        data['epochs'] = int(data['epochs'])
        return data
    except ValueError:
        logger.warning(f"Could not convert types in HP tuning folder name: {name}")
        return {}  # Return empty on conversion error


def parse_sub_exp_name(name: str) -> Dict[str, str]:
    """
    Parses 'model-cnn_agg-fedavg_def-krum_adv-backdoor'
    into {'model': 'cnn', 'agg': 'fedavg', 'def': 'krum', 'adv': 'backdoor'}

    This is flexible for any key-value pairs in your sub-experiment names.
    """
    params = {}
    try:
        parts = name.split('_')
        for part in parts:
            if '-' in part:
                # Split only on the first '-'
                key, value = part.split('-', 1)
                params[key] = value
    except Exception as e:
        logger.warning(f"Could not parse sub-experiment name: {name} ({e})")
    return params


def find_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Finds all final_metrics.json files recursively and parses their context
    from the directory structure created by run_parallel.py.

    Expected structure:
    <results_dir> / <exp_name> / <sub_exp_name> / <run_name> / final_metrics.json
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
            sub_exp_dir = run_dir.parent
            exp_dir = sub_exp_dir.parent

            # 1. Check if the run was successful (using the .success marker)
            if not (run_dir / ".success").exists():
                logger.debug(f"Skipping failed/incomplete run: {run_dir.name}")
                continue

            # 2. Parse context from folder names
            exp_name = exp_dir.name  # e.g., "main_summary_cifar10_cnn"
            sub_exp_name = sub_exp_dir.name  # e.g., "model-cnn_agg-fedavg_def-krum"
            run_name = run_dir.name  # e.g., "run_0_seed_42"

            # 3. Parse parameters from the sub-experiment name
            if "step1_tune_iid" in exp_name or "step4_train_sens" in exp_name:
                parsed_params = parse_hp_tuning_folder_name(sub_exp_dir.name)
            else:
                # Use the original parser for other experiment types
                parsed_params = parse_sub_exp_name(sub_exp_dir.name)
            # 4. Parse seed
            try:
                seed = int(run_name.split('_')[-1])
            except ValueError:
                logger.warning(f"Could not parse seed from: {run_name}")
                seed = -1

            # 5. Load the metrics JSON
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            record = {
                "exp_name": exp_name,
                "sub_exp_name": sub_exp_name,  # Still useful as a raw identifier
                **parsed_params,  # Add the correctly parsed HPs or sub_exp params
                "seed": seed,
                "status": "success",
                "metrics": metrics  # <-- Keep metrics in a nested dict
            }
            all_results.append(record)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {metrics_file}: {e}")
            continue

    return all_results


def analyze_results(results: List[Dict[str, Any]], results_dir: Path):
    """Aggregates results and prints summary table."""

    if not results:
        logger.warning("No successful results to analyze.")
        return

    df = pd.json_normalize(results, sep='_')
    # --- 1. Identify Config vs. Metric Columns ---

    # Config columns are what define a single experiment configuration
    # We'll use 'exp_name' and 'sub_exp_name' as the unique key
    config_cols = ["exp_name", "sub_exp_name"]

    # Dynamically find all parsed parameter columns (e.g., model, agg, def)
    # These are columns that are NOT the core ones or metrics
    known_cols = {"exp_name", "sub_exp_name", "seed", "status"}

    # Find all columns that are numeric and not known config cols
    # These are our metrics to aggregate
    metric_cols_to_agg = [col for col in df.columns if col.startswith('metrics_')]

    # And your config columns are everything else:
    config_cols = [
        col for col in df.columns
        if not col.startswith('metrics_') and col not in ['seed', 'status']
    ]

    if not metric_cols_to_agg:
        logger.error("❌ No numeric metric columns found to aggregate (e.g., 'test_acc').")
        return

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
        agg_df = df.groupby(config_cols).agg(**agg_ops).reset_index()
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        logger.error("This can happen if sub_exp_name is not unique per exp_name.")
        logger.error("Original DataFrame columns:", df.columns)
        return

    # Fill NaN std (for single-seed runs) with 0 for cleaner sorting/display
    for col in std_cols:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0)

    # --- 4. Sort by a Primary Metric ---
    primary_metric = "mean_test_acc"
    if primary_metric not in agg_df.columns:
        # Fallback to sorting by the first available metric
        if mean_cols:
            primary_metric = mean_cols[0]
            logger.warning(f"'mean_test_acc' not found. Sorting by '{primary_metric}'.")
        else:
            primary_metric = "num_success_runs"  # Last resort

    agg_df = agg_df.sort_values(by=primary_metric, ascending=False)

    # --- 5. Display Summary Table ---
    print("\n" + "=" * 80)
    print(f"📈 Aggregated Experiment Results (sorted by {primary_metric})")
    print("=" * 80)

    # Dynamically create display columns, prioritizing key metrics
    display_cols = config_cols.copy()

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
    output_csv = results_dir / "aggregated_results.csv"
    try:
        agg_df.to_csv(output_csv, index=False, float_format="%.4f")
        logger.info(f"\n✅ Full aggregated results saved to: {output_csv}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save aggregated results to CSV: {e}")

    print("\n" + "=" * 80)
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL experiment results from the run_parallel.py script."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        default="./results",  # Assuming your configs save to a 'results' dir
        help="The root directory where all experiment results are stored (default: ./results)"
    )

    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve()  # Use resolve for clean paths

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
