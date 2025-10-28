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

# --- Parsers for ALL folder name patterns ---

HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")
SEED_REGEX_1 = re.compile(r"run_\d+_seed_(\d+)") # e.g., run_1_seed_43
SEED_REGEX_2 = re.compile(r".*_seed-(\d+)") # e.g., ds-texas100_..._seed-42

# This will find '..._iid' or '..._noniid'
DATA_SETTING_REGEX = re.compile(r".*_(?P<data_setting>iid|noniid)$")

# This finds the main experiment name (which also has the data setting)
EXP_NAME_REGEX = re.compile(
    r"(?P<base_name>step1_tune_fedavg_.+?)(_(?P<data_setting_exp>iid|noniid))?$"
)

def parse_hp_from_name(name: str) -> Dict[str, Any]:
    """Parses 'opt_Adam_lr_0.001_epochs_5'"""
    match = HPARAM_REGEX.match(name)
    if not match:
        return {}
    data = match.groupdict()
    try:
        data['lr'] = float(data['lr'])
        data['epochs'] = int(data['epochs'])
        return data
    except ValueError:
        return {}

def parse_sub_exp_name(name: str) -> Dict[str, str]:
    """Parses 'ds-purchase100_model-mlp_agg-fedavg'"""
    params = {}
    try:
        parts = name.split('_')
        for part in parts:
            if '-' in part:
                key, value = part.split('-', 1)
                # We only care about these specific keys
                if key in ['ds', 'model', 'agg']:
                    params[key] = value
    except Exception as e:
        logger.warning(f"Could not parse sub-experiment name: {name} ({e})")
    return params

def parse_seed_from_name(name: str) -> int:
    """Parses seed from 'run_1_seed_43' or '..._seed-42'"""
    for regex in [SEED_REGEX_1, SEED_REGEX_2]:
        match = regex.match(name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    if "seed" in name:
        try:
            return int(name.split('_')[-1])
        except (ValueError, IndexError):
            pass
    return -1

def parse_exp_context(name: str) -> Dict[str, str]:
    """Parses 'step1_tune_..._mlp_iid'"""
    match = EXP_NAME_REGEX.match(name)
    if match:
        data = match.groupdict()
        return {
            "base_name": data.get("base_name") or "unknown_base",
            "data_setting": data.get("data_setting_exp") # Can be None
        }

    # Check if it's just a data_setting string
    ds_match = DATA_SETTING_REGEX.match(name)
    if ds_match:
        return {"data_setting": ds_match.group("data_setting")}

    return {}


def find_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Finds all final_metrics.json files and parses context by
    walking up the parent directories, robustly.
    """
    logger.info(f"🔍 Scanning recursively for results in: {results_dir}...")
    metrics_files = list(results_dir.rglob("final_metrics.json"))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {results_dir}.")
        return []

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")
    all_results = []

    for metrics_file in metrics_files:
        try:
            run_dir = metrics_file.parent
            if not (run_dir / ".success").exists():
                logger.debug(f"Skipping failed/incomplete run: {run_dir.name}")
                continue

            # --- Initialize ALL possible fields ---
            record = {
                "base_name": None,
                "data_setting": None,
                "optimizer": None, "lr": None, "epochs": None,
                "ds": None, "model": None, "agg": None,
                "seed": -1,
                "full_path": str(metrics_file) # For debugging
            }

            # --- Walk up the path from the file ---
            current_path = metrics_file.parent
            while str(current_path).startswith(str(results_dir)) and current_path != results_dir.parent:
                folder_name = current_path.name

                # Try to parse HPs (opt_...)
                if record["optimizer"] is None:
                    hps = parse_hp_from_name(folder_name)
                    if hps: record.update(hps)

                # Try to parse exp context (step1_tune_..._iid)
                exp_context = parse_exp_context(folder_name)
                if exp_context:
                    if record["base_name"] is None and "base_name" in exp_context:
                        record["base_name"] = exp_context["base_name"]
                    if record["data_setting"] is None and "data_setting" in exp_context:
                        record["data_setting"] = exp_context["data_setting"]

                # Try to parse sub-exp name (ds-...)
                if record["ds"] is None:
                    sub_exp_params = parse_sub_exp_name(folder_name)
                    if sub_exp_params: record.update(sub_exp_params)

                # Try to parse seed
                if record["seed"] == -1:
                    seed = parse_seed_from_name(folder_name)
                    if seed != -1: record["seed"] = seed

                current_path = current_path.parent

            # --- Post-processing / Sanity Checks ---

            # Fill in blanks
            if record["data_setting"] is None:
                record["data_setting"] = "unknown_ds" # Explicitly mark as unknown

            # If base_name is still unknown, fall back to ds/model
            if record["base_name"] is None:
                if record["ds"]:
                    record["base_name"] = f"{record['ds']}_{record['model']}"
                else:
                    record["base_name"] = "unknown_base"

            # Check for minimum required info
            if record["optimizer"] is None:
                logger.warning(f"Could not find HP folder (opt_...) for {metrics_file}. Skipping.")
                continue

            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

            # Add metrics to record
            record.update(metrics_data)
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

    # Define ALL columns that describe a configuration
    all_config_cols = [
        "base_name", "data_setting",
        "optimizer", "lr", "epochs",
        "ds", "model", "agg"
    ]

    # Find which of these config columns actually exist in our DataFrame
    config_cols = [c for c in all_config_cols if c in df.columns]

    # Find metrics
    metric_cols_to_agg = []
    known_cols = set(config_cols + ["seed", "status", "full_path"])

    for col in df.columns:
        if col not in known_cols and pd.api.types.is_numeric_dtype(df[col]):
            metric_cols_to_agg.append(col)

    if not metric_cols_to_agg:
        logger.error("❌ No numeric metric columns found to aggregate (e.g., 'test_acc', 'acc').")
        logger.error(f"DataFrame columns found: {df.columns.tolist()}")
        return

    logger.info(f"📊 Aggregating by: {config_cols}")
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
        # FillNa for config columns to prevent dropping rows
        df[config_cols] = df[config_cols].fillna("N/A")
        agg_df = df.groupby(config_cols).agg(**agg_ops).reset_index()
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        return

    agg_df[std_cols] = agg_df[std_cols].fillna(0)

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

    # Show all config columns + primary metrics + run count
    display_cols = config_cols.copy()
    display_cols.append(primary_metric)
    primary_std = primary_metric.replace("mean_", "std_")
    if primary_std in agg_df.columns:
        display_cols.append(primary_std)
    display_cols.append("num_success_runs")

    # Add other common/key metrics if they exist
    for m_key in ['backdoor_asr', 'adv_success_rate', 'test_loss', 'target_class_acc']:
        m_mean = f"mean_{m_key}"
        if m_mean in agg_df.columns and m_mean != primary_metric:
            display_cols.append(m_mean)

    final_display_cols = [c for c in display_cols if c in agg_df.columns]
    print(agg_df[final_display_cols].to_string(index=False, float_format="%.4f"))

    # --- 6. Save Full Results to CSV ---
    output_csv = results_dir / "aggregated_results_UNIFIED.csv"
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
        default="./results",
        help="The root directory where all experiment results are stored (default: ./results)"
    )
    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) # Set wide for all columns

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