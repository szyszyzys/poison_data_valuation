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

# --- NEW: Regex to find data setting ---
DATA_SETTING_REGEX = re.compile(r".*_(?P<data_setting>iid|noniid)$")


def parse_hp_from_name(name: str) -> Dict[str, Any]:
    """
    Parses 'opt_Adam_lr_0.001_epochs_5'
    into {'optimizer': 'Adam', 'lr': 0.001, 'epochs': 5}
    """
    match = HPARAM_REGEX.match(name)
    if not match:
        return {}
    data = match.groupdict()
    try:
        data['lr'] = float(data['lr'])
        data['epochs'] = int(data['epochs'])
        return data
    except ValueError:
        logger.warning(f"Could not convert types in HP folder name: {name}")
        return {}

def parse_seed_from_name(name: str) -> int:
    """Parses seed from various run folder name formats."""
    for regex in [SEED_REGEX_1, SEED_REGEX_2]:
        match = regex.match(name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass

    # Fallback
    if "seed" in name:
        try:
            seed_str = name.split('_')[-1]
            return int(seed_str)
        except (ValueError, IndexError):
            pass

    return -1

def parse_data_setting_from_name(name: str) -> str:
    """Parses 'iid' or 'noniid' from a folder name."""
    match = DATA_SETTING_REGEX.match(name)
    if match:
        return match.group("data_setting")
    return ""


def find_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Finds all final_metrics.json files recursively and parses their context
    from the directory structure.

    This version is robust to different/inconsistent directory structures.
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

            # 2. Initialize context parts
            parsed_hps = {}
            data_setting = "unknown"
            exp_name = "unknown_exp"
            seed = -1

            # 3. Iterate up the parents to find context
            current_path = metrics_file.parent

            # --- This is the key logic ---
            # We walk up and fill in the blanks. We take the *first* one we find
            # (the one closest to the json file).
            while str(current_path).startswith(str(results_dir)) and current_path != results_dir.parent:
                folder_name = current_path.name

                if not parsed_hps:
                    parsed_hps = parse_hp_from_name(folder_name)

                if exp_name == "unknown_exp" and EXP_NAME_REGEX.match(folder_name):
                    exp_name = folder_name

                if seed == -1:
                    seed = parse_seed_from_name(folder_name)

                # --- NEW: Explicitly look for iid/noniid ---
                if data_setting == "unknown":
                    ds = parse_data_setting_from_name(folder_name)
                    if ds:
                        data_setting = ds
                # --- End New ---

                current_path = current_path.parent
            # --- End while loop ---

            # 4. Handle "swapped" paths where exp_name is missing
            #    but HPs are found and data_setting is still unknown.
            if exp_name == "unknown_exp" and parsed_hps:
                logger.debug(f"Found 'swapped' path. Using folder name as exp_name for: {metrics_file}")
                # Fallback to the parent of the HP folder
                exp_name = metrics_file.parent.parent.parent.name

            # 5. Handle missing HPs (if folder was 'opt_...' but parser failed)
            if not parsed_hps:
                logger.warning(f"Could not find HP folder (opt_...) for {metrics_file}. Skipping.")
                continue

            # 6. Load the metrics JSON
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            record = {
                "exp_name": exp_name,
                "data_setting": data_setting, # <-- NEW COLUMN
                **parsed_hps,
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

    if "data_setting" not in df.columns:
        logger.error("❌ 'data_setting' column was not created. This is a script bug.")
        return

    # --- 1. Identify Config vs. Metric Columns ---

    # --- THIS IS THE KEY FIX ---
    # We now explicitly group by 'data_setting'
    config_cols = ["exp_name", "data_setting", "optimizer", "lr", "epochs"]
    # ---------------------------

    # Find all config cols that actually exist in the DataFrame
    existing_config_cols = [c for c in config_cols if c in df.columns]

    if len(existing_config_cols) < 3: # e.g., missing optimizer, lr, etc.
        logger.error(f"❌ Missing critical config columns. Found: {df.columns.tolist()}")
        return

    # Find metrics
    metric_cols_to_agg = []
    known_config_cols = set(existing_config_cols + ["seed", "status"])
    for col in df.columns:
        if col not in known_config_cols and pd.api.types.is_numeric_dtype(df[col]):
            metric_cols_to_agg.append(col)

    if not metric_cols_to_agg:
        logger.error("❌ No numeric metric columns found to aggregate (e.g., 'test_acc').")
        return

    logger.info(f"📊 Aggregating by: {existing_config_cols}")
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
        agg_df = df.groupby(existing_config_cols).agg(**agg_ops).reset_index()
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        logger.error(f"Original DataFrame columns:", df.columns)
        return

    agg_df[std_cols] = agg_df[std_cols].fillna(0)

    # --- 4. Sort by a Primary Metric ---
    primary_metric = "mean_test_acc"
    if primary_metric not in agg_df.columns:
        if "mean_acc" in agg_df.columns:
             primary_metric = "mean_acc"
        elif mean_cols:
            primary_metric = mean_cols[0]
            logger.warning(f"'mean_test_acc' and 'mean_acc' not found. Sorting by '{primary_metric}'.")
        else:
            primary_metric = "num_success_runs"

    agg_df = agg_df.sort_values(by=primary_metric, ascending=False)

    # --- 5. Display Summary Table ---
    print("\n" + "="*80)
    print(f"📈 Aggregated Experiment Results (sorted by {primary_metric})")
    print("="*80)

    display_cols = existing_config_cols.copy() # Show all config columns
    display_cols.append(primary_metric)
    primary_std = primary_metric.replace("mean_", "std_")
    if primary_std in agg_df.columns:
        display_cols.append(primary_std)

    for m_key in ['backdoor_asr', 'adv_success_rate', 'test_loss', 'target_class_acc']:
        m_mean = f"mean_{m_key}"
        if m_mean in agg_df.columns and m_mean != primary_metric:
            display_cols.append(m_mean)
    display_cols.append("num_success_runs")

    final_display_cols = [c for c in display_cols if c in agg_df.columns]
    print(agg_df[final_display_cols].to_string(index=False, float_format="%.4f"))

    # --- 6. Save Full Results to CSV ---
    output_csv = results_dir / "aggregated_results_FINAL.csv"
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