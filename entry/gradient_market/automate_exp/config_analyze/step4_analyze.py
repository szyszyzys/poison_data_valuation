import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regex to parse the hyperparameter folder name, e.g., "opt_Adam_lr_0.001_epochs_5"
HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")

# ==============================================================================
# === 1. USER ACTION REQUIRED: FILL IN YOUR IID BASELINES ===
# ==============================================================================
# Before running, fill this with your "best-case" (IID, FedAvg, no attack)
# accuracy from your Step 1 tuning run. This is the 100% mark.
#
# EXAMPLE VALUES:
# IID_BASELINES = {
#     "texas100": 0.580,    # From your earlier tuning
#     "purchase100": 0.583, # From your earlier tuning
#     "cifar10": 0.821,     # Example: 82.1%
#     "cifar100": 0.55,
#     "trec": 0.91,
# }
#
IID_BASELINES = {
    "texas100": 0.0,  # <-- FILL ME
    "purchase100": 0.0,  # <-- FILL ME
    "cifar10": 0.0,  # <-- FILL ME
    "cifar100": 0.0,  # <-- FILL ME
    "trec": 0.0,  # <-- FILL ME
}

# Define what "usable" means (for the 'usable_hp_count' column), relative to the IID baseline
USABLE_THRESHOLD = 0.90  # i.e., "achieved 90% of the IID baseline accuracy"

# ==============================================================================

def parse_scenario_name_robust(name: str) -> Dict[str, str]:
    """
    Parses 'step4_train_sens_fltrust_with_attack_image_cifar10'
    (Adjusted regex from your v1 script to match your v2 generator)
    """
    regex = re.compile(
        r"step4_train_sens_(?P<defense>[\w-]+)_(?P<attack_state>no_attack|with_attack)_(?P<modality>\w+)_(?P<dataset>[\w\d]+)"
    )
    match = regex.match(name)
    if not match:
        logger.warning(f"Could not parse scenario name: {name}")
        return {}

    data = match.groupdict()
    data["scenario"] = name # Add the full name for grouping
    return data


def parse_hparam_name(name: str) -> Dict[str, Any]:
    """
    Parses 'opt_Adam_lr_0.001_epochs_5'
    """
    match = HPARAM_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse hparam name: {name}")
        return {}
    data = match.groupdict()
    try:
        data['lr'] = float(data['lr'])
        data['epochs'] = int(data['epochs'])
        return data
    except ValueError:
        return {}


def find_all_sensitivity_results(root_dir: Path) -> pd.DataFrame:
    """Finds all final_metrics.json files and parses their full context."""
    logger.info(f"🔍 Scanning for all results in: {root_dir}...")

    # Find metrics files from the correct step4 directory structure
    metrics_files = list(root_dir.glob("step4_train_sens_*/*/run_*/final_metrics.json"))

    if not metrics_files:
        # Fallback for the other structure
        metrics_files.extend(list(root_dir.glob("step4_train_sens_*/*/*/final_metrics.json")))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {root_dir} matching the Step 4 structure.")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            # Path: .../<scenario_name>/<hparam_name>/<seed_name>/final_metrics.json
            seed_dir = metrics_file.parent
            hparam_dir = seed_dir.parent
            scenario_dir = hparam_dir.parent

            # 1. Check for success
            if not (seed_dir / ".success").exists():
                continue

            # 2. Parse context from folder names
            scenario_info = parse_scenario_name_robust(scenario_dir.name)
            hparam_info = parse_hparam_name(hparam_dir.name)

            if not scenario_info or not hparam_info:
                logger.warning(f"Skipping {metrics_file.parent}, failed to parse path.")
                continue

            # 3. Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # 4. Store the combined record
            record = {
                **scenario_info,
                **hparam_info,
                "seed_run": seed_dir.name,
                "test_acc": metrics.get("test_acc") or metrics.get("acc"), # Handle 'acc'
                "backdoor_asr": metrics.get("backdoor_asr") or metrics.get("adv_success_rate"), # Handle 'adv_success_rate'
            }
            all_results.append(record)
        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}")

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def analyze_sensitivity(raw_df: pd.DataFrame):
    """
    Analyzes the raw results to calculate robustness and initialization cost.
    """
    if raw_df.empty:
        logger.warning("No data to analyze.")
        return

    # Check if baselines are filled
    if IID_BASELINES.get("cifar10", 0.0) == 0.0: # Check one example
        logger.error("=" * 80)
        logger.error("STOP: You must fill in the 'IID_BASELINES' dictionary at the top of this script.")
        logger.error("=" * 80)
        return

    # --- 1. Aggregate across seeds ---
    # Group by each unique HP combination and average the seeds
    group_cols = ["scenario", "defense", "attack_state", "dataset", "modality",
                  "optimizer", "lr", "epochs"]

    # Handle cases where some metrics might be missing
    numeric_cols = ["test_acc", "backdoor_asr"]
    # Ensure numeric types
    raw_df[numeric_cols] = raw_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Fill N/A ASRs with 0 (e.g., for benign runs)
    raw_df['backdoor_asr'] = raw_df['backdoor_asr'].fillna(0.0)

    # Drop any row that failed to parse (e.g., no test_acc)
    raw_df = raw_df.dropna(subset=['test_acc'])

    hp_agg_df = raw_df.groupby(group_cols, as_index=False)[numeric_cols].mean()


    # --- 2. Calculate Relative Performance for EACH HP combination ---
    # Map baseline accuracy to each row
    hp_agg_df['iid_baseline_acc'] = hp_agg_df['dataset'].map(IID_BASELINES)

    # Handle missing baselines
    if hp_agg_df['iid_baseline_acc'].isnull().any():
        missing = hp_agg_df[hp_agg_df['iid_baseline_acc'].isnull()]['dataset'].unique()
        logger.error(f"Missing baseline values for: {missing}. Please update IID_BASELINES.")
        hp_agg_df = hp_agg_df.dropna(subset=['iid_baseline_acc'])

    # Calculate relative performance for this HP combo
    hp_agg_df['relative_perf'] = hp_agg_df['test_acc'] / hp_agg_df['iid_baseline_acc']

    # Determine if this HP combo is "usable" (for the count column)
    hp_agg_df['is_usable'] = hp_agg_df['relative_perf'] >= USABLE_THRESHOLD

    # --- 3. Aggregate HP stats to get final "Cost" metrics ---
    scenario_group_cols = ["defense", "attack_state", "dataset", "modality"]

    # === THIS IS THE MERGED LOGIC ===
    # Get stats for test_acc AND backdoor_asr
    cost_df = hp_agg_df.groupby(scenario_group_cols, as_index=False).agg(
        # Your original 'test_acc' aggregations
        max_test_acc=('test_acc', 'max'),
        avg_test_acc=('test_acc', 'mean'),
        std_test_acc=('test_acc', 'std'),
        total_hp_combos=('test_acc', 'count'),

        # Add 'backdoor_asr' aggregations
        min_asr=('backdoor_asr', 'min'),
        avg_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std')
    )
    # === END MERGED LOGIC ===

    # Get "usable" count
    usable_counts_df = hp_agg_df.groupby(scenario_group_cols, as_index=False)['is_usable'].sum().rename(columns={"is_usable": "usable_hp_count"})

    # Merge counts back
    cost_df = cost_df.merge(usable_counts_df, on=scenario_group_cols)

    # --- 4. Calculate Final Relative Metrics and "Initialization Cost" ---
    cost_df['iid_baseline_acc'] = cost_df['dataset'].map(IID_BASELINES)

    # Relative performance of the *best* HP
    cost_df['relative_max_perf'] = cost_df['max_test_acc'] / cost_df['iid_baseline_acc']

    # Relative performance on *average*
    cost_df['relative_avg_perf'] = cost_df['avg_test_acc'] / cost_df['iid_baseline_acc']

    # Robustness = 0.5 * (Best Perf) + 0.5 * (Avg Perf)
    cost_df['robustness_score'] = (cost_df['relative_max_perf'] + cost_df['relative_avg_perf']) / 2

    # Initialization Cost = 1.0 - Robustness
    cost_df['initialization_cost'] = 1.0 - cost_df['robustness_score']

    # --- 5. Clean up ASR for 'no_attack' rows ---
    # Set ASR stats to N/A for benign runs, as they are meaningless
    cost_df.loc[cost_df['attack_state'] == 'no_attack', ['min_asr', 'avg_asr', 'std_asr']] = pd.NA

    # --- 6. Display Final Table ---
    cost_df = cost_df.sort_values(by=['dataset', 'attack_state', 'initialization_cost'])

    display_cols = [
        "dataset",
        "attack_state",
        "defense",
        "initialization_cost", # <-- Your final metric (0 = best)
        "min_asr",             # <-- ADDED: Best-case robustness
        "avg_asr",             # <-- ADDED: Average-case robustness
        "usable_hp_count",     # <-- How many HPs worked
        "relative_max_perf", # <-- Best HP's score
        "relative_avg_perf", # <-- Average HP's score
        "total_hp_combos",
    ]

    # Filter for columns that actually exist
    display_cols = [col for col in display_cols if col in cost_df.columns]

    print("\n" + "=" * 120)
    print(f"📊 Initialization Cost Analysis (Usable Threshold: {USABLE_THRESHOLD*100}%)")
    print(f"   'initialization_cost' = 0.0 (Best), 1.0 (Worst)")
    print(f"   'min_asr' / 'avg_asr' = ASR for 'with_attack' scenarios (Lower is better)")
    print("=" * 120)

    with pd.option_context('display.max_rows', None,
                           'display.width', 1000,
                           'display.float_format', '{:,.3f}'.format):
        print(cost_df[display_cols].to_string(index=False, na_rep="N/A"))

    print("\n" + "=" * 120)

    # Save the file
    output_csv = Path(args.results_dir) / "step4_sensitivity_summary_BEST.csv"
    try:
        cost_df.to_csv(output_csv, index=False, float_format="%.4f")
        logger.info(f"✅ Summary results saved to: {output_csv}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")

    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL Sensitivity Analysis results (Step 4)."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="The root results directory for the Step 4 run (e.g., './results/')"
    )
    args = parser.parse_args()
    results_path = Path(args.results_dir)

    if not results_path.exists():
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
        return

    try:
        raw_results_df = find_all_sensitivity_results(results_path)
        analyze_sensitivity(raw_results_df)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()