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
# Fill this with your "best-case" (IID, FedAvg, no attack)
# accuracy from your Step 1 tuning run.
IID_BASELINES = {
    "Texas100": 0.6250,  # <-- EXAMPLE (Fill with your value)
    "Purchase100": 0.6002,  # <-- EXAMPLE (Fill with your value)
    "CIFAR10": 0.8248,  # <-- EXAMPLE (Fill with your value)
    "CIFAR100": 0.5536,  # <-- EXAMPLE (Fill with your value)
    "TREC": 0.7985,  # <-- EXAMPLE (Fill with your value)
}
USABLE_THRESHOLD = 0.90  # i.e., "achieved 90% of the IID baseline accuracy"


# ==============================================================================

def parse_scenario_name_robust(name: str) -> Dict[str, str]:
    """
    Parses 'step4_train_sens_...' OR 'step2.5_find_hps_...'
    """
    # --- THIS IS THE FIX ---
    # Try the Step 4 regex first
    regex_step4 = re.compile(
        r"step4_train_sens_(?P<defense>[\w-]+)_(?P<attack_state>no_attack|with_attack)_(?P<modality>\w+)_(?P<dataset>[\w\d]+)"
    )
    match = regex_step4.match(name)

    if match:
        data = match.groupdict()
        data["scenario"] = name  # Add the full name for grouping
        return data

    # Try the Step 2.5 regex if the first one failed
    regex_step2_5 = re.compile(
        r"step2.5_find_hps_(?P<defense>[\w-]+)_(?P<modality>\w+)_(?P<dataset>[\w\d]+)"
    )
    match = regex_step2_5.match(name)

    if not match:
        logger.warning(f"Could not parse scenario name: {name}")
        return {}

    data = match.groupdict()
    data["scenario"] = name
    # We know this is *always* the "with_attack" state
    data["attack_state"] = "with_attack"
    # --- END FIX ---
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

    # --- THIS IS THE FIX ---
    # Find metrics files from EITHER Step 4 or Step 2.5
    search_pattern_4 = "step4_train_sens_*/*/run_*/final_metrics.json"
    search_pattern_2_5 = "step2.5_find_hps_*/*/run_*/final_metrics.json"

    metrics_files = list(root_dir.glob(search_pattern_4))
    metrics_files.extend(list(root_dir.glob(search_pattern_2_5)))
    # --- END FIX ---

    if not metrics_files:
        logger.error(
            f"❌ ERROR: No 'final_metrics.json' files found in {root_dir} matching the Step 4 or 2.5 structure.")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            seed_dir = metrics_file.parent
            hparam_dir = seed_dir.parent
            scenario_dir = hparam_dir.parent

            if not (seed_dir / ".success").exists():
                continue

            scenario_info = parse_scenario_name_robust(scenario_dir.name)
            hparam_info = parse_hparam_name(hparam_dir.name)

            if not scenario_info or not hparam_info:
                logger.warning(f"Skipping {metrics_file.parent}, failed to parse path.")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # This code is slightly simplified by finding the model_suffix
            # from the golden params file, but we need it from the path.
            # We will assume 'dataset' is enough for now, but
            # a better parser would get the 'model_config_name'

            record = {
                **scenario_info,
                **hparam_info,
                "seed_run": seed_dir.name,
                "test_acc": metrics.get("test_acc") or metrics.get("acc"),
                "backdoor_asr": metrics.get("backdoor_asr") or metrics.get("adv_success_rate") or 0.0,
                "score": 0.0  # Will calculate later
            }
            all_results.append(record)
        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}")

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def analyze_sensitivity(raw_df: pd.DataFrame, results_dir: Path):
    """
    Analyzes the raw results to calculate sensitivity AND
    prints the new GOLDEN_TRAINING_PARAMS dictionary.
    """
    if raw_df.empty:
        logger.warning("No data to analyze.")
        return

    if IID_BASELINES.get("CIFAR10", 0.0) == 0.0:
        logger.error("=" * 80)
        logger.error("STOP: You must fill in the 'IID_BASELINES' dictionary at the top of this script.")
        logger.error("=" * 80)
        return

    # --- 1. Aggregate across seeds ---
    group_cols = ["scenario", "defense", "attack_state", "dataset", "modality",
                  "optimizer", "lr", "epochs"]
    numeric_cols = ["test_acc", "backdoor_asr"]
    raw_df[numeric_cols] = raw_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    raw_df['backdoor_asr'] = raw_df['backdoor_asr'].fillna(0.0)
    raw_df = raw_df.dropna(subset=['test_acc'])

    hp_agg_df = raw_df.groupby(group_cols, as_index=False)[numeric_cols].mean()

    # --- 2. Calculate Relative Performance & Score for EACH HP combination ---
    hp_agg_df['iid_baseline_acc'] = hp_agg_df['dataset'].map(IID_BASELINES)
    hp_agg_df = hp_agg_df.dropna(subset=['iid_baseline_acc'])
    hp_agg_df['relative_perf'] = hp_agg_df['test_acc'] / hp_agg_df['iid_baseline_acc']
    hp_agg_df['is_usable'] = hp_agg_df['relative_perf'] >= USABLE_THRESHOLD

    # This is the key metric for finding the *best* HP
    hp_agg_df['score'] = hp_agg_df['test_acc'] - hp_agg_df['backdoor_asr']

    # --- 3. Aggregate HP stats to get final "Cost" metrics (Your Step 4 Analysis) ---
    scenario_group_cols = ["defense", "attack_state", "dataset", "modality"]
    cost_df = hp_agg_df.groupby(scenario_group_cols, as_index=False).agg(
        max_test_acc=('test_acc', 'max'),
        avg_test_acc=('test_acc', 'mean'),
        std_test_acc=('test_acc', 'std'),
        total_hp_combos=('test_acc', 'count'),
        min_asr=('backdoor_asr', 'min'),
        avg_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std')
    )
    usable_counts_df = hp_agg_df.groupby(scenario_group_cols, as_index=False)['is_usable'].sum().rename(
        columns={"is_usable": "usable_hp_count"})
    cost_df = cost_df.merge(usable_counts_df, on=scenario_group_cols)
    cost_df['iid_baseline_acc'] = cost_df['dataset'].map(IID_BASELINES)
    cost_df['relative_max_perf'] = cost_df['max_test_acc'] / cost_df['iid_baseline_acc']
    cost_df['relative_avg_perf'] = cost_df['avg_test_acc'] / cost_df['iid_baseline_acc']
    cost_df['robustness_score'] = (cost_df['relative_max_perf'] + cost_df['relative_avg_perf']) / 2
    cost_df['initialization_cost'] = 1.0 - cost_df['robustness_score']
    cost_df.loc[cost_df['attack_state'] == 'no_attack', ['min_asr', 'avg_asr', 'std_asr']] = pd.NA
    cost_df = cost_df.sort_values(by=['dataset', 'attack_state', 'initialization_cost'])

    display_cols = [
        "dataset", "attack_state", "defense", "initialization_cost",
        "min_asr", "avg_asr", "usable_hp_count",
        "relative_max_perf", "relative_avg_perf", "total_hp_combos",
    ]
    display_cols = [col for col in display_cols if col in cost_df.columns]

    print("\n" + "=" * 120)
    print(f"📊 Initialization Cost Analysis (Usable Threshold: {USABLE_THRESHOLD * 100}%)")
    print("=" * 120)
    with pd.option_context('display.max_rows', None,
                           'display.width', 1000,
                           'display.float_format', '{:,.3f}'.format):
        print(cost_df[display_cols].to_string(index=False, na_rep="N/A"))
    print("\n" + "=" * 120)

    # --- 4. NEW: Find Best HPs and Print the Dictionary ---
    print("\n" + "=" * 120)
    print("📋 New 'GOLDEN_TRAINING_PARAMS' Dictionary (from Step 2.5)")
    print("=" * 120)

    # We only care about the "with_attack" state for this
    attack_hp_df = hp_agg_df[hp_agg_df['attack_state'] == 'with_attack'].copy()

    # Find the best HPs (max score) for each group
    # We need to map 'dataset' back to 'model_config_name'
    # This is a bit of a hack; a better parser would get the model_config_name
    dataset_to_model_map = {
        "Texas100": "mlp_texas100_baseline",
        "Purchase100": "mlp_purchase100_baseline",
        "CIFAR10": "cifar10_cnn",
        "CIFAR100": "cifar100_cnn",
        "TREC": "textcnn_trec_baseline",
    }

    # Find the best row for each group
    best_hp_indices = attack_hp_df.groupby(['defense', 'dataset'])['score'].idxmax()
    best_hp_df = attack_hp_df.loc[best_hp_indices]

    print("# Copy and paste this dictionary into your 'config_common_utils.py':\n")
    print("GOLDEN_TRAINING_PARAMS = {")

    # Loop and print
    for _, row in best_hp_df.iterrows():
        # Get the key (e.g., "fltrust_cifar10_cnn")
        model_config_name = dataset_to_model_map.get(row['dataset'])
        if not model_config_name:
            continue
        key = f"{row['defense']}_{model_config_name}"

        # Build the HP dictionary
        hp_dict = {
            "training.optimizer": row['optimizer'],
            "training.learning_rate": row['lr'],
            "training.local_epochs": int(row['epochs']),
        }
        # Add SGD params if needed
        if row['optimizer'] == 'SGD':
            hp_dict["training.momentum"] = 0.9
            hp_dict["training.weight_decay"] = 5e-4  # Or your default

        print(f'    "{key}": {hp_dict},')

    print("}")
    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL Sensitivity Analysis results (Step 2.5 or 4)."
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

    if not results_path.exists():
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
        return

    try:
        raw_results_df = find_all_sensitivity_results(results_path)
        if not raw_results_df.empty:
            analyze_sensitivity(raw_results_df, results_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()