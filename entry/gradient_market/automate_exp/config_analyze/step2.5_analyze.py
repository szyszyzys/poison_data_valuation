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

# --- Parsers from your Step 1 Analyzer ---
HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")
SEED_REGEX_1 = re.compile(r"run_\d+_seed_(\d+)")
SEED_REGEX_2 = re.compile(r".*_seed-(\d+)")
DATA_SETTING_REGEX = re.compile(r".*_(?P<data_setting>iid|noniid)$")
EXP_NAME_REGEX = re.compile(
    r"(?P<base_name>step(1|2.5|3|4|5)_.+?)(_(?P<data_setting_exp>iid|noniid))?$"
)


def parse_hp_from_name(name: str) -> Dict[str, Any]:
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
    params = {}
    try:
        parts = name.split('_')
        for part in parts:
            if '-' in part:
                key, value = part.split('-', 1)
                if key in ['ds', 'model', 'agg']:
                    params[key] = value
    except Exception as e:
        logger.warning(f"Could not parse sub-experiment name: {name} ({e})")
    return params


def parse_seed_from_name(name: str) -> int:
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
    match = EXP_NAME_REGEX.match(name)
    if match:
        data = match.groupdict()
        return {
            "base_name": data.get("base_name") or "unknown_base",
            "data_setting": data.get("data_setting_exp")
        }
    ds_match = DATA_SETTING_REGEX.match(name)
    if ds_match:
        return {"data_setting": ds_match.group("data_setting")}
    return {}


# --- This is YOUR find_all_results function (it's better) ---
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

            record = {
                "base_name": None, "data_setting": None, "optimizer": None,
                "lr": None, "epochs": None, "ds": None, "model": None, "agg": None,
                "seed": -1, "full_path": str(metrics_file)
            }

            current_path = metrics_file.parent
            while str(current_path).startswith(str(results_dir)) and current_path != results_dir.parent:
                folder_name = current_path.name

                if record["optimizer"] is None:
                    hps = parse_hp_from_name(folder_name)
                    if hps: record.update(hps)

                exp_context = parse_exp_context(folder_name)
                if exp_context:
                    if record["base_name"] is None and "base_name" in exp_context:
                        record["base_name"] = exp_context["base_name"]
                    if record["data_setting"] is None and "data_setting" in exp_context:
                        record["data_setting"] = exp_context["data_setting"]

                if record["ds"] is None:
                    sub_exp_params = parse_sub_exp_name(folder_name)
                    if sub_exp_params: record.update(sub_exp_params)

                if record["seed"] == -1:
                    seed = parse_seed_from_name(folder_name)
                    if seed != -1: record["seed"] = seed

                current_path = current_path.parent

            # --- THIS IS THE FIX for TypeError ---
            # Check for None *before* checking the content of the string
            if record["base_name"] is None or "step2.5_find_hps" not in record["base_name"]:
                logger.debug(f"Skipping non-Step2.5 file: {metrics_file}")
                continue
            # --- END FIX ---

            if record["optimizer"] is None:
                logger.warning(f"Could not find HP folder (opt_...) for {metrics_file}. Skipping.")
                continue

            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

            record.update(metrics_data)
            all_results.append(record)

        except Exception as e:
            logger.warning(f"Warning: Could not process file {metrics_file}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully processed {len(all_results)} Step 2.5 runs.")
    return all_results


# ==============================================================================
# === 1. USER ACTION REQUIRED: FILL IN YOUR IID BASELINES ===
# ==============================================================================
IID_BASELINES = {
    "Texas100": 0.6250,  # <-- EXAMPLE (Fill with your value)
    "Purchase100": 0.6002,  # <-- EXAMPLE (Fill with your value)
    "CIFAR10": 0.8248,  # <-- EXAMPLE (Fill with your value)
    "CIFAR100": 0.5536,  # <-- EXAMPLE (Fill with your value)
    "TREC": 0.7985,  # <-- EXAMPLE (Fill with your value)
}
USABLE_THRESHOLD = 0.0


# ==============================================================================

# --- This is the analysis function from analyze_step4.py ---
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

    # --- 1. Add 'defense' and 'attack_state' columns ---
    raw_df['defense'] = raw_df['agg']
    raw_df['attack_state'] = 'with_attack'  # All Step 2.5 runs are "with_attack"
    raw_df['dataset'] = raw_df['ds']

    # --- 2. Aggregate across seeds ---
    group_cols = ["scenario", "defense", "attack_state", "dataset", "modality",
                  "optimizer", "lr", "epochs"]
    available_group_cols = [col for col in group_cols if col in raw_df.columns]

    # Use metric names from your JSON ('acc' and 'asr')
    # Your example JSON only has 'acc', but attacked runs MUST have 'asr'
    numeric_cols = ["acc", "asr"]
    raw_df[numeric_cols] = raw_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    raw_df['asr'] = raw_df['asr'].fillna(0.0)  # Fill benign/missing ASR with 0
    raw_df = raw_df.dropna(subset=['acc'])

    raw_df = raw_df.rename(columns={"acc": "test_acc", "asr": "backdoor_asr"})
    numeric_cols = ["test_acc", "backdoor_asr"]

    hp_agg_df = raw_df.groupby(available_group_cols, as_index=False)[numeric_cols].mean()

    # --- 3. Calculate Relative Performance & Score for EACH HP combination ---
    hp_agg_df['iid_baseline_acc'] = hp_agg_df['dataset'].map(IID_BASELINES)
    hp_agg_df = hp_agg_df.dropna(subset=['iid_baseline_acc'])
    hp_agg_df['relative_perf'] = hp_agg_df['test_acc'] / hp_agg_df['iid_baseline_acc']
    hp_agg_df['is_usable'] = hp_agg_df['relative_perf'] >= USABLE_THRESHOLD
    hp_agg_df['score'] = hp_agg_df['test_acc'] - hp_agg_df['backdoor_asr']

    # --- 4. Aggregate HP stats to get final "Cost" metrics (Your Step 4 Analysis) ---
    scenario_group_cols = ["defense", "attack_state", "dataset", "modality"]
    available_scenario_group_cols = [col for col in scenario_group_cols if col in hp_agg_df.columns]

    cost_df = hp_agg_df.groupby(available_scenario_group_cols, as_index=False).agg(
        max_test_acc=('test_acc', 'max'),
        avg_test_acc=('test_acc', 'mean'),
        std_test_acc=('test_acc', 'std'),
        total_hp_combos=('test_acc', 'count'),
        min_asr=('backdoor_asr', 'min'),
        avg_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std')
    )
    usable_counts_df = hp_agg_df.groupby(available_scenario_group_cols, as_index=False)['is_usable'].sum().rename(
        columns={"is_usable": "usable_hp_count"})
    cost_df = cost_df.merge(usable_counts_df, on=available_scenario_group_cols)
    cost_df['iid_baseline_acc'] = cost_df['dataset'].map(IID_BASELINES)
    cost_df['relative_max_perf'] = cost_df['max_test_acc'] / cost_df['iid_baseline_acc']
    cost_df['relative_avg_perf'] = cost_df['avg_test_acc'] / cost_df['iid_baseline_acc']
    cost_df['robustness_score'] = (cost_df['relative_max_perf'] + cost_df['relative_avg_perf']) / 2
    cost_df['initialization_cost'] = 1.0 - cost_df['robustness_score']
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

    # --- 5. NEW: Find Best HPs and Print the Dictionary ---
    print("\n" + "=" * 120)
    print("📋 New 'GOLDEN_TRAINING_PARAMS' Dictionary (from Step 2.5)")
    print("=" * 120)

    attack_hp_df = hp_agg_df  # All rows are "with_attack"

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
        model_config_name = dataset_to_model_map.get(row['dataset'])
        if not model_config_name:
            continue
        key = f"{row['defense']}_{model_config_name}"

        hp_dict = {
            "training.optimizer": row['optimizer'],
            "training.learning_rate": row['lr'],
            "training.local_epochs": int(row['epochs']),
        }
        if row['optimizer'] == 'SGD':
            hp_dict["training.momentum"] = 0.9
            hp_dict["training.weight_decay"] = 5e-4

        print(f'    "{key}": {hp_dict},')

    print("}")
    print("\n" + "=" * 120)


# --- This is YOUR main function ---
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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    try:
        raw_results_df = find_all_results(results_path)

        # --- THIS IS THE FIX for AttributeError ---
        # Check if the LIST is empty, not if the DATAFRAME is empty
        if not raw_results_df:
            # --- END FIX ---
            logger.warning("No valid Step 2.5 results found to analyze.")
        else:
            # Convert list of dicts to DataFrame *before* passing to analyze
            df = pd.DataFrame(raw_results_df)
            analyze_sensitivity(df, results_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()