import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parsers (Unchanged) ---
HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")
SEED_REGEX_1 = re.compile(r"run_\d+_seed_(\d+)")
SEED_REGEX_2 = re.compile(r".*_seed-(\d+)")
DATA_SETTING_REGEX = re.compile(r".*_(?P<data_setting>iid|noniid)$")
EXP_NAME_REGEX = re.compile(
    r"(?P<base_name>(?:step|new_step)[\w\.-]+?)(_(?P<data_setting_exp>iid|noniid))?$"
)

def parse_hp_from_name(name: str) -> Dict[str, Any]:
    match = HPARAM_REGEX.match(name)
    if not match: return {}
    try:
        data = match.groupdict(); data['lr'] = float(data['lr']); data['epochs'] = int(data['epochs']); return data
    except ValueError:
        return {}

def parse_sub_exp_name(name: str) -> Dict[str, str]:
    params = {}
    try:
        parts = name.split('_');
        for part in parts:
            if '-' in part:
                key, value = part.split('-', 1)
                if key in ['ds', 'model', 'agg']: params[key] = value
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
        return {"base_name": data.get("base_name") or "unknown_base", "data_setting": data.get("data_setting_exp")}
    ds_match = DATA_SETTING_REGEX.match(name)
    if ds_match: return {"data_setting": ds_match.group("data_setting")}
    return {}

def find_all_results(results_dir: Path, clip_mode: str, exp_filter: str) -> List[Dict[str, Any]]:
    logger.info(f"🔍 Scanning recursively for results in: {results_dir}...")
    logger.info(f"   Filtering by --clip_mode: '{clip_mode}'")
    logger.info(f"   Filtering by --exp_filter: must contain '{exp_filter}'")

    # 1. Glob all json files
    metrics_files = list(results_dir.rglob("final_metrics.json"))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {results_dir}.")
        return []

    logger.info(f"   Found {len(metrics_files)} JSON files. Filtering now...")
    all_results = []

    for metrics_file in metrics_files:
        try:
            # --- 2. FAST EXCLUSION (Avoid Reading File) ---
            # If we only want 'local_clip' (standard), we strictly IGNORE paths with 'nolocalclip'
            if clip_mode == "local_clip" and "nolocalclip" in str(metrics_file):
                # logger.debug(f"Skipping nolocalclip file: {metrics_file.name}")
                continue

            # If we ONLY want 'no_local_clip', we skip paths that DON'T have it
            if clip_mode == "no_local_clip" and "nolocalclip" not in str(metrics_file):
                continue

            run_dir = metrics_file.parent
            if not (run_dir / ".success").exists():
                # logger.debug(f"Skipping failed/incomplete run: {run_dir.name}")
                continue

            record = {
                "base_name": None, "data_setting": None, "optimizer": None,
                "lr": None, "epochs": None, "ds": None, "model": None, "agg": None,
                "seed": -1, "full_path": str(metrics_file),
                "clip_setting": "unknown"
            }

            current_path = metrics_file.parent
            while str(current_path).startswith(str(results_dir)) and current_path != results_dir.parent:
                folder_name = current_path.name

                if record["optimizer"] is None:
                    hps = parse_hp_from_name(folder_name)
                    if hps: record.update(hps)

                if record["base_name"] is None:
                    # Detect clip setting from folder name
                    if "nolocalclip" in folder_name:
                        record["clip_setting"] = "no_local_clip"
                        folder_name = folder_name.replace("_nolocalclip", "")
                    else:
                        record["clip_setting"] = "local_clip"

                    exp_context = parse_exp_context(folder_name)
                    if exp_context and "base_name" in exp_context:
                        record.update(exp_context)

                if record["ds"] is None:
                    sub_exp_params = parse_sub_exp_name(folder_name)
                    if sub_exp_params: record.update(sub_exp_params)

                if record["seed"] == -1:
                    seed = parse_seed_from_name(folder_name)
                    if seed != -1: record["seed"] = seed

                current_path = current_path.parent

            # Final check on base name filter
            if record.get("base_name") is None or exp_filter not in record["base_name"]:
                continue

            # Check for missing HP folder
            if record["optimizer"] is None:
                continue

            # --- 3. LOAD DATA (Only if we passed filters) ---
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

            record.update(metrics_data)
            all_results.append(record)

        except Exception as e:
            logger.warning(f"Warning: Could not process file {metrics_file}: {e}")
            continue

    logger.info(f"✅ Successfully processed {len(all_results)} runs after filtering.")
    return all_results

# ==============================================================================
# === 1. USER ACTION REQUIRED: FILL IN YOUR IID BASELINES ===
# ==============================================================================
IID_BASELINES = {
    "texas100": 0.6250,
    "purchase100": 0.6002,
    "cifar10": 0.8248,
    "cifar100": 0.5536,
    "trec": 0.7985,
}
USABLE_THRESHOLD = 0.90
# ==============================================================================

def analyze_sensitivity(raw_df: pd.DataFrame, results_dir: Path, dataset_filter: Optional[str] = None):
    if raw_df.empty:
        logger.warning("No data to analyze.")
        return

    if IID_BASELINES.get("cifar10", 0.0) == 0.0:
        logger.error("STOP: You must fill in the 'IID_BASELINES' dictionary.")
        return

    # --- Pre-processing ---
    raw_df['defense'] = raw_df['agg']
    raw_df['attack_state'] = 'with_attack'
    raw_df['dataset'] = raw_df['ds']

    numeric_cols = ["acc", "asr"]
    raw_df[numeric_cols] = raw_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    raw_df['asr'] = raw_df['asr'].fillna(0.0)
    raw_df = raw_df.dropna(subset=['acc'])

    raw_df = raw_df.rename(columns={"acc": "test_acc", "asr": "backdoor_asr"})

    # Grouping
    group_cols = ["defense", "attack_state", "dataset",
                  "optimizer", "lr", "epochs", "clip_setting"]
    available_group_cols = [col for col in group_cols if col in raw_df.columns]

    # 1. Average across seeds
    hp_agg_df = raw_df.groupby(available_group_cols, as_index=False)[["test_acc", "backdoor_asr"]].mean()

    # 2. Metrics
    hp_agg_df['iid_baseline_acc'] = hp_agg_df['dataset'].map(IID_BASELINES)
    hp_agg_df['relative_perf'] = hp_agg_df['test_acc'] / hp_agg_df['iid_baseline_acc']
    hp_agg_df['is_usable'] = hp_agg_df['relative_perf'] >= USABLE_THRESHOLD
    hp_agg_df['score'] = hp_agg_df['test_acc'] - hp_agg_df['backdoor_asr']

    # --- OPTION A: DETAILED DATASET VIEW ---
    if dataset_filter:
        logger.info(f"--- Filtering for dataset: '{dataset_filter}' ---")
        detailed_df = hp_agg_df[hp_agg_df['dataset'].str.lower() == dataset_filter.lower()]

        if detailed_df.empty:
            logger.warning(f"No results found for {dataset_filter}.")
            return

        display_cols = ["defense", "optimizer", "lr", "epochs",
                        "test_acc", "backdoor_asr", "score", "is_usable"]

        detailed_df = detailed_df.sort_values(by=["defense", "score"], ascending=[True, False])

        print("\n" + "=" * 100)
        print(f"📊 DETAIL VIEW: {dataset_filter.upper()}")
        print("=" * 100)
        # Improved formatting for visuals
        with pd.option_context('display.max_rows', None, 'display.width', 1000, 'display.float_format', '{:,.4f}'.format):
            print(detailed_df[display_cols].to_string(index=False))
        print("=" * 100 + "\n")
        return

    # --- OPTION B: SUMMARY VIEW ---
    # Find Best HP per (Dataset, Defense)
    best_indices = hp_agg_df.groupby(['dataset', 'defense'])['score'].idxmax()
    best_df = hp_agg_df.loc[best_indices].sort_values(by=['dataset', 'score'], ascending=[True, False])

    display_cols = ["dataset", "defense", "optimizer", "lr", "epochs",
                    "test_acc", "backdoor_asr", "score"]

    print("\n" + "=" * 100)
    print(f"🏆 BEST CONFIGURATIONS (Sorted by Score = Acc - ASR)")
    print("=" * 100)
    with pd.option_context('display.max_rows', None, 'display.width', 1000, 'display.float_format', '{:,.4f}'.format):
        print(best_df[display_cols].to_string(index=False))
    print("=" * 100 + "\n")

    # Generate Dictionary (Optional)
    print("📋 GOLDEN_TRAINING_PARAMS = {")
    dataset_to_model = {
        "texas100": "mlp_texas100_baseline",
        "purchase100": "mlp_purchase100_baseline",
        "cifar10": "cifar10_cnn",
        "cifar100": "cifar100_cnn",
        "trec": "textcnn_trec_baseline",
    }
    for _, row in best_df.iterrows():
        model_name = dataset_to_model.get(row['dataset'])
        if not model_name: continue
        key = f"{row['defense']}_{model_name}_{row['clip_setting']}"
        print(f'    "{key}": {{"training.optimizer": "{row["optimizer"]}", "training.learning_rate": {row["lr"]}, "training.local_epochs": {int(row["epochs"])}}},')
    print("}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, nargs="?", default="./results")

    # === CHANGED DEFAULT TO 'local_clip' TO AVOID NOLOCALCLIP ===
    parser.add_argument("--clip_mode", type=str, default="local_clip",
                        choices=["all", "local_clip", "no_local_clip"],
                        help="Default is 'local_clip', which explicitly ignores '_nolocalclip' folders.")

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--exp_filter", type=str, default="step2.5_find_hps")

    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve()

    if not results_path.exists():
        logger.error(f"❌ Directory not found: {results_path}")
        return

    raw_results = find_all_results(results_path, args.clip_mode, args.exp_filter)

    if raw_results:
        analyze_sensitivity(pd.DataFrame(raw_results), results_path, args.dataset)
    else:
        logger.warning("No results found.")

if __name__ == "__main__":
    main()