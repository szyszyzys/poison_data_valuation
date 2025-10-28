# FILE: analyze_attack_sensitivity.py

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

# --- Parsers for the Step 5 Directory Structure ---

# Parses: step5_atk_sens_adv_fltrust_backdoor_image_cifar10
SCENARIO_REGEX = re.compile(
    r"step5_atk_sens_(?P<sweep_type>adv|poison)_(?P<defense>[\w-]+)_(?P<attack_type>backdoor|labelflip)_(?P<modality>\w+)_(?P<dataset>[\w\d]+)"
)

# Parses: adv_0.1_poison_0.5
HP_REGEX = re.compile(r"adv_(?P<adv_rate>[\d\.]+)_poison_(?P<poison_rate>[\d\.]+)")

def parse_scenario_name(name: str) -> Dict[str, Any]:
    """Parses the main scenario folder name."""
    match = SCENARIO_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse scenario name: {name}")
        return {}
    return match.groupdict()

def parse_hp_folder_name(name: str) -> Dict[str, Any]:
    """Parses 'adv_0.1_poison_0.5'"""
    match = HP_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse HP folder name: {name}")
        return {}
    data = match.groupdict()
    try:
        data['adv_rate'] = float(data['adv_rate'])
        data['poison_rate'] = float(data['poison_rate'])
        return data
    except ValueError:
        return {}

def find_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Finds all final_metrics.json files recursively and parses their context.
    Expected structure:
    <results_dir> / <scenario_name> / <hp_suffix> / <run_name> / final_metrics.json
    """
    logger.info(f"🔍 Scanning recursively for results in: {results_dir}...")

    # Use a glob pattern that matches the Step 5 structure
    metrics_files = list(results_dir.glob("step5_atk_sens_*/*/run_*/final_metrics.json"))
    if not metrics_files:
         metrics_files.extend(list(results_dir.glob("step5_atk_sens_*/*/*/final_metrics.json")))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {results_dir} matching Step 5 structure.")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")
    all_results = []

    for metrics_file in metrics_files:
        try:
            run_dir = metrics_file.parent
            hp_dir = run_dir.parent
            scenario_dir = hp_dir.parent

            if not (run_dir / ".success").exists():
                continue

            # 1. Parse context from folder names
            scenario_params = parse_scenario_name(scenario_dir.name)
            hp_params = parse_hp_folder_name(hp_dir.name)

            if not scenario_params or not hp_params:
                logger.warning(f"Skipping {metrics_file.parent}, failed to parse path.")
                continue

            # 2. Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # 3. Combine all data
            record = {
                **scenario_params,
                **hp_params,
                "seed_run": run_dir.name,
                "test_acc": metrics.get("test_acc") or metrics.get("acc"),
                "backdoor_asr": metrics.get("backdoor_asr") or metrics.get("adv_success_rate"),
            }
            all_results.append(record)
        except Exception as e:
            logger.warning(f"Warning: Could not process file {metrics_file}: {e}")
            continue

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)

def analyze_results(df: pd.DataFrame, results_dir: Path):
    """Aggregates results and saves summaries for plotting."""
    if df.empty:
        logger.warning("No successful results to analyze.")
        return

    # --- 1. Clean up Data ---
    numeric_cols = ["test_acc", "backdoor_asr", "adv_rate", "poison_rate"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['backdoor_asr'] = df['backdoor_asr'].fillna(0.0)
    df = df.dropna(subset=numeric_cols)

    # --- 2. Aggregate over Seeds ---
    group_cols = ["sweep_type", "defense", "attack_type", "modality", "dataset",
                  "adv_rate", "poison_rate"]

    agg_df = df.groupby(group_cols, as_index=False).agg(
        mean_acc=('test_acc', 'mean'),
        std_acc=('test_acc', 'std'),
        mean_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std'),
        n_runs=('test_acc', 'count')
    )

    # --- 3. Separate the two experiments ---

    # Experiment 1: Sweeping Adversary Rate
    adv_sweep_df = agg_df[agg_df['sweep_type'] == 'adv'].copy()
    adv_sweep_df = adv_sweep_df.sort_values(by=["defense", "attack_type", "adv_rate"])

    # Experiment 2: Sweeping Poison Rate
    poison_sweep_df = agg_df[agg_df['sweep_type'] == 'poison'].copy()
    poison_sweep_df = poison_sweep_df.sort_values(by=["defense", "attack_type", "poison_rate"])

    # --- 4. Display Summary Tables ---
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.4f}'.format)

    print("\n" + "="*100)
    print("📊 Attack Sensitivity: Adversary Rate Sweep (Fixed Poison Rate)")
    print("="*100)
    print(adv_sweep_df.to_string(index=False))

    print("\n" + "="*100)
    print("📊 Attack Sensitivity: Poison Rate Sweep (Fixed Adversary Rate)")
    print("="*100)
    print(poison_sweep_df.to_string(index=False))

    # --- 5. Save Full Results to CSV ---
    try:
        adv_csv = results_dir / "step5_sensitivity_adv_rate_summary.csv"
        adv_sweep_df.to_csv(adv_csv, index=False, float_format="%.4f")
        logger.info(f"\n✅ Adversary rate sweep summary saved to: {adv_csv}")

        poison_csv = results_dir / "step5_sensitivity_poison_rate_summary.csv"
        poison_sweep_df.to_csv(poison_csv, index=False, float_format="%.4f")
        logger.info(f"✅ Poison rate sweep summary saved to: {poison_csv}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save results to CSV: {e}")

    print("\n" + "="*100)
    print("Analysis complete. You can now use the saved CSV files to generate plots.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attack strength sensitivity (Step 5) results."
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
        raw_results_df = find_all_results(results_path)
        analyze_results(raw_results_df, results_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()