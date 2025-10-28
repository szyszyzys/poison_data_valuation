# FILE: analyze_step6_sybil.py

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from entry.gradient_market.automate_exp.configs_generation.config_common_utils import DEFAULT_ADV_RATE, \
    DEFAULT_POISON_RATE

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parsers for the Step 6 Directory Structure ---

# Parses: step6_adv_sybil_oracle_blend_fltrust_cifar10
SCENARIO_REGEX = re.compile(
    r"step6_adv_sybil_(?P<strategy>[\w-]+)_(?P<defense>[\w-]+)_(?P<dataset>[\w\d]+)"
)

# Parses: adv_0.3_poison_0.5_blend_alpha_0.1 (and handles missing alpha)
HP_REGEX = re.compile(
    r"adv_(?P<adv_rate>[\d\.]+)_poison_(?P<poison_rate>[\d\.]+)(_blend_alpha_(?P<blend_alpha>[\d\.]+))?"
)


def parse_scenario_name(name: str) -> Dict[str, Any]:
    """Parses the main scenario folder name."""
    match = SCENARIO_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse scenario name: {name}")
        return {}
    return match.groupdict()


def parse_hp_folder_name(name: str) -> Dict[str, Any]:
    """Parses 'adv_0.3_poison_0.5_blend_alpha_0.1'"""
    match = HP_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse HP folder name: {name}")
        return {}
    data = match.groupdict()
    try:
        data['adv_rate'] = float(data['adv_rate'])
        data['poison_rate'] = float(data['poison_rate'])
        # Handle optional blend_alpha
        if data['blend_alpha']:
            data['blend_alpha'] = float(data['blend_alpha'])
        else:
            data['blend_alpha'] = pd.NA
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

    metrics_files = list(results_dir.glob("step6_adv_sybil_*/*/run_*/final_metrics.json"))
    if not metrics_files:
        metrics_files.extend(list(results_dir.glob("step6_adv_sybil_*/*/*/final_metrics.json")))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {results_dir} matching Step 6 structure.")
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
                # Add valuation metrics if they exist
                "inf_avg_score": metrics.get("influence_scores", {}).get("avg_score", pd.NA),
                "loo_avg_score": metrics.get("loo_scores", {}).get("avg_score", pd.NA),
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
    numeric_cols = ["test_acc", "backdoor_asr", "adv_rate", "poison_rate",
                    "blend_alpha", "inf_avg_score", "loo_avg_score"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['backdoor_asr'] = df['backdoor_asr'].fillna(0.0)
    df = df.dropna(subset=["test_acc", "defense", "strategy"])

    # --- 2. Aggregate over Seeds ---
    group_cols = ["strategy", "defense", "dataset", "adv_rate", "poison_rate", "blend_alpha"]

    agg_df = df.groupby(group_cols, as_index=False, dropna=False).agg(
        mean_acc=('test_acc', 'mean'),
        std_acc=('test_acc', 'std'),
        mean_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std'),
        mean_inf_val=('inf_avg_score', 'mean'),
        mean_loo_val=('loo_avg_score', 'mean'),
        n_runs=('test_acc', 'count')
    )

    # --- 3. Separate the two analyses ---

    # Analysis 1: Direct Strategy Comparison (non-swept params)
    summary_df = agg_df[agg_df['strategy'] != 'oracle_blend'].copy()
    summary_df = summary_df.sort_values(by=["defense", "strategy"])

    # Analysis 2: Oracle Blend Alpha Sweep
    oracle_sweep_df = agg_df[agg_df['strategy'] == 'oracle_blend'].copy()
    oracle_sweep_df = oracle_sweep_df.sort_values(by=["defense", "blend_alpha"])

    # --- 4. Display Summary Tables ---
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.4f}'.format)

    print("\n" + "=" * 120)
    print("📊 Sybil Strategy Comparison (Fixed Attack Strength)")
    print(f"   (adv_rate={DEFAULT_ADV_RATE}, poison_rate={DEFAULT_POISON_RATE})")
    print("=" * 120)
    display_cols_summary = ["defense", "strategy", "mean_acc", "std_acc", "mean_asr", "std_asr", "mean_inf_val",
                            "mean_loo_val", "n_runs"]
    print(summary_df.reindex(columns=display_cols_summary).to_string(index=False, na_rep="N/A"))

    print("\n" + "=" * 120)
    print("📊 Sybil Strategy: Oracle Blend (Alpha Sweep)")
    print(f"   (adv_rate={DEFAULT_ADV_RATE}, poison_rate={DEFAULT_POISON_RATE})")
    print("=" * 120)
    display_cols_sweep = ["defense", "blend_alpha", "mean_acc", "std_acc", "mean_asr", "std_asr", "mean_inf_val",
                          "mean_loo_val", "n_runs"]
    print(oracle_sweep_df.reindex(columns=display_cols_sweep).to_string(index=False, na_rep="N/A"))

    # --- 5. Save Full Results to CSV ---
    try:
        summary_csv = results_dir / "step6_sybil_strategy_summary.csv"
        summary_df.to_csv(summary_csv, index=False, float_format="%.4f")
        logger.info(f"\n✅ Strategy comparison summary saved to: {summary_csv}")

        sweep_csv = results_dir / "step6_sybil_oracle_sweep_summary.csv"
        oracle_sweep_df.to_csv(sweep_csv, index=False, float_format="%.4f")
        logger.info(f"✅ Oracle sweep summary saved to: {sweep_csv}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save results to CSV: {e}")

    print("Complete" + "=" * 120)
    print("Analysis complete. You can now use the saved CSV files to generate plots.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Sybil attack strategy (Step 6) results."
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
