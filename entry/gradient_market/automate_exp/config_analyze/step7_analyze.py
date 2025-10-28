# FILE: analyze_step7_adaptive.py

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging
import warnings

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Parsers for the Step 7 Directory Structure ---

# Parses: step7_adaptive_black_box_gradient_manipulation_fltrust_cifar10
SCENARIO_REGEX = re.compile(
    r"step7_adaptive_(?P<threat_model>[\w-]+)_(?P<attack_mode>[\w-]+)_(?P<defense>[\w-]+)_(?P<dataset>[\w\d]+)"
)

# Parses: ds-cifar10_model-resnet18_..._seed-42
RUN_NAME_REGEX = re.compile(r".*_seed-(?P<seed>\d+)$")

def parse_scenario_name(name: str) -> Dict[str, Any]:
    """Parses the main scenario folder name."""
    match = SCENARIO_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse scenario name: {name}")
        return {}
    return match.groupdict()

def parse_run_name(name: str) -> Dict[str, Any]:
    """Parses the seed from the run folder name."""
    match = RUN_NAME_REGEX.search(name)
    if not match:
        # Fallback for "run_1_seed_42"
        if "seed" in name:
            try: return {"seed": int(name.split('_')[-1])}
            except Exception: pass
        logger.warning(f"Could not parse seed from run name: {name}")
        return {"seed": 0}
    return {"seed": int(match.group("seed"))}

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a .jsonl file line by line."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        logger.debug(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return []

def find_all_results(results_dir: Path) -> (pd.DataFrame, pd.DataFrame):
    """
    Finds and processes all results for Step 7.
    Returns two dataframes:
    1. final_results_df: One row per run, with final metrics.
    2. round_by_round_df: One row per round per run, with time-series metrics.
    """
    logger.info(f"🔍 Scanning recursively for Step 7 results in: {results_dir}...")

    # Find all final_metrics.json files
    scenario_dirs = list(results_dir.glob("step7_adaptive_*"))
    if not scenario_dirs:
        logger.error(f"❌ ERROR: No 'step7_adaptive_*' directories found in {results_dir}.")
        return pd.DataFrame(), pd.DataFrame()

    final_results = []
    round_by_round_results = []

    run_dirs_count = 0

    for scenario_dir in scenario_dirs:
        if not scenario_dir.is_dir():
            continue

        scenario_params = parse_scenario_name(scenario_dir.name)
        if not scenario_params:
            continue

        # Find all run directories (e.g., ds-cifar10...)
        for run_dir in scenario_dir.glob("ds-*_seed-*"):
            if not (run_dir / ".success").exists():
                continue

            run_dirs_count += 1
            run_params = parse_run_name(run_dir.name)
            base_info = {**scenario_params, **run_params}

            # 1. Load Final Metrics
            try:
                with open(run_dir / "final_metrics.json", 'r') as f:
                    final_metrics = json.load(f)

                # Add influence scores if they exist
                final_inf = load_jsonl(run_dir / "influence_scores.jsonl")
                if final_inf:
                    # Get the average score from the *last* valuation round
                    final_metrics["final_influence_score"] = final_inf[-1].get("avg_score", pd.NA)

                final_results.append({
                    **base_info,
                    "test_acc": final_metrics.get("test_acc") or final_metrics.get("acc"),
                    "backdoor_asr": final_metrics.get("backdoor_asr") or final_metrics.get("adv_success_rate"),
                    "final_influence": final_metrics.get("final_influence_score", pd.NA)
                })
            except FileNotFoundError:
                logger.warning(f"Missing final_metrics.json in {run_dir}")
                continue
            except Exception as e:
                logger.warning(f"Error loading final_metrics.json for {run_dir}: {e}")

            # 2. Load Round-by-Round Metrics
            round_metrics = load_jsonl(run_dir / "metrics_per_round.jsonl")
            if not round_metrics:
                logger.warning(f"Missing metrics_per_round.jsonl in {run_dir}")
                continue

            for round_data in round_metrics:
                round_by_round_results.append({
                    **base_info,
                    "round": round_data.get("round"),
                    "test_acc": round_data.get("test_acc"),
                    "backdoor_asr": round_data.get("backdoor_asr"),
                    "attacker_selection_rate": round_data.get("selection_rates", {}).get("adaptive_attacker", pd.NA)
                })

    logger.info(f"✅ Found and processed {len(final_results)} final results from {run_dirs_count} successful runs.")
    logger.info(f"✅ Found {len(round_by_round_results)} total round-by-round metric points.")

    return pd.DataFrame(final_results), pd.DataFrame(round_by_round_results)


def analyze_results(final_df: pd.DataFrame, round_df: pd.DataFrame, results_dir: Path):
    """Aggregates results and saves summaries for plotting."""

    if final_df.empty or round_df.empty:
        logger.warning("No successful results to analyze (empty dataframes).")
        return

    # --- 1. Analyze Final Metrics ---
    logger.info("Aggregating final results...")
    final_group_cols = ["defense", "attack_mode", "threat_model", "dataset"]
    final_agg = final_df.groupby(final_group_cols, as_index=False).agg(
        mean_acc=('test_acc', 'mean'),
        std_acc=('test_acc', 'std'),
        mean_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std'),
        mean_final_inf=('final_influence', 'mean'),
        n_runs=('test_acc', 'count')
    )
    final_agg = final_agg.sort_values(by=["attack_mode", "defense"])

    # --- 2. Analyze Round-by-Round Metrics ---
    logger.info("Aggregating round-by-round results...")
    round_group_cols = ["defense", "attack_mode", "threat_model", "dataset", "round"]

    # Clean and convert round_df
    numeric_cols = ["round", "test_acc", "backdoor_asr", "attacker_selection_rate"]
    for col in numeric_cols:
        round_df[col] = pd.to_numeric(round_df[col], errors='coerce')
    round_df = round_df.dropna(subset=["round", "test_acc"])
    round_df["backdoor_asr"] = round_df["backdoor_asr"].fillna(0.0)
    round_df["attacker_selection_rate"] = round_df["attacker_selection_rate"].fillna(0.0)

    round_agg = round_df.groupby(round_group_cols, as_index=False).agg(
        mean_acc=('test_acc', 'mean'),
        std_acc=('test_acc', 'std'),
        mean_asr=('backdoor_asr', 'mean'),
        std_asr=('backdoor_asr', 'std'),
        mean_selection_rate=('attacker_selection_rate', 'mean'),
        std_selection_rate=('attacker_selection_rate', 'std')
    )
    round_agg = round_agg.sort_values(by=["attack_mode", "defense", "round"])

    # --- 3. Display Summary Tables ---
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.4f}'.format)

    print("\n" + "="*120)
    print("📊 Adaptive Attack Analysis: FINAL Performance")
    print("   (Shows the end-game state after attacker has finished learning)")
    print("="*120)
    print(final_agg.to_string(index=False, na_rep="N/A"))

    print("\n" + "="*120)
    print("📊 Adaptive Attack Analysis: ROUND-BY-ROUND Performance (Head)")
    print("   (Shows the learning process. Full data saved to CSV for plotting)")
    print("="*120)
    print(round_agg.head(20).to_string(index=False, na_rep="N/A"))

    # --- 4. Save Full Results to CSV ---
    try:
        final_csv = results_dir / "step7_summary_final.csv"
        final_agg.to_csv(final_csv, index=False, float_format="%.4f")
        logger.info(f"\n✅ Final summary saved to: {final_csv}")

        round_csv = results_dir / "step7_summary_round_by_round.csv"
        round_agg.to_csv(round_csv, index=False, float_format="%.4f")
        logger.info(f"✅ Round-by-round data (for plotting) saved to: {round_csv}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save results to CSV: {e}")

    print("\n" + "="*120)
    print("Analysis complete. Use the '_round_by_round.csv' file to plot learning curves.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze adaptive attack (Step 7) results."
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
        final_df, round_df = find_all_results(results_path)
        analyze_results(final_df, round_df, results_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()