# FILE: analyze_step3_defense_tuning.py

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

# ## USER ACTION ##: Define how to score/rank defense performance.
RANKING_METRIC = 'acc_minus_asr'  # Choose one
ASR_THRESHOLD = 0.1  # Used if filtering first (Option 1) - not used by default score


# --- Parsing Functions ---

def parse_scenario_name_step3(name: str) -> Dict[str, str]:
    """ Parses 'step3_tune_fltrust_backdoor_image_cifar10_resnet18' """
    try:
        parts = name.split('_')
        # step3_tune_{defense}_{attack_type}_{modality}_{dataset}_{model}
        if len(parts) < 7:
            logger.warning(f"Could not parse scenario name: {name}")
            return {}
        return {
            "defense": parts[2],
            "attack_type": parts[3],
            "modality": parts[4],
            "dataset": parts[5],
            "model_suffix": "_".join(parts[6:]),  # Handle models with underscores
            "scenario": name,
        }
    except Exception as e:
        logger.warning(f"Error parsing scenario name '{name}': {e}")
        return {}


def parse_defense_hp_folder_name(name: str) -> Dict[str, Any]:
    """
    Parses 'aggregation.martfl.max_k_5_aggregation.clip_norm_10.0'
    Handles multiple key_value pairs separated by '_'.
    Converts numeric values to float/int.
    """
    params = {}
    raw_params = {}  # Store raw string values for grouping
    try:
        parts = name.split('_')
        i = 0
        while i < len(parts):
            # Find the full key (can contain dots)
            key_parts = []
            while i < len(parts) and not parts[i].replace('.', '', 1).replace('-', '', 1).isdigit():
                # Allow negative numbers starting with '-'
                key_parts.append(parts[i])
                i += 1
            param_key = "_".join(key_parts)
            # Use short key (e.g., 'max_k') for column name
            param_key_short = param_key.split('.')[-1]

            if not param_key_short:  # Skip if key is empty
                continue

            # Get the value
            if i < len(parts):
                raw_value = parts[i]
                i += 1
                # Attempt to convert to numeric
                try:
                    if '.' in raw_value:
                        value = float(raw_value)
                    # Handle None/null string representation if needed
                    elif raw_value.lower() == 'none' or raw_value.lower() == 'null':
                        value = None
                        raw_value = 'None'  # Consistent string rep
                    else:
                        value = int(raw_value)
                    params[param_key_short] = value
                    raw_params[param_key_short] = raw_value  # Store original string
                except ValueError:
                    # Keep as string if conversion fails
                    params[param_key_short] = raw_value
                    raw_params[param_key_short] = raw_value
            else:
                # logger.warning(f"No value found after key '{param_key}' in '{name}'")
                pass

        # Add raw params prefixed with 'raw_' for reliable grouping if needed
        for k, v in raw_params.items():
            params[f'raw_{k}'] = v

        return params
    except Exception as e:
        logger.warning(f"Error parsing HP folder name '{name}': {e}")
        return {}


def find_all_tuning_results(root_dir: Path) -> pd.DataFrame:
    """Finds all final_metrics.json files and parses context including defense HPs."""
    logger.info(f"🔍 Scanning for Step 3 results in: {root_dir}...")

    # Use rglob to recursively find all final_metrics.json
    metrics_files = list(root_dir.rglob("**/final_metrics.json"))
    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found recursively in {root_dir}.")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            # --- Robust Path Finding ---
            # 1. Find the 'step3_tune_' parent (scenario_dir)
            current_path = metrics_file.parent
            scenario_dir = None
            while current_path != root_dir and current_path != current_path.parent:
                if current_path.name.startswith("step3_tune_"):
                    scenario_dir = current_path
                    break
                current_path = current_path.parent

            if scenario_dir is None:
                continue

                # 2. Get relative path parts
            relative_path_parts = metrics_file.parent.relative_to(scenario_dir).parts

            if len(relative_path_parts) < 1:
                logger.warning(f"Unexpected path structure for {metrics_file}. Skipping.")
                continue

            # 3. Assume HP folder is the first, seed folder is the last
            hp_dir_name = relative_path_parts[0]
            seed_dir_name = relative_path_parts[-1]  # This is the leaf folder
            seed_dir = metrics_file.parent

            if not (seed_dir / ".success").exists():
                continue  # Skip failed runs

            # 4. Parse info
            scenario_info = parse_scenario_name_step3(scenario_dir.name)
            hp_info = parse_defense_hp_folder_name(hp_dir_name)

            if not scenario_info or ("defense" not in scenario_info):
                continue  # Must have valid scenario info

            # ## NEW FILTER from previous request##
            # Only include results where the model_suffix ends with '_new'
            if not scenario_info.get("model_suffix", "").endswith("_new"):
                continue
            # ## END NEW FILTER ##

            try:
                # Try parsing seed from folder name, e.g., 'run_0_seed_42' -> 42
                seed = int(seed_dir_name.split('_')[-1])
            except:
                seed = -1  # Fallback

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # --- **MODIFIED METRIC READING** ---
            test_acc = metrics.get("test_acc", metrics.get("acc"))
            backdoor_asr = metrics.get("backdoor_asr", metrics.get("test_asr", metrics.get("asr")))
            # --- **END MODIFICATION** ---

            # Store combined record
            record = {
                **scenario_info,
                **hp_info,  # Parsed defense HPs (e.g., clip_norm: 10.0)
                "hp_folder": hp_dir_name,  # Keep original HP folder name
                "seed": seed,
                "status": "success",
                "test_acc": test_acc,
                "backdoor_asr": backdoor_asr,
            }

            if record["backdoor_asr"] is None:
                record["backdoor_asr"] = 0.0

            all_results.append(record)
        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}", exc_info=False)

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results: return pd.DataFrame()

    return pd.DataFrame(all_results)


def analyze_defense_tuning(raw_df: pd.DataFrame, results_dir: Path):
    """Aggregates results and finds the best defense HPs."""
    if raw_df.empty:
        logger.warning("No data to analyze.")
        return

    # Identify defense HP columns (those parsed from folder name, excluding 'raw_')
    hp_cols = [col for col in raw_df.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'hp_folder', 'seed', 'status', 'test_acc', 'backdoor_asr'
    ] and not col.startswith('raw_')]
    logger.info(f"Identified Defense HP columns: {hp_cols}")

    # Define grouping columns (base scenario + specific HPs)
    group_cols = ['scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix'] + hp_cols
    group_cols = [col for col in group_cols if col in raw_df.columns]

    if not group_cols:
        logger.warning("No grouping columns found. Analyzing raw data.")
        group_cols = ['scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix']

    # --- 1. Aggregate across seeds ---
    agg_df = raw_df.groupby(group_cols, dropna=False).agg(
        mean_test_acc=('test_acc', 'mean'),
        std_test_acc=('test_acc', 'std'),
        mean_backdoor_asr=('backdoor_asr', 'mean'),
        std_backdoor_asr=('backdoor_asr', 'std'),
        num_success_runs=('seed', 'count')
    ).reset_index()

    # --- **MODIFIED FAILSAFE** ---
    agg_df['std_test_acc'] = agg_df['std_test_acc'].fillna(0)
    agg_df['std_backdoor_asr'] = agg_df['std_backdoor_asr'].fillna(0)
    agg_df['mean_backdoor_asr'] = agg_df['mean_backdoor_asr'].fillna(0)
    agg_df['mean_test_acc'] = agg_df['mean_test_acc'].fillna(0)
    # --- **END MODIFICATION** ---

    # --- 2. Calculate Ranking Score ---
    if RANKING_METRIC == 'acc_minus_asr':
        agg_df['score'] = agg_df['mean_test_acc'] - agg_df['mean_backdoor_asr']
    elif RANKING_METRIC == 'acc_scaled_by_asr':
        agg_df['score'] = agg_df['mean_test_acc'] * (1.0 - agg_df['mean_backdoor_asr'])
    else:  # Default to accuracy only
        logger.warning(f"Unknown RANKING_METRIC '{RANKING_METRIC}', using accuracy only.")
        agg_df['score'] = agg_df['mean_test_acc']

    # --- 3. Find Best HPs for Each Scenario Group ---
    scenario_group_cols = ['defense', 'attack_type', 'modality', 'dataset', 'model_suffix']

    try:
        best_idx = agg_df.loc[agg_df.groupby(scenario_group_cols)['score'].idxmax()]
    except ValueError as ve:
        logger.error(f"Error finding best HPs, possibly due to empty groups: {ve}")
        return

    best_df = best_idx.sort_values(by=['modality', 'dataset', 'model_suffix', 'attack_type', 'defense'])

    # --- 4. Display Final Table of Best HPs ---
    display_cols = scenario_group_cols + hp_cols + [
        'mean_test_acc', 'std_test_acc', 'mean_backdoor_asr', 'std_backdoor_asr', 'score', 'num_success_runs'
    ]
    display_cols = [col for col in display_cols if col in best_df.columns]

    print("\n" + "=" * 120)
    print(f"🏆 Best Defense Hyperparameters Found (Ranked by: '{RANKING_METRIC}')")
    print("(This table shows the single best HP combination for each scenario found in the data)")
    print("=" * 120)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.float_format', '{:,.4f}'.format):
        print(best_df[display_cols].to_string(index=False))

    # --- 5. Save Full Aggregated Results & Best Results ---
    output_csv_all = results_dir / "step3_defense_tuning_all_aggregated.csv"
    output_csv_best = results_dir / "step3_defense_tuning_best_hps.csv"
    try:
        # Save the full aggregated results (not just the best)
        agg_df.sort_values(by=scenario_group_cols + ['score'],
                           ascending=[True] * len(scenario_group_cols) + [False]).to_csv(output_csv_all, index=False,
                                                                                         float_format="%.5f")
        logger.info(f"\n✅ Full aggregated tuning results saved to: {output_csv_all}")

        # Save just the best HPs
        best_df[display_cols].to_csv(output_csv_best, index=False, float_format="%.5f")
        logger.info(f"✅ Best hyperparameters saved to: {output_csv_best}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save aggregated results to CSV: {e}")

    print("\n" + "=" * 120)
    print(
        "Analysis complete. Use the 'Best Hyperparameters' table to update TUNED_DEFENSE_PARAMS in config_common_utils.py")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL Defense Tuning results (Step 3) to find best HPs."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="The ROOT results directory containing the Step 3 run folders (e.g., './results/')"
    )
    # --- NEW OPTIONAL ARGUMENTS ---
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional: Filter results for a specific dataset (e.g., 'CIFAR10')."
    )
    parser.add_argument(
        "--defense",
        type=str,
        default=None,
        help="Optional: Filter results for a specific defense (e.g., 'fltrust')."
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default=None,
        help="Optional: Filter results for a specific attack type (e.g., 'backdoor')."
    )
    # --- END NEW ARGUMENTS ---

    args = parser.parse_args()

    results_path = Path(args.results_dir).resolve()

    if not results_path.exists():
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
        logger.error("Please provide a valid path to the results directory.")
        return

    try:
        # --- 1. Run the full analysis and save CSVs ---
        raw_results_df = find_all_tuning_results(results_path)
        analyze_defense_tuning(raw_results_df, results_path)

        # --- 2. If filters are provided, show detailed results for that slice ---
        filters_applied = bool(args.dataset or args.defense or args.attack_type)

        if filters_applied:
            logger.info("--- Detailed Analysis for Specific Filters ---")

            # Read from the CSV we just created
            agg_csv_path = results_path / "step3_defense_tuning_all_aggregated.csv"
            if not agg_csv_path.exists():
                logger.error(f"Could not find aggregated file: {agg_csv_path}")
                return

            agg_df = pd.read_csv(agg_csv_path)

            # Apply filters
            if args.dataset:
                agg_df = agg_df[agg_df['dataset'].str.lower() == args.dataset.lower()]
            if args.defense:
                agg_df = agg_df[agg_df['defense'].str.lower() == args.defense.lower()]
            if args.attack_type:
                # Use user's 'attack_type' argument
                agg_df = agg_df[agg_df['attack_type'].str.lower() == args.attack_type.lower()]

            if agg_df.empty:
                logger.warning("No aggregated data matches the provided filters.")
            else:
                # --- Display logic from inspect_defense.py ---
                df_to_print = agg_df.sort_values(by='score', ascending=False)
                # Drop columns that are all NaN *for this slice*
                df_to_print = df_to_print.dropna(axis=1, how='all')

                # Dynamically find HP columns
                known_cols = [
                    'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
                    'mean_test_acc', 'std_test_acc', 'mean_backdoor_asr', 'std_backdoor_asr',
                    'num_success_runs', 'score'
                ]
                hp_cols = [col for col in df_to_print.columns if col not in known_cols and not col.startswith('raw_')]
                display_cols = hp_cols + ['mean_test_acc', 'std_test_acc', 'mean_backdoor_asr', 'std_backdoor_asr',
                                          'score', 'num_success_runs']
                display_cols = [col for col in display_cols if col in df_to_print.columns]

                print("\n" + "=" * 120)
                print(
                    f"🔍 Detailed Results for: Dataset='{args.dataset or 'Any'}', Defense='{args.defense or 'Any'}', Attack='{args.attack_type or 'Any'}'")
                print(f"(Ranked by: '{RANKING_METRIC}')")
                print("=" * 120)
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000,
                                       'display.float_format', '{:,.4f}'.format):
                    print(df_to_print[display_cols].to_string(index=False))
                print("=" * 120)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()