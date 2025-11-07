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
        'mean_test_acc', 'std_test_acc', 'mean_backdoor_asr', 'std_backdoor_asr',