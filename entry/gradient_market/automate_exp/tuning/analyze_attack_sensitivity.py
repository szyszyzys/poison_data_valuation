import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regex to parse the experiment sub-folder name, e.g., "adv_rate_0.3_poison_rate_1.0"
ATTACK_PARAM_REGEX = re.compile(
    r"adv_rate_(?P<adv_rate>[\d\.]+)_poison_rate_(?P<poison_rate>[\d\.]+)"
)

# ==============================================================================

def parse_scenario_name_attack_sens(name: str) -> Dict[str, str]:
    """
    Parses 'step5_atk_sens_fltrust_cifar10'
    """
    try:
        parts = name.split('_')
        if len(parts) < 5:
            logger.warning(f"Could not parse scenario name: {name}")
            return {}

        return {
            "defense": parts[3],
            "dataset": parts[4],
            "scenario": name,
        }
    except Exception as e:
        logger.warning(f"Error parsing scenario '{name}': {e}")
        return {}


def parse_attack_param_name(name: str) -> Dict[str, Any]:
    """
    Parses 'adv_rate_0.3_poison_rate_1.0'
    """
    match = ATTACK_PARAM_REGEX.match(name)
    if not match:
        logger.warning(f"Could not parse attack param name: {name}")
        return {}
    data = match.groupdict()
    try:
        data['adv_rate'] = float(data['adv_rate'])
        data['poison_rate'] = float(data['poison_rate'])
        return data
    except ValueError:
        return {}


def find_all_attack_sensitivity_results(root_dir: Path) -> pd.DataFrame:
    """Finds all final_metrics.json files and parses their full context."""
    logger.info(f"🔍 Scanning for all attack sensitivity results in: {root_dir}...")

    metrics_files = list(root_dir.rglob("final_metrics.json"))
    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {root_dir}.")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            # Path: .../<scenario_name>/<attack_param_name>/<seed_name>/final_metrics.json
            seed_dir = metrics_file.parent
            attack_param_dir = seed_dir.parent
            scenario_dir = attack_param_dir.parent

            # 1. Check for success
            if not (seed_dir / ".success").exists():
                continue

            # 2. Parse context from folder names
            scenario_info = parse_scenario_name_attack_sens(scenario_dir.name)
            attack_param_info = parse_attack_param_name(attack_param_dir.name)

            if not scenario_info or not attack_param_info:
                # Log the problematic paths if parsing fails
                logger.debug(f"Skipping due to parsing error:")
                logger.debug(f"  Scenario dir: {scenario_dir}")
                logger.debug(f"  Attack param dir: {attack_param_dir}")
                continue

            # 3. Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # 4. Store the combined record
            record = {
                **scenario_info,
                **attack_param_info,
                "seed_run": seed_dir.name,
                "test_acc": metrics.get("test_acc"),
                "backdoor_asr": metrics.get("backdoor_asr"), # Make sure your metrics file includes this
            }
            all_results.append(record)
        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}")

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def plot_attack_sensitivity(agg_df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves line plots for test accuracy and ASR vs. adv_rate.
    """
    if agg_df.empty:
        logger.warning("No aggregated data to plot.")
        return

    datasets = agg_df['dataset'].unique()
    logger.info(f"Plotting results for datasets: {list(datasets)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for dataset in datasets:
        logger.info(f"  Generating plots for dataset: {dataset}")
        dataset_df = agg_df[agg_df['dataset'] == dataset].copy()

        # Determine the x-axis (the parameter you varied)
        # If adv_rate has multiple values, use it. Otherwise, use poison_rate.
        if dataset_df['adv_rate'].nunique() > 1:
            x_var = 'adv_rate'
            x_label = 'Adversary Rate (adv_rate)'
        elif dataset_df['poison_rate'].nunique() > 1:
            x_var = 'poison_rate'
            x_label = 'Poison Rate (within adversary data)'
        else:
            logger.warning(f"Skipping plot for {dataset}: Neither adv_rate nor poison_rate varies.")
            continue

        # --- Plot 1: Test Accuracy vs. Attack Strength ---
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=dataset_df,
            x=x_var,
            y='mean_test_acc',
            hue='defense',
            style='defense',
            markers=True,
            dashes=False
        )
        plt.title(f'Test Accuracy vs. Attack Strength ({dataset})')
        plt.xlabel(x_label)
        plt.ylabel('Mean Test Accuracy')
        plt.ylim(0, 1.05) # Set y-axis limits from 0 to 1
        plt.legend(title='Defense')
        plt.grid(True)
        # Add horizontal line for 50% accuracy
        plt.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
        # Add horizontal line for random guess (adjust if needed, e.g., 0.1 for CIFAR10)
        random_guess = 0.1 if 'cifar' in dataset else (0.5 if 'texas' in dataset or 'purchase' in dataset else 0.2) # Basic guess
        plt.axhline(random_guess, color='red', linestyle=':', linewidth=0.8, label=f'Random Guess ({random_guess:.1f})')
        plt.legend(title='Defense')


        acc_plot_path = output_dir / f'sensitivity_{dataset}_accuracy.png'
        plt.savefig(acc_plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"    Saved accuracy plot: {acc_plot_path}")
        #


        # --- Plot 2: Attack Success Rate vs. Attack Strength ---
        # Only plot if ASR data exists
        if 'mean_backdoor_asr' in dataset_df.columns and not dataset_df['mean_backdoor_asr'].isnull().all():
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=dataset_df,
                x=x_var,
                y='mean_backdoor_asr',
                hue='defense',
                style='defense',
                markers=True,
                dashes=False
            )
            plt.title(f'Attack Success Rate (ASR) vs. Attack Strength ({dataset})')
            plt.xlabel(x_label)
            plt.ylabel('Mean Backdoor ASR')
            plt.ylim(0, 1.05) # Set y-axis limits from 0 to 1
            plt.legend(title='Defense')
            plt.grid(True)
             # Add horizontal line for 10% ASR (often considered a threshold for defense effectiveness)
            plt.axhline(0.1, color='green', linestyle='--', linewidth=0.8, label='10% ASR Threshold')
            plt.legend(title='Defense')


            asr_plot_path = output_dir / f'sensitivity_{dataset}_asr.png'
            plt.savefig(asr_plot_path, bbox_inches='tight')
            plt.close()
            logger.info(f"    Saved ASR plot: {asr_plot_path}")
            #
        else:
             logger.warning(f"    Skipping ASR plot for {dataset}: 'mean_backdoor_asr' column not found or is all NaN.")

    logger.info("Plot generation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL Attack Sensitivity results (Step 5) and generate plots."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="The root results directory for the Step 5 run (e.g., './results/')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_plots/step5_attack_sensitivity",
        help="Directory to save the generated plots."
    )
    args = parser.parse_args()
    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)

    if not results_path.exists():
        logger.error(f"❌ ERROR: The results directory '{results_path}' does not exist.")
        return

    try:
        # 1. Find and parse all results
        raw_results_df = find_all_attack_sensitivity_results(results_path)

        if raw_results_df.empty:
            logger.warning("No valid results found. Exiting.")
            return

        # 2. Aggregate across seeds
        group_cols = ["scenario", "defense", "dataset", "adv_rate", "poison_rate"]
        agg_df = raw_results_df.groupby(group_cols).agg(
            mean_test_acc=('test_acc', 'mean'),
            std_test_acc=('test_acc', 'std'),
            mean_backdoor_asr=('backdoor_asr', 'mean'),
            std_backdoor_asr=('backdoor_asr', 'std'),
            run_count=('seed_run', 'count')
        ).reset_index()

        # Fill NaNs in std dev if only one run exists
        agg_df['std_test_acc'] = agg_df['std_test_acc'].fillna(0)
        agg_df['std_backdoor_asr'] = agg_df['std_backdoor_asr'].fillna(0)

        # 3. Generate and save plots
        plot_attack_sensitivity(agg_df, output_path)

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)


if __name__ == "__main__":
    main()