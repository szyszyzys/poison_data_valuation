import argparse
import re
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This regex will parse the sub-folder name, e.g.:
# "ds-cifar10_model-flexiblecnn_agg-fedavg_image_backdoor_adv-0p3_poison-1p0_seed-42"
# We only care about 'ds' and 'model'
HP_FOLDER_REGEX = re.compile(r"ds-([a-zA-Z0-9]+)_model-([a-zA-Z0-9_]+)")

# The experiment name pattern from your step 1 generator
EXPERIMENT_PATTERN = "step1_tune_fedavg_*"

def find_and_analyze_norms(results_dir: Path) -> pd.DataFrame:
    """Finds all seller_metrics.csv files from Step 1 and analyzes their benign norms."""

    # --- THIS IS THE FIX ---
    # The pattern for .glob() must be *relative* to the results_dir path.
    # We build the relative path pattern directly.
    search_pattern = f"{EXPERIMENT_PATTERN}/*"

    # Check for the two possible directory structures
    path_structure_1 = list(results_dir.glob(f"{search_pattern}/run_*/seller_metrics.csv"))
    path_structure_2 = list(results_dir.glob(f"{search_pattern}/*/run_*/seller_metrics.csv"))

    metrics_files = path_structure_1 + path_structure_2
    # --- END FIX ---

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'seller_metrics.csv' files found in {results_dir} matching the Step 1 structure.")
        logger.error(f"   Searched pattern: {results_dir / search_pattern}/.../seller_metrics.csv")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} seller_metrics.csv files to analyze.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            # 2. Parse context from folder names
            run_dir = metrics_file.parent
            hp_dir = run_dir.parent

            # Check for success
            if not (run_dir / ".success").exists():
                logger.debug(f"Skipping failed/incomplete run: {run_dir.name}")
                continue

            # Parse dataset and model from the hp_dir name
            match = HP_FOLDER_REGEX.search(hp_dir.name)
            if not match:
                # Try the parent's parent directory
                hp_dir = hp_dir.parent
                match = HP_FOLDER_REGEX.search(hp_dir.name)
                if not match:
                    logger.warning(f"Could not parse ds/model from: {hp_dir.name}. Skipping.")
                    continue

            dataset, model = match.groups()

            # 3. Read the CSV and filter for benign sellers
            df = pd.read_csv(metrics_file)

            if 'gradient_norm' not in df.columns or 'seller_id' not in df.columns:
                logger.warning(f"Skipping {metrics_file}: missing required columns.")
                continue

            # Filter for benign sellers only
            benign_df = df[df['seller_id'].str.startswith('bn_')].copy()

            if benign_df.empty:
                logger.warning(f"No benign sellers found in: {metrics_file}. Skipping.")
                continue

            # 4. Calculate the average norm for this run
            # Convert to numeric and drop NaNs (e.g., from round 0)
            benign_df['gradient_norm'] = pd.to_numeric(benign_df['gradient_norm'], errors='coerce')
            benign_df = benign_df.dropna(subset=['gradient_norm'])

            if benign_df.empty:
                logger.warning(f"No valid gradient norms found in: {metrics_file}. Skipping.")
                continue

            avg_norm = benign_df['gradient_norm'].mean()

            all_results.append({
                "dataset": dataset,
                "model": model,
                "avg_benign_norm": avg_norm,
                "run_dir": run_dir.name
            })
        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}")
            continue

    if not all_results:
        logger.error("❌ No valid benign norm data could be processed.")
        return pd.DataFrame()

    # 5. Aggregate across seeds
    results_df = pd.DataFrame(all_results)
    agg_df = results_df.groupby(['dataset', 'model'])['avg_benign_norm'].agg(
        mean_norm='mean',
        std_norm='std',
        run_count='count'
    )

    return agg_df.sort_values(by='mean_norm', ascending=False)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze benign gradient norms from Step 1 (FedAvg) runs."
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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    try:
        norm_results_df = find_and_analyze_norms(results_path)

        if not norm_results_df.empty:
            print("\n" + "="*80)
            print("📈 Average Benign L2-Norm of Gradients (from Step 1)")
            print("="*80)
            print(norm_results_df.to_string(float_format="%.2f"))
            print("\n" + "="*80)
            print("ACTION: Use these 'mean_norm' values to create your new, specialized")
            print("        `TUNING_GRIDS` in `generate_step3_defense_tuning.py`.")
            print("        (Or, better yet, implement relative `clip_factor`!)")
            print("="*80)

            # Save the file
            output_csv = results_path / "step1_benign_norm_analysis.csv"
            norm_results_df.to_csv(output_csv, float_format="%.4f")
            logger.info(f"\n✅ Norm analysis summary saved to: {output_csv}")

    except FileNotFoundError:
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()