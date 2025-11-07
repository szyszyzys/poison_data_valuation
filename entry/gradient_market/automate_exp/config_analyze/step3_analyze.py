import argparse
import pandas as pd
import logging
from pathlib import Path

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# We assume the script is run from a directory that can see the CSV
DATA_FILE = "step3_defense_tuning_all_aggregated.csv"

# Based on our analysis, we hardcode these filters
MODEL_SUFFIX_FILTER = '_new'
ATTACK_TYPE_FILTER = 'backdoor'


def analyze_specific_defense(dataset: str, defense: str):
    """
    Loads the aggregated data and shows results for a specific
    dataset and defense, focusing on backdoor attacks on _new models.
    """

    # --- 1. Load Data ---
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        logger.error(f"❌ ERROR: Cannot find the data file: {DATA_FILE}")
        logger.error("Please make sure this script is in the same directory as the CSV.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return

    # --- 2. Apply Filters ---
    logger.info(f"Loading data... Found {len(df)} total aggregated runs.")

    # Apply our standard filters
    df_filtered = df[
        (df['model_suffix'].str.endswith(MODEL_SUFFIX_FILTER)) &
        (df['attack_type'] == ATTACK_TYPE_FILTER)
        ]
    logger.info(
        f"Filtered for '{ATTACK_TYPE_FILTER}' attacks on '{MODEL_SUFFIX_FILTER}' models. {len(df_filtered)} runs remaining.")

    # Apply user-specific filters
    df_specific = df_filtered[
        (df_filtered['dataset'].str.lower() == dataset.lower()) &
        (df_filtered['defense'].str.lower() == defense.lower())
        ]

    if df_specific.empty:
        logger.warning(f"⚠️ No results found for Defense: '{defense}' on Dataset: '{dataset}'")
        logger.warning("Please check your spelling or try a different combination.")
        return

    logger.info(f"Found {len(df_specific)} parameter combinations for '{defense}' on '{dataset}'.")

    # --- 3. Find Relevant HP Columns ---
    # Drop any columns that are *entirely* NaN for this specific subset
    # This automatically hides HPs for other defenses (e.g., max_k for fltrust)
    df_clean = df_specific.dropna(axis=1, how='all')

    # Identify what columns are the HPs
    known_cols = [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'mean_test_acc', 'std_test_acc', 'mean_backdoor_asr', 'std_backdoor_asr',
        'num_success_runs', 'score'
    ]

    hp_cols = [col for col in df_clean.columns if col not in known_cols and not col.startswith('raw_')]

    if not hp_cols:
        logger.warning("No unique hyperparameters were varied for this defense (e.g., FedAvg).")

    # --- 4. Display Results ---
    display_cols = hp_cols + ['mean_test_acc', 'mean_backdoor_asr', 'score', 'num_success_runs']

    # Ensure all selected columns exist (just in case)
    display_cols = [col for col in display_cols if col in df_clean.columns]

    df_final = df_clean[display_cols].sort_values(by='score', ascending=False)

    print("\n" + "=" * 120)
    print(f"🏆 Analysis for Defense: '{defense}' on Dataset: '{dataset}'")
    print(f"(Filtered for '{ATTACK_TYPE_FILTER}' attacks on '{MODEL_SUFFIX_FILTER}' models, ranked by 'score')")
    print("=" * 120)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.float_format', '{:,.4f}'.format):
        print(df_final.to_string(index=False))
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze defense HPs for a specific dataset and defense."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to analyze (e.g., 'CIFAR10', 'TREC')."
    )
    parser.add_argument(
        "--defense",
        type=str,
        required=True,
        help="The defense to analyze (e.g., 'fltrust', 'skymask')."
    )
    args = parser.parse_args()

    analyze_specific_defense(args.dataset, args.defense)


if __name__ == "__main__":
    main()