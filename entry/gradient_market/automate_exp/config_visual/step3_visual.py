import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Parsing Functions (Copied from your analyze_step3 script) ---

def parse_scenario_name_step3(name: str) -> Dict[str, str]:
    """ Parses 'step3_tune_fltrust_backdoor_image_cifar10_cifar10_cnn_new' """
    try:
        parts = name.split('_')
        if len(parts) < 7:
            logger.warning(f"Could not parse scenario name: {name}")
            return {}
        return {
            "defense": parts[2],
            "attack_type": parts[3],
            "modality": parts[4],
            "dataset": parts[5],
            "model_suffix": "_".join(parts[6:]),  # Handles 'cifar10_cnn_new'
            "scenario": name,
        }
    except Exception as e:
        logger.warning(f"Error parsing scenario name '{name}': {e}")
        return {}


def parse_defense_hp_folder_name(name: str) -> Dict[str, Any]:
    """
    Parses 'aggregation.martfl.max_k_5_aggregation.clip_norm_10.0'
    """
    params = {}
    try:
        parts = name.split('_')
        i = 0
        while i < len(parts):
            key_parts = []
            while i < len(parts) and not parts[i].replace('.', '', 1).replace('-', '', 1).isdigit():
                key_parts.append(parts[i])
                i += 1
            param_key = "_".join(key_parts)
            param_key_short = param_key.split('.')[-1]
            if not param_key_short:
                i += 1  # Handle empty key part
                continue
            if i < len(parts):
                raw_value = parts[i]
                i += 1
                try:
                    if '.' in raw_value:
                        value = float(raw_value)
                    elif raw_value.lower() == 'none' or raw_value.lower() == 'null':
                        value = None
                    else:
                        value = int(raw_value)
                    params[param_key_short] = value
                except ValueError:
                    params[param_key_short] = raw_value
            else:
                pass
        return params
    except Exception as e:
        logger.warning(f"Error parsing HP folder name '{name}': {e}")
        return {}


# --- NEW Core Logic ---

def calculate_msr(selection_df: pd.DataFrame) -> float:
    """
    Calculates the Malicious Selection Rate (MSR) from a selection_history_df.
    """
    # Filter for only selected sellers
    selected_sellers = selection_df[selection_df['selected'] == True]

    if selected_sellers.empty:
        return 0.0  # No sellers were selected

    # Identify malicious sellers (whose IDs start with 'adv_')
    malicious_selected = selected_sellers['seller_id'].str.startswith('adv_').sum()

    total_selected = len(selected_sellers)

    return malicious_selected / total_selected


# FILE: visualize_step3.py

# ... (parsing functions) ...

def find_and_analyze_msr(root_dir: Path) -> pd.DataFrame:
    """Finds all selection_history.csv files and calculates MSR for each run."""
    logger.info(f"🔍 Scanning for Step 3 'selection_history.csv' files in: {root_dir}...")

    # --- THIS IS THE FIX ---
    # The correct structure is: results/step3_tune_.../hp_folder/run_folder/
    # This corresponds to: step3_tune_* / * / run_* / selection_history.csv
    search_pattern = "step3_tune_*/*/run_*/selection_history.csv"
    metrics_files = list(root_dir.rglob(search_pattern))
    # --- END FIX ---

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'selection_history.csv' files found matching the structure: {search_pattern}")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            # --- Robust Path Finding ---
            current_path = metrics_file.parent
            scenario_dir = None
            while current_path != root_dir and current_path != current_path.parent:
                if current_path.name.startswith("step3_tune_"):
                    scenario_dir = current_path
                    break
                current_path = current_path.parent
            if scenario_dir is None:
                logger.warning(f"Could not find scenario dir for {metrics_file}")
                continue

            relative_path_parts = metrics_file.parent.relative_to(scenario_dir).parts
            if len(relative_path_parts) < 2:
                logger.warning(f"Skipping {metrics_file}, unexpected path structure.")
                continue

            hp_dir_name = relative_path_parts[0]
            seed_dir_name = relative_path_parts[-1]
            seed_dir = metrics_file.parent

            if not (seed_dir / ".success").exists():
                logger.debug(f"Skipping failed run: {seed_dir.name}")
                continue

            scenario_info = parse_scenario_name_step3(scenario_dir.name)
            hp_info = parse_defense_hp_folder_name(hp_dir_name)

            if not scenario_info or "defense" not in scenario_info:
                logger.warning(f"Could not parse scenario info from {scenario_dir.name}")
                continue

            # Load the selection history
            selection_df = pd.read_csv(metrics_file)

            # --- Calculate MSR ---
            # For stability, calculate MSR over the last 50% of rounds
            max_round = selection_df['round'].max()
            stable_selection_df = selection_df[selection_df['round'] > (max_round / 2)]

            if stable_selection_df.empty:
                msr = calculate_msr(selection_df)  # Fallback to all rounds
            else:
                msr = calculate_msr(stable_selection_df)

            # Store combined record
            record = {
                **scenario_info,
                **hp_info,
                "msr": msr,
            }
            all_results.append(record)

        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}", exc_info=False)

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results: return pd.DataFrame()

    # --- Aggregate across seeds ---
    df = pd.DataFrame(all_results)

    # Identify HP columns (those parsed)
    hp_cols = [col for col in df.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix', 'msr'
    ] and not col.startswith('raw_')]

    group_cols = ['scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix'] + hp_cols

    agg_df = df.groupby(group_cols, dropna=False).agg(
        mean_msr=('msr', 'mean'),
        std_msr=('msr', 'std'),
        num_success_runs=('msr', 'count')
    ).reset_index()

    agg_df['std_msr'] = agg_df['std_msr'].fillna(0)

    return agg_df


# --- Visualization Functions (similar to your other script) ---
def create_msr_heatmap(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    if df_slice.empty:
        return

    hp_cols = [col for col in df_slice.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'mean_msr', 'std_msr', 'num_success_runs'
    ] and not col.startswith('raw_')]

    if len(hp_cols) != 2:
        logger.info(f"Skipping heatmap for {defense} (requires exactly 2 HPs, found {len(hp_cols)}).")
        return

    hp_x, hp_y = hp_cols[0], hp_cols[1]

    logger.info(f"Generating MSR heatmap for: {dataset} / {defense} / {attack}...")

    try:
        pivot_df = df_slice.pivot(index=hp_y, columns=hp_x, values='mean_msr')
    except Exception as e:
        logger.error(f"Failed to create pivot table for heatmap: {e}")
        return

    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".1%",  # <-- Format as percentage
        cmap="Greys",  # B/W colormap (lower is better, so dark is good)
        linewidths=.5,
        cbar_kws={'label': "Malicious Selection Rate (MSR)"}
    )

    ax.set_title(f"MSR Analysis (Filtering): {defense}\n({dataset} / {attack})")
    ax.set_xlabel(hp_x.replace("_", " ").title())
    ax.set_ylabel(hp_y.replace("_", " ").title())

    output_dir = Path("figures") / "step3_msr_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"msr_heatmap_{dataset}_{defense}_{attack}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def create_msr_barchart(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    if df_slice.empty:
        return

    # For 1-HP defenses like FLTrust
    hp_cols = [col for col in df_slice.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'mean_msr', 'std_msr', 'num_success_runs'
    ] and not col.startswith('raw_')]

    if len(hp_cols) != 1:
        logger.info(f"Skipping barchart for {defense} (requires exactly 1 HP, found {len(hp_cols)}).")
        return

    hp_x = hp_cols[0]

    logger.info(f"Generating MSR barchart for: {dataset} / {defense} / {attack} (vs {hp_x})...")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # We must convert the HP column to string to prevent seaborn
    # from plotting it as a continuous number line
    df_slice[hp_x] = df_slice[hp_x].astype(str)

    ax = sns.barplot(
        data=df_slice,
        x=hp_x,
        y='mean_msr',
        palette='Greys',
        edgecolor='black'
    )

    ax.set_title(f"MSR vs. {hp_x.title()} (Filtering): {defense}\n({dataset} / {attack})")
    ax.set_xlabel(hp_x.replace("_", " ").title())
    ax.set_ylabel("Malicious Selection Rate (MSR)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    output_dir = Path("figures") / "step3_msr_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"msr_barchart_{dataset}_{defense}_{attack}_{hp_x}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


# --- Main function to drive plotting ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze Malicious Selection Rate (MSR) from Step 3."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="The ROOT results directory (e.g., './results/')"
    )
    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve()

    if not results_path.exists():
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
        return

    try:
        agg_df = find_and_analyze_msr(results_path)

        if agg_df.empty:
            logger.error("No data was aggregated. Exiting.")
            return

        # --- Loop and create plots ---
        scenarios = agg_df[['dataset', 'attack_type', 'modality', 'model_suffix']].drop_duplicates()

        for _, row in scenarios.iterrows():
            dataset = row['dataset']
            attack = row['attack_type']
            model = row['model_suffix']

            # Get the full data for this scenario
            full_df_slice = agg_df[
                (agg_df['dataset'] == dataset) &
                (agg_df['attack_type'] == attack) &
                (agg_df['model_suffix'] == model)
                ]

            # --- Generate plots for each defense type ---

            # 1. MartFL (2D Heatmap)
            martfl_df = full_df_slice[full_df_slice['defense'] == 'martfl']
            create_msr_heatmap(martfl_df, dataset, attack, 'martfl')

            # 2. FLTrust (1D Barchart)
            fltrust_df = full_df_slice[full_df_slice['defense'] == 'fltrust']
            create_msr_barchart(fltrust_df, dataset, attack, 'fltrust')

            # 3. SkyMask (2D Heatmap)
            skymask_df = full_df_slice[full_df_slice['defense'] == 'skymask']
            create_msr_heatmap(skymask_df, dataset, attack, 'skymask')

        logger.info("\n✅ All MSR visualizations generated.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()