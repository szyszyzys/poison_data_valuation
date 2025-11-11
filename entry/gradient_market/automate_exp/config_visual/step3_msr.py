import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANKING_METRIC = 'acc_minus_asr'  # score = mean_test_acc - mean_backdoor_asr


# --- Parsing Functions (Unchanged) ---

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
            "model_suffix": "_".join(parts[6:]),
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
                i += 1
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


# --- Core Logic (Unchanged) ---

def calculate_msr(selection_df: pd.DataFrame) -> float:
    """
    Calculates the Malicious Selection Rate (MSR) from a selection_history_df.
    """
    if 'selected' in selection_df.columns:
        selected_sellers = selection_df[selection_df['selected'] == True]
    elif 'weight' in selection_df.columns:
        selected_sellers = selection_df[selection_df['weight'] > 0]
    else:
        logger.warning("Could not find 'selected' or 'weight' column. MSR will be 0.")
        return 0.0

    if selected_sellers.empty:
        return 0.0

    malicious_selected = selected_sellers['seller_id'].str.startswith('adv_').sum()
    total_selected = len(selected_sellers)

    return malicious_selected / total_selected


def find_all_tuning_results(root_dir: Path) -> pd.DataFrame:
    """Finds all final_metrics.json AND seller_metrics.csv to parse context."""
    logger.info(f"🔍 Scanning for Step 3 results in: {root_dir}...")
    search_pattern = "step3_tune_*/*/*/run_*/final_metrics.json"
    metrics_files = list(root_dir.rglob(search_pattern))

    if not metrics_files:
        logger.error(
            f"❌ ERROR: No 'final_metrics.json' files found recursively in {root_dir} matching {search_pattern}.")
        return pd.DataFrame()

    logger.info(f"✅ Found {len(metrics_files)} individual run results.")

    all_results = []
    for metrics_file in metrics_files:
        try:
            current_path = metrics_file.parent
            scenario_dir = None
            while current_path != root_dir and current_path != current_path.parent:
                if current_path.name.startswith("step3_tune_"):
                    scenario_dir = current_path
                    break
                current_path = current_path.parent
            if scenario_dir is None: continue

            relative_path_parts = metrics_file.parent.relative_to(scenario_dir).parts
            if len(relative_path_parts) < 2: continue

            hp_dir_name = relative_path_parts[0]
            seed_dir_name = relative_path_parts[-1]
            seed_dir = metrics_file.parent

            if not (seed_dir / ".success").exists():
                continue

            scenario_info = parse_scenario_name_step3(scenario_dir.name)
            hp_info = parse_defense_hp_folder_name(hp_dir_name)

            if not scenario_info or ("defense" not in scenario_info):
                continue

            msr = np.nan
            seller_metrics_path = metrics_file.parent / "seller_metrics.csv"
            if not seller_metrics_path.exists():
                seller_metrics_path = metrics_file.parent / "selection_history.csv"

            if seller_metrics_path.exists():
                selection_df = pd.read_csv(seller_metrics_path)
                max_round = selection_df['round'].max()
                stable_selection_df = selection_df[selection_df['round'] > (max_round / 2)]

                if stable_selection_df.empty:
                    msr = calculate_msr(selection_df)
                else:
                    msr = calculate_msr(stable_selection_df)
            else:
                logger.warning(f"No seller_metrics.csv or selection_history.csv found in {seed_dir}")

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            test_acc = metrics.get("test_acc", metrics.get("acc"))
            backdoor_asr = metrics.get("backdoor_asr", metrics.get("test_asr", metrics.get("asr")))
            if backdoor_asr is None:
                backdoor_asr = 0.0

            record = {
                **scenario_info,
                **hp_info,
                "seed": int(seed_dir_name.split('_')[-1]) if "run" in seed_dir_name else -1,
                "test_acc": test_acc,
                "backdoor_asr": backdoor_asr,
                "msr": msr
            }
            all_results.append(record)

        except Exception as e:
            logger.warning(f"Could not process file {metrics_file}: {e}", exc_info=False)

    logger.info(f"Successfully processed {len(all_results)} valid runs.")
    if not all_results: return pd.DataFrame()

    return pd.DataFrame(all_results)


def analyze_defense_tuning(raw_df: pd.DataFrame, results_dir: Path) -> (pd.DataFrame, pd.DataFrame):  # Type hint
    """Aggregates results and finds the best defense HPs."""
    if raw_df.empty:
        logger.warning("No data to analyze.")
        return pd.DataFrame(), pd.DataFrame()

    hp_cols = [col for col in raw_df.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'hp_folder', 'seed', 'status', 'test_acc', 'backdoor_asr', 'msr'
    ] and not col.startswith('raw_')]
    logger.info(f"Identified Defense HP columns: {hp_cols}")

    group_cols = ['scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix'] + hp_cols

    agg_df = raw_df.groupby(group_cols, dropna=False).agg(
        mean_test_acc=('test_acc', 'mean'),
        std_test_acc=('test_acc', 'std'),
        mean_backdoor_asr=('backdoor_asr', 'mean'),
        std_backdoor_asr=('backdoor_asr', 'std'),
        mean_msr=('msr', 'mean'),
        std_msr=('msr', 'std'),
        num_success_runs=('seed', 'count')
    ).reset_index()

    agg_df = agg_df.fillna(0)  # Fill all NaNs for simplicity

    if RANKING_METRIC == 'acc_minus_asr':
        agg_df['score'] = agg_df['mean_test_acc'] - agg_df['mean_backdoor_asr']
    else:
        agg_df['score'] = agg_df['mean_test_acc']

    scenario_group_cols = ['defense', 'attack_type', 'modality', 'dataset', 'model_suffix']
    try:
        best_idx = agg_df.loc[agg_df.groupby(scenario_group_cols)['score'].idxmax()]
    except ValueError as ve:
        logger.error(f"Error finding best HPs, possibly due to empty groups: {ve}")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DFs

    best_df = best_idx.sort_values(by=['modality', 'dataset', 'model_suffix', 'attack_type', 'defense'])

    display_cols = scenario_group_cols + hp_cols + [
        'mean_test_acc', 'mean_backdoor_asr', 'mean_msr', 'score'  # <-- ADDED MSR
    ]
    display_cols = [col for col in display_cols if col in best_df.columns]

    print("\n" + "=" * 120)
    print(f"🏆 Best Defense Hyperparameters Found (Ranked by: '{RANKING_METRIC}')")
    print("=" * 120)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.float_format', '{:,.4f}'.format):
        print(best_df[display_cols].to_string(index=False))

    output_csv_all = results_dir / "step3_defense_tuning_all_aggregated.csv"
    output_csv_best = results_dir / "step3_defense_tuning_best_hps.csv"
    try:
        agg_df.sort_values(by=scenario_group_cols + ['score'],
                           ascending=[True] * len(scenario_group_cols) + [False]).to_csv(output_csv_all, index=False,
                                                                                         float_format="%.5f")
        logger.info(f"\n✅ Full aggregated tuning results saved to: {output_csv_all}")
        best_df[display_cols].to_csv(output_csv_best, index=False, float_format="%.5f")
        logger.info(f"✅ Best hyperparameters saved to: {output_csv_best}")
    except Exception as e:
        logger.warning(f"\n⚠️ Could not save aggregated results to CSV: {e}")

    print("\n" + "=" * 120)
    print("Analysis complete. Use this table to update TUNED_DEFENSE_PARAMS.")

    return agg_df, best_df


# --- Visualization Functions (MSR) ---
def create_msr_heatmap(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    if df_slice.empty: return

    # --- THIS IS THE FIX ---
    # Explicitly define which HPs to plot for 2D defenses
    hp_map = {
        'martfl': ('clip_norm', 'max_k'),
        'skymask': ('mask_lr', 'mask_threshold')  # Example: plot 2 of SkyMask's 3 HPs
        # (You can change 'skymask' to plot other pairs)
    }

    if defense not in hp_map:
        logger.info(f"[MSR Plot] Skipping heatmap for {defense} (not in hp_map).")
        return

    hp_x, hp_y = hp_map[defense]

    # Check if this slice actually swept these HPs
    if hp_x not in df_slice.columns or hp_y not in df_slice.columns:
        logger.warning(f"SkiTry topping heatmap for {defense}: Data is missing HP columns {hp_x} or {hp_y}")
        return

    logger.info(f"Generating MSR heatmap for: {dataset} / {defense} / {attack}...")

    try:
        # We must fillna for pivoting to work
        df_slice[hp_x] = df_slice[hp_x].fillna('None')
        df_slice[hp_y] = df_slice[hp_y].fillna('None')
        pivot_df = df_slice.pivot(index=hp_y, columns=hp_x, values='mean_msr')
    except Exception as e:
        logger.error(f"Failed to create MSR pivot table for heatmap: {e}");
        return

    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    ax = sns.heatmap(
        pivot_df, annot=True, fmt=".1%", cmap="Greys", linewidths=.5,
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


# --- **** NEW ASR VISUALIZATION FUNCTIONS **** ---

def create_asr_heatmap(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    """ Generates a heatmap for ASR vs. 2 defense HPs. """
    if df_slice.empty: return
    df_slice = df_slice.dropna(axis=1, how='all')
    hp_cols = [col for col in df_slice.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'mean_msr', 'std_msr', 'num_success_runs', 'mean_test_acc', 'std_test_acc',
        'mean_backdoor_asr', 'std_backdoor_asr', 'score'
    ] and not col.startswith('raw_')]

    if len(hp_cols) != 2:
        logger.info(f"[ASR Plot] Skipping heatmap for {defense} (requires 2 HPs, found {len(hp_cols)}: {hp_cols}).")
        return

    hp_x, hp_y = hp_cols[0], hp_cols[1]
    logger.info(f"Generating ASR heatmap for: {dataset} / {defense} / {attack}...")

    try:
        pivot_df = df_slice.pivot(index=hp_y, columns=hp_x, values='mean_backdoor_asr')
    except Exception as e:
        logger.error(f"Failed to create ASR pivot table for heatmap: {e}");
        return

    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    ax = sns.heatmap(
        pivot_df, annot=True, fmt=".1%", cmap="Greys", linewidths=.5,
        cbar_kws={'label': "Attack Success Rate (ASR)"}
    )
    ax.set_title(f"ASR Analysis (Robustness): {defense}\n({dataset} / {attack})")
    ax.set_xlabel(hp_x.replace("_", " ").title())
    ax.set_ylabel(hp_y.replace("_", " ").title())

    output_dir = Path("figures") / "step3_asr_analysis"  # <-- New folder
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"asr_heatmap_{dataset}_{defense}_{attack}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


def create_msr_barchart(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    if df_slice.empty: return

    # --- THIS IS THE FIX ---
    # Explicitly define which HPs to plot for 1D defenses
    hp_map = {
        'fltrust': 'clip_norm'
    }

    if defense not in hp_map:
        logger.info(f"[MSR Plot] Skipping barchart for {defense} (not in hp_map).")
        return

    hp_x = hp_map[defense]

    if hp_x not in df_slice.columns:
        logger.warning(f"Skipping barchart for {defense}: Data is missing HP column {hp_x}")
        return
    # --- END FIX ---

    logger.info(f"Generating MSR barchart for: {dataset} / {defense} / {attack} (vs {hp_x})...")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    # We must fillna and convert to string to treat 'None' as a category
    df_slice[hp_x] = df_slice[hp_x].fillna('None').astype(str)

    ax = sns.barplot(
        data=df_slice, x=hp_x, y='mean_msr', palette='Greys', edgecolor='black'
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

def create_asr_barchart(df_slice: pd.DataFrame, dataset: str, attack: str, defense: str):
    """ Generates a barchart for ASR vs. 1 defense HP. """
    if df_slice.empty: return
    df_slice = df_slice.dropna(axis=1, how='all')
    hp_cols = [col for col in df_slice.columns if col not in [
        'scenario', 'defense', 'attack_type', 'modality', 'dataset', 'model_suffix',
        'mean_msr', 'std_msr', 'num_success_runs', 'mean_test_acc', 'std_test_acc',
        'mean_backdoor_asr', 'std_backdoor_asr', 'score'
    ] and not col.startswith('raw_')]

    if len(hp_cols) != 1:
        logger.info(f"[ASR Plot] Skipping barchart for {defense} (requires 1 HP, found {len(hp_cols)}: {hp_cols}).")
        return

    hp_x = hp_cols[0]
    logger.info(f"Generating ASR barchart for: {dataset} / {defense} / {attack} (vs {hp_x})...")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    df_slice[hp_x] = df_slice[hp_x].astype(str)

    ax = sns.barplot(
        data=df_slice, x=hp_x, y='mean_backdoor_asr', palette='Greys', edgecolor='black'
    )
    ax.set_title(f"ASR vs. {hp_x.title()} (Robustness): {defense}\n({dataset} / {attack})")
    ax.set_xlabel(hp_x.replace("_", " ").title())
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    output_dir = Path("figures") / "step3_asr_analysis"  # <-- New folder
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"asr_barchart_{dataset}_{defense}_{attack}_{hp_x}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()


# --- MODIFIED main ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze and Visualize FL Defense Tuning results (Step 3)."
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
        raw_results_df = find_all_tuning_results(results_path)
        agg_df, best_df = analyze_defense_tuning(raw_results_df, results_path)

        if agg_df.empty:
            logger.error("No data was aggregated. Exiting.")
            return

        # --- 2. Loop and create plots ---
        scenarios = agg_df[['dataset', 'attack_type', 'modality', 'model_suffix']].drop_duplicates()

        for _, row in scenarios.iterrows():
            dataset = row['dataset']
            attack = row['attack_type']
            model = row['model_suffix']

            # Get the full data (all HPs) for this scenario
            full_df_slice = agg_df[
                (agg_df['dataset'] == dataset) &
                (agg_df['attack_type'] == attack) &
                (agg_df['model_suffix'] == model)
                ]

            # --- Generate plots for each defense type ---
            for defense_name in full_df_slice['defense'].unique():
                if defense_name == 'fedavg': continue

                defense_df = full_df_slice[full_df_slice['defense'] == defense_name]

                # Plot 1: MSR (Filtering Rate)
                create_msr_heatmap(defense_df, dataset, attack, defense_name)
                create_msr_barchart(defense_df, dataset, attack, defense_name)

                # Plot 2: ASR (Robustness Result)
                create_asr_heatmap(defense_df, dataset, attack, defense_name)
                create_asr_barchart(defense_df, dataset, attack, defense_name)

        logger.info("\n✅ All MSR and ASR visualizations generated.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
