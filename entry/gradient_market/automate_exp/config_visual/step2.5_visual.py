import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Parsers (Copied from your script) ---
HPARAM_REGEX = re.compile(r"opt_(?P<optimizer>\w+)_lr_(?P<lr>[\d\.]+)_epochs_(?P<epochs>\d+)")
SEED_REGEX_1 = re.compile(r"run_\d+_seed_(\d+)")
SEED_REGEX_2 = re.compile(r".*_seed-(\d+)")
DATA_SETTING_REGEX = re.compile(r".*_(?P<data_setting>iid|noniid)$")
EXP_NAME_REGEX = re.compile(
    r"(?P<base_name>(?:step|new_step)[\w\.-]+?)(_(?P<data_setting_exp>iid|noniid))?$"
)


def parse_hp_from_name(name: str) -> Dict[str, Any]:
    match = HPARAM_REGEX.match(name)
    if not match: return {}
    try:
        data = match.groupdict();
        data['lr'] = float(data['lr']);
        data['epochs'] = int(data['epochs']);
        return data
    except ValueError:
        return {}


def parse_sub_exp_name(name: str) -> Dict[str, str]:
    params = {}
    try:
        parts = name.split('_');
        for part in parts:
            if '-' in part:
                key, value = part.split('-', 1)
                if key in ['ds', 'model', 'agg']: params[key] = value
    except Exception as e:
        logger.warning(f"Could not parse sub-experiment name: {name} ({e})")
    return params


def parse_seed_from_name(name: str) -> int:
    for regex in [SEED_REGEX_1, SEED_REGEX_2]:
        match = regex.match(name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    if "seed" in name:
        try:
            return int(name.split('_')[-1])
        except (ValueError, IndexError):
            pass
    return -1


def parse_exp_context(name: str) -> Dict[str, str]:
    match = EXP_NAME_REGEX.match(name)
    if match:
        data = match.groupdict()
        return {"base_name": data.get("base_name") or "unknown_base", "data_setting": data.get("data_setting_exp")}
    ds_match = DATA_SETTING_REGEX.match(name)
    if ds_match: return {"data_setting": ds_match.group("data_setting")}
    return {}


# --- This is YOUR find_all_results function ---
def find_all_results(results_dir: Path, clip_mode: str, exp_filter: str) -> List[Dict[str, Any]]:
    logger.info(f"🔍 Scanning recursively for results in: {results_dir}...")
    logger.info(f"   Filtering by --clip_mode: '{clip_mode}'")
    logger.info(f"   Filtering by --exp_filter: must contain '{exp_filter}'")
    metrics_files = list(results_dir.rglob("final_metrics.json"))

    if not metrics_files:
        logger.error(f"❌ ERROR: No 'final_metrics.json' files found in {results_dir}.")
        return []

    logger.info(f"✅ Found {len(metrics_files)} total individual run results.")
    all_results = []

    for metrics_file in metrics_files:
        try:
            run_dir = metrics_file.parent
            if not (run_dir / ".success").exists():
                logger.debug(f"Skipping failed/incomplete run: {run_dir.name}")
                continue

            record = {
                "base_name": None, "data_setting": None, "optimizer": None,
                "lr": None, "epochs": None, "ds": None, "model": None, "agg": None,
                "seed": -1, "full_path": str(metrics_file),
                "clip_setting": "unknown"
            }

            current_path = metrics_file.parent
            while str(current_path).startswith(str(results_dir)) and current_path != results_dir.parent:
                folder_name = current_path.name

                if record["optimizer"] is None:
                    hps = parse_hp_from_name(folder_name)
                    if hps: record.update(hps)

                if record["base_name"] is None:
                    if "nolocalclip" in folder_name:
                        record["clip_setting"] = "no_local_clip"
                        folder_name = folder_name.replace("_nolocalclip", "")
                    else:
                        record["clip_setting"] = "local_clip"

                    exp_context = parse_exp_context(folder_name)
                    if exp_context and "base_name" in exp_context:
                        record.update(exp_context)

                if record["ds"] is None:
                    sub_exp_params = parse_sub_exp_name(folder_name)
                    if sub_exp_params: record.update(sub_exp_params)

                if record["seed"] == -1:
                    seed = parse_seed_from_name(folder_name)
                    if seed != -1: record["seed"] = seed

                current_path = current_path.parent

            if record.get("base_name") is None or exp_filter not in record["base_name"]:
                logger.debug(f"Skipping file: 'base_name' ({record.get('base_name')}) does not contain '{exp_filter}'")
                continue

            if clip_mode != "all" and record["clip_setting"] != clip_mode:
                logger.debug(f"Skipping file due to --clip_mode filter: {record['clip_setting']}")
                continue

            if record["optimizer"] is None:
                logger.warning(f"Could not find HP folder (opt_...) for {metrics_file}. Skipping.")
                continue

            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)

            record.update(metrics_data)
            all_results.append(record)

        except Exception as e:
            logger.warning(f"Warning: Could not process file {metrics_file}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully processed {len(all_results)} runs matching filter.")
    return all_results


# ==============================================================================
# === 1. USER ACTION REQUIRED: FILL IN YOUR IID BASELINES ===
# ==============================================================================
IID_BASELINES = {
    "texas100": 0.6250,
    "purchase100": 0.6002,
    "cifar10": 0.8248,
    "cifar100": 0.5536,
    "trec": 0.7985,
}
USABLE_THRESHOLD = 0.90


# ==============================================================================

# --- NEW: Function to prepare the data for plotting ---
def prepare_plot_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw dataframe and returns a new dataframe with
    the 'usable_rate' calculated for each (defense, dataset, clip_setting).
    """
    if raw_df.empty:
        logger.warning("No data to analyze.")
        return pd.DataFrame()

    if IID_BASELINES.get("cifar10", 0.0) == 0.0:
        logger.error("STOP: 'IID_BASELINES' dictionary is not filled.")
        return pd.DataFrame()

    # --- 1. Add 'defense' and 'dataset' columns ---
    raw_df['defense'] = raw_df['agg']
    raw_df['dataset'] = raw_df['ds']

    # --- 2. Aggregate across seeds (same as your script) ---
    group_cols = ["defense", "dataset", "modality", "optimizer", "lr", "epochs", "clip_setting"]
    available_group_cols = [col for col in group_cols if col in raw_df.columns]

    numeric_cols = ["acc", "asr"]
    raw_df[numeric_cols] = raw_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    raw_df = raw_df.dropna(subset=['acc'])

    raw_df = raw_df.rename(columns={"acc": "test_acc", "asr": "backdoor_asr"})
    numeric_cols = ["test_acc", "backdoor_asr"]

    hp_agg_df = raw_df.groupby(available_group_cols, as_index=False)[numeric_cols].mean()

    # --- 3. Calculate 'is_usable' (same as your script) ---
    hp_agg_df['iid_baseline_acc'] = hp_agg_df['dataset'].map(IID_BASELINES)
    hp_agg_df = hp_agg_df.dropna(subset=['iid_baseline_acc'])
    hp_agg_df['relative_perf'] = hp_agg_df['test_acc'] / hp_agg_df['iid_baseline_acc']
    hp_agg_df['is_usable'] = hp_agg_df['relative_perf'] >= USABLE_THRESHOLD

    # --- 4. Calculate Usable Rate ---
    # This is the new part for plotting
    plot_group_cols = ['dataset', 'defense', 'clip_setting']

    # Calculate the mean of the 'is_usable' (True/False or 1/0) column
    usable_rate_df = hp_agg_df.groupby(plot_group_cols, as_index=False)['is_usable'].mean()

    # Rename for clarity in the plot
    usable_rate_df = usable_rate_df.rename(columns={'is_usable': 'usable_rate'})

    return usable_rate_df


# --- NEW: Plotting Function ---
def create_usability_plot(plot_df: pd.DataFrame, clip_setting: str, output_filename: str):
    """
    Creates a black-and-white bar chart of the usable rate for a specific clip_setting.
    """
    df_filtered = plot_df[plot_df['clip_setting'] == clip_setting]

    if df_filtered.empty:
        logger.warning(f"No data to plot for clip_setting='{clip_setting}'. Skipping plot.")
        return

    logger.info(f"Generating plot for: {clip_setting}...")

    # Set style
    sns.set_style("whitegrid")

    # Create the plot
    g = sns.catplot(
        data=df_filtered,
        x='dataset',
        y='usable_rate',
        hue='defense',
        kind='bar',
        palette='grayscale',  # Creates B/W plot with shades of gray
        edgecolor='black',
        aspect=1.5  # Make the plot wider
    )

    # --- Customization ---
    # Set Y-axis to percentage
    g.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    g.ax.set_ylim(0, 1.05)  # Set Y-axis from 0% to 105%

    # Set titles and labels
    g.set_axis_labels("Dataset", "Usable HP Rate (%)")
    g.fig.suptitle(f"Training HP Usability Rate ({clip_setting.replace('_', ' ').title()})", y=1.03)

    # Adjust legend
    g.legend.set_title("Defense")

    # Save the figure
    g.fig.savefig(output_filename, bbox_inches='tight')
    logger.info(f"✅ Plot saved to: {output_filename}")
    plt.close(g.fig)


# --- NEW: Main function to drive plotting ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze and VISUALIZE FL Sensitivity Analysis results (Step 2.5)."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        default="./results",
        help="The root directory where all experiment results are stored (default: ./results)"
    )
    parser.add_argument(
        "--exp_filter",
        type=str,
        default="step2.5_find_hps",
        help="A string that must be present in the experiment base name (default: 'step2.5_find_hps')."
    )

    args = parser.parse_args()
    results_path = Path(args.results_dir).resolve()

    if not results_path.exists():
        logger.error(f"❌ ERROR: The directory '{results_path}' does not exist.")
        return

    pd.set_option('display.float_format', '{:,.4f}'.format)

    try:
        # --- 1. Load data for BOTH clip settings ---
        raw_results_all = find_all_results(results_path, clip_mode="all", exp_filter=args.exp_filter)

        if not raw_results_all:
            logger.warning(f"No valid results found matching filter: --exp_filter='{args.exp_filter}'")
            return

        df_all = pd.DataFrame(raw_results_all)

        # --- 2. Prepare the plotting dataframe ---
        plot_df = prepare_plot_data(df_all)

        if plot_df.empty:
            logger.error("❌ Failed to prepare data for plotting.")
            return

        # --- 3. Create the plots ---
        create_usability_plot(
            plot_df,
            clip_setting="local_clip",
            output_filename="step2.5_usability_with_local_clip.png"
        )

        create_usability_plot(
            plot_df,
            clip_setting="no_local_clip",
            output_filename="step2.5_usability_no_local_clip.png"
        )

        logger.info("\n✅ Visualization complete.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()