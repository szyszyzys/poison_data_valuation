# analyze_experiment_results.py
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # For nicer plots and heatmaps

from entry.gradient_market.visual.visual_privacy import display_gia_instance

# (Import your GIA visualization helpers: load_image_tensor_from_path, prep_for_grid, display_gia_instance)
# You'll also need the `flatten_dict` utility if you used it in save_round_logs_to_csv
# and `collate_batch_new` if you want to load raw text data for some reason (less likely here).

# --- Configuration ---
OUTPUT_ANALYSIS_DIR = Path("./analysis_output")
METRIC_MAPPING = {  # For prettier plot labels
    "perf_global_accuracy": "Global Model Accuracy",
    "perf_global_loss": "Global Model Loss",
    "perf_global_attack_success_rate": "Attack Success Rate (ASR)",
    "selection_rate_info_detection_rate (TPR)": "Adversary Detection Rate (TPR)",
    "selection_rate_info_false_positive_rate (FPR)": "Benign Misclassification Rate (FPR)",
    "metric_psnr": "PSNR (GIA)",
    "metric_ssim": "SSIM (GIA)",
    "metric_lpips": "LPIPS (GIA)",
    "duration_sec": "Attack Duration (s) (GIA)"
}
AGG_NAME_MAPPING = {  # For prettier legend entries
    "fedavg": "FedAvg",
    "martfl": "MartFL",
    "skymask": "SkyMask",
    "fltrust": "FLTrust"
}


# --- 1. Data Loading and Preprocessing ---

def load_experiment_config(config_path: Path) -> Optional[dict]:
    if config_path.exists():
        with open(config_path, 'r') as f:
            try:
                return json.load(f)  # Assuming you saved experiment_params.json
            except json.JSONDecodeError:
                try:  # Fallback to YAML if original config was saved
                    import yaml
                    # Need your MyDumper or a SafeLoader
                    return yaml.safe_load(f)
                except Exception as e_yaml:
                    logging.error(f"Could not load config {config_path} as JSON or YAML: {e_yaml}")
                    return None
    return None


def load_all_results(base_results_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads all round_results.csv and attack_results.csv from subdirectories.
    Extracts experiment parameters from corresponding experiment_params.json or config YAML.
    """
    all_round_logs = []
    all_attack_logs = []

    # Find all experiment_params.json files, then go up to find the run_X folder
    # Or, find all round_results.csv files.
    # Let's assume structure: base_results_dir / experiment_group / experiment_id / run_i / files.csv

    for round_results_file in base_results_dir.glob("**/round_results.csv"):
        run_dir = round_results_file.parent
        experiment_id_dir = run_dir.parent  # This should be the 'exp_id' directory

        # Try to load the specific config for this experiment_id
        # Path to experiment_params.json or the original config.yaml
        # This depends on how attack_new.py saved it. Let's assume experiment_params.json
        exp_params_path = experiment_id_dir / "experiment_params.json"
        config_data = load_experiment_config(exp_params_path)

        if config_data is None:
            logging.warning(f"Could not load params for results in {run_dir}. Skipping this run's logs.")
            continue

        # Extract key parameters from the loaded config_data
        # The config_data['full_config'] holds the original YAML structure
        original_config = config_data.get('full_config', {})

        run_params = {
            'experiment_id': original_config.get('experiment_id', experiment_id_dir.name),
            'dataset_name': original_config.get('dataset_name'),
            'model_structure': original_config.get('model_structure'),
            'aggregation_method': original_config.get('aggregation_method'),
            'adv_rate': original_config.get('data_split', {}).get('adv_rate'),
            'attack_type': original_config.get('attack', {}).get('attack_type') if original_config.get('attack',
                                                                                                       {}).get(
                'enabled') else 'None',
            'is_sybil': original_config.get('sybil', {}).get('is_sybil'),
            'sybil_amp_factor': original_config.get('sybil', {}).get('amplify_factor'),
            'discovery_quality': original_config.get('data_split', {}).get('dm_params', {}).get('discovery_quality'),
            'gia_performed': original_config.get('privacy_attack', {}).get('perform_gradient_inversion'),
            'run_seed_id': run_dir.name  # e.g., "run_0"
        }
        # Add more parameters you varied and want to analyze by

        try:
            df_round = pd.read_csv(round_results_file)
            for col, val in run_params.items():
                df_round[col] = val
            all_round_logs.append(df_round)
        except Exception as e:
            logging.error(f"Error loading/processing {round_results_file}: {e}")

        # Load corresponding attack_results.csv if it exists
        attack_results_file = run_dir / "attack_results.csv"  # GIA results
        if attack_results_file.exists():
            try:
                df_attack = pd.read_csv(attack_results_file)
                for col, val in run_params.items():
                    df_attack[col] = val
                all_attack_logs.append(df_attack)
            except Exception as e:
                logging.error(f"Error loading/processing {attack_results_file}: {e}")

    if not all_round_logs:
        logging.warning("No round_results.csv files were successfully loaded.")
        final_round_df = None
    else:
        final_round_df = pd.concat(all_round_logs, ignore_index=True)

    if not all_attack_logs:
        logging.info("No attack_results.csv (GIA) files found or loaded.")
        final_attack_df = None
    else:
        final_attack_df = pd.concat(all_attack_logs, ignore_index=True)

    return final_round_df, final_attack_df


# --- 2. Generic Plotting Utilities ---
def plot_metric_over_rounds(
        df: pd.DataFrame,
        metric_col: str,
        group_by_col: str,  # e.g., 'aggregation_method'
        filter_dict: Optional[Dict[str, Any]] = None,  # To select a subset of experiments
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        output_path: Optional[Path] = None,
        show_std: bool = True,
        rename_groups: Optional[Dict] = None  # For legend: {'fedavg': 'FedAvg'}
):
    """Plots a metric over rounds, grouped by a column, with optional filtering."""
    if df is None or df.empty or metric_col not in df.columns or 'round_number' not in df.columns:
        logging.warning(
            f"DataFrame empty or missing required columns ('{metric_col}', 'round_number') for plot_metric_over_rounds.")
        return

    plt.figure(figsize=(10, 6))

    # Apply filters
    query_parts = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in df.columns:
                logging.warning(f"Filter column '{col}' not in DataFrame. Skipping this filter.")
                continue
            if isinstance(val, list):
                query_parts.append(f"`{col}` in {val}")
            elif isinstance(val, str):
                query_parts.append(f"`{col}` == '{val}'")
            else:
                query_parts.append(f"`{col}` == {val}")
        if query_parts:
            df_filtered = df.query(" and ".join(query_parts),
                                   engine='python')  # engine='python' for more complex queries
        else:
            df_filtered = df
    else:
        df_filtered = df

    if df_filtered.empty:
        logging.warning(f"No data left after filtering for plot: {title if title else metric_col}")
        plt.close()
        return

    sns.set_theme(style="whitegrid")
    # Calculate mean and std for each group at each round
    # Make sure metric_col is numeric
    df_filtered[metric_col] = pd.to_numeric(df_filtered[metric_col], errors='coerce')

    grouped = df_filtered.groupby([group_by_col, 'round_number'])[metric_col]
    mean_metric = grouped.mean().reset_index()
    std_metric = grouped.std().reset_index()

    for group_val, group_df_mean in mean_metric.groupby(group_by_col):
        group_df_std = std_metric[std_metric[group_by_col] == group_val]
        legend_label = rename_groups.get(group_val, group_val) if rename_groups else group_val

        plt.plot(group_df_mean['round_number'], group_df_mean[metric_col], marker='o', linestyle='-',
                 label=legend_label, markersize=4, linewidth=1.5)
        if show_std and not group_df_std.empty:
            plt.fill_between(
                group_df_std['round_number'],
                group_df_mean[metric_col] - group_df_std[metric_col],
                group_df_mean[metric_col] + group_df_std[metric_col],
                alpha=0.2
            )

    plt.xlabel("Round Number")
    plt.ylabel(ylabel if ylabel else METRIC_MAPPING.get(metric_col, metric_col))
    plt.title(
        title if title else f"{METRIC_MAPPING.get(metric_col, metric_col)} vs. Rounds (Grouped by {group_by_col})")
    plt.legend(title=group_by_col)
    plt.grid(True, linestyle='--', alpha=0.7)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_final_metric_comparison_bar(
        df: pd.DataFrame,
        metric_col: str,
        main_group_col: str,  # e.g., 'aggregation_method' - for x-axis groups
        hue_col: Optional[str] = None,  # e.g., 'adv_rate' - for bars within each group
        filter_dict: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        output_path: Optional[Path] = None,
        rename_groups: Optional[Dict] = None
):
    """Plots a bar chart comparing the final (or best) value of a metric."""
    if df is None or df.empty or metric_col not in df.columns:
        logging.warning(f"DataFrame empty or missing '{metric_col}' for plot_final_metric_comparison_bar.")
        return

    # Apply filters
    query_parts = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in df.columns:
                logging.warning(f"Filter column '{col}' not in DataFrame. Skipping this filter.")
                continue
            if isinstance(val, list):
                query_parts.append(f"`{col}` in {val}")
            elif isinstance(val, str):
                query_parts.append(f"`{col}` == '{val}'")
            else:
                query_parts.append(f"`{col}` == {val}")
        if query_parts:
            df_filtered = df.query(" and ".join(query_parts), engine='python')
        else:
            df_filtered = df
    else:
        df_filtered = df

    if df_filtered.empty:
        logging.warning(f"No data left after filtering for bar plot: {title if title else metric_col}")
        return

    # Get data from the last round for each run
    # Ensure metric_col is numeric
    df_filtered[metric_col] = pd.to_numeric(df_filtered[metric_col], errors='coerce')

    # Group by experiment_id and run_seed_id to find the last round for each run
    last_round_indices = df_filtered.groupby(['experiment_id', 'run_seed_id'])['round_number'].idxmax()
    df_final_round = df_filtered.loc[last_round_indices]

    plt.figure(figsize=(12, 7) if hue_col else (8, 6))
    sns.set_theme(style="whitegrid")

    # Map main_group_col values for better legend/axis labels if rename_groups is provided
    if rename_groups and main_group_col in rename_groups:  # This is for the x-axis itself
        # This requires more complex handling if renaming x-axis ticks.
        # Simpler to rename hue_col or use rename_groups for hue legend.
        pass

    sns.barplot(x=main_group_col, y=metric_col, hue=hue_col, data=df_final_round, errorbar='sd', capsize=.1)

    plt.xlabel(main_group_col.replace("_", " ").title())
    plt.ylabel(ylabel if ylabel else METRIC_MAPPING.get(metric_col, metric_col))
    plt.title(title if title else f"Final {METRIC_MAPPING.get(metric_col, metric_col)} Comparison")
    if hue_col:
        plt.legend(title=hue_col.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout(rect=[0, 0, 0.85 if hue_col else 1, 1])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)  # bbox_inches='tight' might be needed if legend is outside
        logging.info(f"Saved bar plot: {output_path}")
    else:
        plt.show()
    plt.close()


# --- 3. Specific Analysis Functions ---

def analyze_baselines(df_rounds: pd.DataFrame, output_dir: Path):
    if df_rounds is None: return
    logging.info("\n--- Analyzing Baselines ---")
    baseline_df = df_rounds[df_rounds['attack_type'] == 'None'].copy()  # Ensure it's a copy
    if baseline_df.empty:
        logging.info("No baseline data found.")
        return

    for dataset in baseline_df['dataset_name'].unique():
        plot_metric_over_rounds(
            df=baseline_df,
            metric_col='perf_global_accuracy',
            group_by_col='aggregation_method',
            filter_dict={'dataset_name': dataset},
            title=f'Baseline Global Accuracy ({dataset})',
            output_path=output_dir / f"baseline_acc_{dataset}.png",
            rename_groups=AGG_NAME_MAPPING
        )


def analyze_poisoning_attacks(df_rounds: pd.DataFrame, attack_type_name: str, output_dir: Path):
    if df_rounds is None: return
    logging.info(f"\n--- Analyzing {attack_type_name} Attacks ---")
    attack_df = df_rounds[df_rounds['attack_type'] == attack_type_name].copy()
    if attack_df.empty:
        logging.info(f"No {attack_type_name} data found.")
        return

    # ASR vs. Rounds for different aggregators
    for dataset in attack_df['dataset_name'].unique():
        for adv_rate in attack_df['adv_rate'].unique():
            plot_metric_over_rounds(
                df=attack_df,
                metric_col='perf_global_attack_success_rate',
                group_by_col='aggregation_method',
                filter_dict={'dataset_name': dataset, 'adv_rate': adv_rate},
                title=f'{attack_type_name} ASR ({dataset}, Adv Rate: {adv_rate * 100:.0f}%)',
                output_path=output_dir / f"{attack_type_name}_asr_{dataset}_adv{adv_rate * 100:.0f}.png",
                rename_groups=AGG_NAME_MAPPING
            )
            # Main Task Accuracy vs. Rounds under attack
            plot_metric_over_rounds(
                df=attack_df,
                metric_col='perf_global_accuracy',
                group_by_col='aggregation_method',
                filter_dict={'dataset_name': dataset, 'adv_rate': adv_rate},
                title=f'{attack_type_name} Global Accuracy ({dataset}, Adv Rate: {adv_rate * 100:.0f}%)',
                output_path=output_dir / f"{attack_type_name}_acc_{dataset}_adv{adv_rate * 100:.0f}.png",
                rename_groups=AGG_NAME_MAPPING
            )
            # Bar plot of final ASR
            plot_final_metric_comparison_bar(
                df=attack_df,
                metric_col='perf_global_attack_success_rate',
                main_group_col='aggregation_method',
                # hue_col='trigger_type' or 'poison_rate' if you varied them for this attack_type
                filter_dict={'dataset_name': dataset, 'adv_rate': adv_rate},
                title=f'Final {attack_type_name} ASR ({dataset}, Adv Rate: {adv_rate * 100:.0f}%)',
                output_path=output_dir / f"final_{attack_type_name}_asr_{dataset}_adv{adv_rate * 100:.0f}.png",
                rename_groups=AGG_NAME_MAPPING
            )
            # Defense TPR/FPR if applicable
            if 'selection_rate_info_detection_rate (TPR)' in attack_df.columns:
                plot_final_metric_comparison_bar(
                    df=attack_df, metric_col='selection_rate_info_detection_rate (TPR)',
                    main_group_col='aggregation_method', filter_dict={'dataset_name': dataset, 'adv_rate': adv_rate},
                    title=f'Defense TPR against {attack_type_name} ({dataset}, Adv Rate: {adv_rate * 100:.0f}%)',
                    output_path=output_dir / f"defense_tpr_{attack_type_name}_{dataset}_adv{adv_rate * 100:.0f}.png",
                    rename_groups=AGG_NAME_MAPPING
                )


def analyze_sybil_attacks(df_rounds: pd.DataFrame, output_dir: Path):
    if df_rounds is None: return
    logging.info("\n--- Analyzing Sybil Attacks ---")
    sybil_df = df_rounds[df_rounds['is_sybil'] == True].copy()
    if sybil_df.empty:
        logging.info("No Sybil attack data found.")
        return

    # Compare accuracy with Sybil vs. No Sybil (from baselines)
    # This requires merging or careful filtering. For now, just plot Sybil results.
    for dataset in sybil_df['dataset_name'].unique():
        for amp_factor in sybil_df['sybil_amp_factor'].unique():
            # Plot accuracy vs rounds for different aggregators under Sybil
            plot_metric_over_rounds(
                df=sybil_df,
                metric_col='perf_global_accuracy',
                group_by_col='aggregation_method',
                filter_dict={'dataset_name': dataset, 'sybil_amp_factor': amp_factor},
                title=f'Global Accuracy under Sybil (Dataset: {dataset}, Amp: {amp_factor})',
                output_path=output_dir / f"sybil_acc_{dataset}_amp{amp_factor}.png",
                rename_groups=AGG_NAME_MAPPING
            )
            # If Sybil is combined with backdoor, plot ASR
            if 'perf_global_attack_success_rate' in sybil_df.columns:
                plot_metric_over_rounds(
                    df=sybil_df[sybil_df['attack_type'] == 'backdoor'],  # Filter for backdoor + sybil
                    metric_col='perf_global_attack_success_rate',
                    group_by_col='aggregation_method',
                    filter_dict={'dataset_name': dataset, 'sybil_amp_factor': amp_factor},
                    title=f'ASR under Sybil+Backdoor (Dataset: {dataset}, Amp: {amp_factor})',
                    output_path=output_dir / f"sybil_backdoor_asr_{dataset}_amp{amp_factor}.png",
                    rename_groups=AGG_NAME_MAPPING
                )


def analyze_discovery_split(df_rounds: pd.DataFrame, output_dir: Path):
    if df_rounds is None: return
    logging.info("\n--- Analyzing Discovery Split ---")
    # Assuming 'discovery_quality' and 'buyer_data_mode' are columns from config
    discovery_df = df_rounds[df_rounds['data_split_mode'] == 'discovery'].copy()  # Example filter
    if discovery_df.empty:
        logging.info("No discovery split data found.")
        return

    for dataset in discovery_df['dataset_name'].unique():
        # Compare final accuracy for different discovery_quality values
        plot_final_metric_comparison_bar(
            df=discovery_df,
            metric_col='perf_global_accuracy',
            main_group_col='aggregation_method',  # Or 'discovery_quality'
            hue_col='discovery_quality',  # Or 'buyer_data_mode'
            filter_dict={'dataset_name': dataset},
            title=f'Final Accuracy by Discovery Quality ({dataset})',
            output_path=output_dir / f"discovery_acc_vs_quality_{dataset}.png",
            rename_groups=AGG_NAME_MAPPING
        )


def analyze_gia_results(df_gia: pd.DataFrame, base_data_path: Path, output_dir: Path, num_viz_examples: int = 3):
    if df_gia is None or df_gia.empty:
        logging.info("No GIA results to analyze.")
        return
    logging.info("\n--- Analyzing Gradient Inversion Attacks ---")

    # Summary table for GIA metrics
    gia_metric_cols = [col for col in df_gia.columns if col.startswith('metric_')]
    if gia_metric_cols:
        # Convert to numeric, coercing errors
        for col in gia_metric_cols:
            df_gia[col] = pd.to_numeric(df_gia[col], errors='coerce')

        gia_summary = df_gia[gia_metric_cols + ['duration_sec']].agg(['mean', 'median', 'std', 'count'])
        print(
            "\nGIA Metrics Summary (successful attacks if 'error' column was used for filtering before this function):")
        print(gia_summary.T.to_string())
        gia_summary.T.to_csv(output_dir / "gia_metrics_summary.csv")
        logging.info(f"Saved GIA metrics summary to {output_dir / 'gia_metrics_summary.csv'}")

    # Visualize some examples (best/worst PSNR, best LPIPS)
    # Filter for entries where GIA itself didn't fail (top-level 'error' is NaN)
    successful_gia_df = df_gia[df_gia['error'].isna()].copy()
    if successful_gia_df.empty:
        logging.warning("No successful GIA attempts logged for visualization.")
        return

    for col in gia_metric_cols:  # Ensure numeric again for sorting on this subset
        successful_gia_df[col] = pd.to_numeric(successful_gia_df[col], errors='coerce')

    # Best PSNR
    if 'metric_psnr' in successful_gia_df.columns and num_viz_examples > 0:
        best_psnr_entries = successful_gia_df.nlargest(num_viz_examples, 'metric_psnr')
        logging.info(f"\nVisualizing {len(best_psnr_entries)} GIA cases with Best PSNR...")
        for _, row in best_psnr_entries.iterrows():
            display_gia_instance(row, base_data_path, output_dir / "gia_visualizations")

    # Best LPIPS (lower is better)
    if 'metric_lpips' in successful_gia_df.columns and num_viz_examples > 0:
        best_lpips_entries = successful_gia_df[successful_gia_df['metric_lpips'].notna()].nsmallest(num_viz_examples,
                                                                                                    'metric_lpips')
        logging.info(f"\nVisualizing {len(best_lpips_entries)} GIA cases with Best LPIPS...")
        for _, row in best_lpips_entries.iterrows():
            display_gia_instance(row, base_data_path, output_dir / "gia_visualizations")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from federated learning experiments.")
    parser.add_argument(
        "base_results_dir", type=Path,
        help="Base directory containing all experiment results (e.g., './experiment_results_revised')"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_ANALYSIS_DIR,
        help=f"Directory to save analysis plots and tables (default: {OUTPUT_ANALYSIS_DIR})"
    )
    # Add more specific args if needed, e.g., which analyses to run

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and aggregate all data
    df_rounds_all, df_gia_all = load_all_results(args.base_results_dir)

    if df_rounds_all is None and df_gia_all is None:
        logging.error("No data loaded. Exiting analysis.")
        exit()

    # 2. Run specific analyses
    if df_rounds_all is not None:
        analyze_baselines(df_rounds_all, args.output_dir / "baselines")
        analyze_poisoning_attacks(df_rounds_all, "backdoor", args.output_dir / "backdoor_attacks")
        analyze_poisoning_attacks(df_rounds_all, "label_flip", args.output_dir / "label_flip_attacks")
        analyze_sybil_attacks(df_rounds_all, args.output_dir / "sybil_attacks")
        analyze_discovery_split(df_rounds_all, args.output_dir / "discovery_split")

    if df_gia_all is not None:
        # For GIA, base_data_path is tricky. It's the save_dir used by perform_and_evaluate_inversion_attack.
        # If your 'privacy_attack_path' in config was relative to the run's save_path, then
        # base_data_path for GIA images would be the run_dir.
        # The `display_gia_instance` expects `base_data_path / row['reconstructed_image_file']`.
        # If `reconstructed_image_file` in CSV is already effectively `exp_id/run_i/tensors/file.pt`, then `base_data_path`
        # for GIA should be `args.base_results_dir.parent` or similar, or you adjust path joining.
        # Assuming `reconstructed_image_file` logged by your GIA function is relative to the *run's save path*
        # e.g. if run save path is results/exp1/run0, and image is run0/tensors/img.pt
        # then `base_data_path` needs to be the root of that, i.e. results/exp1.
        # This is complex. For simplicity now, let's assume files in CSV for GIA are relative to args.base_results_dir
        analyze_gia_results(df_gia_all, args.base_results_dir, args.output_dir / "gia_analysis", num_viz_examples=3)
        # NOTE: The base_data_path for analyze_gia_results needs careful thought on how paths are stored in attack_results.csv

    logging.info(f"Analysis complete. Outputs saved in {args.output_dir}")
