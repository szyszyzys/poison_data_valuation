import argparse
import logging  # For GIA viz
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from entry.gradient_market.visual.new_result_process import load_all_results

# --- Assume these are imported from your analyze_experiment_results.py or a utils file ---
# from analyze_experiment_results import (
#     plot_metric_over_rounds,
#     plot_final_metric_comparison_bar,
#     display_gia_instance, # For GIA qualitative
#     load_image_tensor_from_path, # For GIA qualitative
#     prep_for_grid, # For GIA qualitative
#     METRIC_MAPPING,
#     AGG_NAME_MAPPING
# )
# --- For standalone use, let's redefine stubs for plotting functions ---
# In a real script, you'd import these.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METRIC_MAPPING = {
    "perf_global_accuracy": "Global Model Accuracy",
    "perf_global_loss": "Global Model Loss",
    "perf_global_attack_success_rate": "Attack Success Rate (ASR)",
    "selection_rate_info_detection_rate (TPR)": "Adversary Detection Rate (TPR)",
    "selection_rate_info_false_positive_rate (FPR)": "Benign Misclassification Rate (FPR)",
    "metric_psnr": "PSNR (GIA)", "metric_ssim": "SSIM (GIA)", "metric_lpips": "LPIPS (GIA)",
    "duration_sec": "Attack Duration (s) (GIA)"
}
AGG_NAME_MAPPING = {"fedavg": "FedAvg", "martfl": "MartFL", "skymask_utils": "SkyMask", "fltrust": "FLTrust"}


# --- Placeholder Plotting Functions (Replace with your actual implementations) ---
def plot_metric_over_rounds(df, metric_col, group_by_col, filter_dict, title, output_path, ylabel=None,
                            rename_groups=None, show_std=True, **kwargs):
    if df is None or df.empty: logging.warning(f"No data for plot: {title}"); return
    logging.info(f"Generating line plot: {title} -> {output_path}")
    # Dummy plot logic
    plt.figure()
    # Simplified: just plot first group found after filtering
    query_parts = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in df.columns: continue
            if isinstance(val, list):
                query_parts.append(f"`{col}` in {val}")
            elif isinstance(val, str):
                query_parts.append(f"`{col}` == '{val}'")
            else:
                query_parts.append(f"`{col}` == {val}")
    df_filtered = df.query(" and ".join(query_parts), engine='python') if query_parts else df
    if df_filtered.empty: logging.warning(f"Plotting: No data after filter for {title}"); plt.close(); return

    # Actual plotting logic from analyze_experiment_results.py should be used here
    # For placeholder:
    if not df_filtered.empty and group_by_col in df_filtered.columns and metric_col in df_filtered.columns and 'round_number' in df_filtered.columns:
        for group_name, group_data in df_filtered.groupby(group_by_col):
            mean_data = group_data.groupby('round_number')[metric_col].mean()
            plt.plot(mean_data.index, mean_data.values,
                     label=rename_groups.get(group_name, group_name) if rename_groups else group_name, marker='.')
    plt.title(title)
    plt.xlabel("Round Number")
    plt.ylabel(ylabel if ylabel else METRIC_MAPPING.get(metric_col, metric_col))
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_final_metric_comparison_bar(df, metric_col, main_group_col, filter_dict, title, output_path, hue_col=None,
                                     ylabel=None, rename_groups=None, **kwargs):
    if df is None or df.empty: logging.warning(f"No data for plot: {title}"); return
    logging.info(f"Generating bar plot: {title} -> {output_path}")
    # Dummy plot logic
    plt.figure()
    query_parts = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in df.columns: continue
            if isinstance(val, list):
                query_parts.append(f"`{col}` in {val}")
            elif isinstance(val, str):
                query_parts.append(f"`{col}` == '{val}'")
            else:
                query_parts.append(f"`{col}` == {val}")
    df_filtered = df.query(" and ".join(query_parts), engine='python') if query_parts else df
    if df_filtered.empty: logging.warning(f"Plotting: No data after filter for {title}"); plt.close(); return

    # Actual plotting logic from analyze_experiment_results.py should be used here
    # For placeholder:
    last_round_indices = df_filtered.groupby(['experiment_id', 'run_seed_id'])['round_number'].idxmax()
    df_final_round = df_filtered.loc[last_round_indices]
    if not df_final_round.empty and main_group_col in df_final_round.columns and metric_col in df_final_round.columns:
        sns.barplot(x=main_group_col, y=metric_col, hue=hue_col, data=df_final_round, errorbar='sd')
    plt.title(title)
    plt.xlabel(main_group_col)
    plt.ylabel(ylabel if ylabel else METRIC_MAPPING.get(metric_col, metric_col))
    if hue_col: plt.legend(title=hue_col)
    plt.savefig(output_path)
    plt.close()


# Assume display_gia_instance, load_image_tensor_from_path, prep_for_grid are defined as in analyze_experiment_results.py
# --- End of Placeholder Plotting Functions ---

# --- Main Analysis Script ---
def generate_paper_figures(
        df_rounds_all: pd.DataFrame,
        df_gia_all: Optional[pd.DataFrame],
        # base_data_path_gia: Path, # Path for GIA raw tensor files (passed to display_gia_instance)
        output_dir_paper: Path
):
    """
    Generates figures as outlined for the paper.
    Args:
        df_rounds_all: DataFrame containing all round_results.csv data.
        df_gia_all: DataFrame containing all attack_results.csv (GIA) data.
        base_data_path_gia: Base path to where GIA .pt files are stored, relative to paths in df_gia_all.
        output_dir_paper: Directory to save the generated paper figures.
    """
    output_dir_paper.mkdir(parents=True, exist_ok=True)

    # --- Helper: Define Datasets and Adv Rates for iteration ---
    # These should match what you used in generate_configs.py
    # Or get them from df_rounds_all.dataset_name.unique() etc.
    datasets_to_plot = df_rounds_all['dataset_name'].unique()
    adv_rates_to_plot = sorted(df_rounds_all['adv_rate'].unique())  # e.g., [0.0, 0.1, 0.3]
    # Filter out adv_rate = 0 for attack plots, or handle separately
    poison_adv_rates = [r for r in adv_rates_to_plot if r > 0]
    if not poison_adv_rates and any(ar > 0 for ar in adv_rates_to_plot):  # If only one non-zero adv_rate exists
        poison_adv_rates = [max(adv_rates_to_plot)]

    # == Subsection 1: Baseline Performance Comparison (No Attacks) ==
    sec1_dir = output_dir_paper / "1_baselines"
    sec1_dir.mkdir(parents=True, exist_ok=True)
    df_baselines = df_rounds_all[df_rounds_all['attack_type'] == 'None'].copy()

    if not df_baselines.empty:
        for i, dataset in enumerate(datasets_to_plot):
            # Fig 1a-X: Accuracy/Loss curves per dataset comparing aggregators.
            plot_metric_over_rounds(
                df=df_baselines, metric_col='perf_global_accuracy', group_by_col='aggregation_method',
                filter_dict={'dataset_name': dataset},
                title=f'Baseline Global Accuracy on {dataset}',
                output_path=sec1_dir / f"fig1_{chr(97 + i)}_acc_{dataset}.png",
                rename_groups=AGG_NAME_MAPPING
            )
            plot_metric_over_rounds(
                df=df_baselines, metric_col='perf_global_loss', group_by_col='aggregation_method',
                filter_dict={'dataset_name': dataset},
                title=f'Baseline Global Loss on {dataset}',
                output_path=sec1_dir / f"fig1_{chr(97 + i)}_loss_{dataset}.png",
                rename_groups=AGG_NAME_MAPPING
            )
        # Fig 1z: Final accuracy comparison (bar plots per dataset).
        plot_final_metric_comparison_bar(
            df=df_baselines, metric_col='perf_global_accuracy', main_group_col='aggregation_method',
            hue_col='dataset_name',  # Compare aggregators, with bars for each dataset
            filter_dict=None,  # Use all baseline data
            title='Baseline Final Global Accuracy Comparison',
            output_path=sec1_dir / "fig1_z_final_acc_comparison.png",
            rename_groups=AGG_NAME_MAPPING
        )
    else:
        logging.warning("No baseline data found for Section 1 plots.")

    # == Subsection 2: Robustness Against Backdoor Attacks ==
    sec2_dir = output_dir_paper / "2_backdoor_robustness"
    sec2_dir.mkdir(parents=True, exist_ok=True)
    df_backdoor = df_rounds_all[df_rounds_all['attack_type'] == 'backdoor'].copy()

    if not df_backdoor.empty:
        for i, dataset in enumerate(datasets_to_plot):
            for j, adv_rate in enumerate(poison_adv_rates):  # Assuming you have results for these rates
                filter_crit = {'dataset_name': dataset, 'adv_rate': adv_rate}
                adv_rate_str = f"adv{int(adv_rate * 100)}"
                # Fig 2a-X: ASR/Accuracy curves
                plot_metric_over_rounds(
                    df=df_backdoor, metric_col='perf_global_attack_success_rate', group_by_col='aggregation_method',
                    filter_dict=filter_crit,
                    title=f'Backdoor ASR ({dataset}, {adv_rate_str})',
                    output_path=sec2_dir / f"fig2_{chr(97 + i)}_{chr(97 + j)}_asr_{dataset}_{adv_rate_str}.png",
                    rename_groups=AGG_NAME_MAPPING
                )
                plot_metric_over_rounds(
                    df=df_backdoor, metric_col='perf_global_accuracy', group_by_col='aggregation_method',
                    filter_dict=filter_crit,
                    title=f'Global Accuracy under Backdoor ({dataset}, {adv_rate_str})',
                    output_path=sec2_dir / f"fig2_{chr(97 + i)}_{chr(97 + j)}_acc_{dataset}_{adv_rate_str}.png",
                    rename_groups=AGG_NAME_MAPPING
                )
        # Fig 2y, 2z: Final ASR/Accuracy bar plots (summarizing robustness)
        plot_final_metric_comparison_bar(
            df=df_backdoor, metric_col='perf_global_attack_success_rate', main_group_col='aggregation_method',
            hue_col='dataset_name',  # Or hue_col='adv_rate' if you have multiple
            filter_dict={'adv_rate': poison_adv_rates},  # Show for relevant adv_rates
            title='Final Backdoor ASR Comparison',
            output_path=sec2_dir / "fig2_y_final_asr.png",
            rename_groups=AGG_NAME_MAPPING
        )
        plot_final_metric_comparison_bar(
            df=df_backdoor, metric_col='perf_global_accuracy', main_group_col='aggregation_method',
            hue_col='dataset_name',
            filter_dict={'adv_rate': poison_adv_rates},
            title='Final Global Accuracy under Backdoor Attack',
            output_path=sec2_dir / "fig2_z_final_acc_backdoor.png",
            rename_groups=AGG_NAME_MAPPING
        )
        # (Optional) Plots for selection_fpr / selection_fnr / malicious_selection_rate (TPR).
        if 'selection_rate_info_detection_rate (TPR)' in df_backdoor.columns:
            plot_final_metric_comparison_bar(
                df=df_backdoor, metric_col='selection_rate_info_detection_rate (TPR)',
                main_group_col='aggregation_method',
                hue_col='dataset_name', filter_dict={'adv_rate': poison_adv_rates},
                title='Defense TPR against Backdoor',
                output_path=sec2_dir / "fig2_opt_tpr_backdoor.png",
                rename_groups=AGG_NAME_MAPPING
            )
    else:
        logging.warning("No backdoor attack data found for Section 2 plots.")

    # == Subsection 3: Robustness Against Label Flipping Attacks ==
    sec3_dir = output_dir_paper / "3_labelflip_robustness"
    sec3_dir.mkdir(parents=True, exist_ok=True)
    df_labelflip = df_rounds_all[df_rounds_all['attack_type'] == 'label_flip'].copy()

    if not df_labelflip.empty:
        for i, dataset in enumerate(datasets_to_plot):
            for j, adv_rate in enumerate(poison_adv_rates):
                filter_crit = {'dataset_name': dataset, 'adv_rate': adv_rate}
                adv_rate_str = f"adv{int(adv_rate * 100)}"
                # Fig 3a-X: Accuracy curves
                plot_metric_over_rounds(
                    df=df_labelflip, metric_col='perf_global_accuracy', group_by_col='aggregation_method',
                    filter_dict=filter_crit,
                    title=f'Global Accuracy under Label Flip ({dataset}, {adv_rate_str})',
                    output_path=sec3_dir / f"fig3_{chr(97 + i)}_{chr(97 + j)}_acc_{dataset}_{adv_rate_str}.png",
                    rename_groups=AGG_NAME_MAPPING
                )
        # Fig 3z: Final Accuracy bar plots.
        plot_final_metric_comparison_bar(
            df=df_labelflip, metric_col='perf_global_accuracy', main_group_col='aggregation_method',
            hue_col='dataset_name', filter_dict={'adv_rate': poison_adv_rates},
            title='Final Global Accuracy under Label Flip Attack',
            output_path=sec3_dir / "fig3_z_final_acc_labelflip.png",
            rename_groups=AGG_NAME_MAPPING
        )
        # (Optional) filtering/malicious selection rate plots (TPR).
        if 'selection_rate_info_detection_rate (TPR)' in df_labelflip.columns:
            plot_final_metric_comparison_bar(
                df=df_labelflip, metric_col='selection_rate_info_detection_rate (TPR)',
                main_group_col='aggregation_method',
                hue_col='dataset_name', filter_dict={'adv_rate': poison_adv_rates},
                title='Defense TPR against Label Flip',
                output_path=sec3_dir / "fig3_opt_tpr_labelflip.png",
                rename_groups=AGG_NAME_MAPPING
            )
    else:
        logging.warning("No label flip attack data found for Section 3 plots.")

    # == Subsection 4: Robustness Against Adaptive Mimicry Attacks (Sybil) ==
    sec4_dir = output_dir_paper / "4_sybil_robustness"
    sec4_dir.mkdir(parents=True, exist_ok=True)
    # Assuming Sybil attacks are marked by 'is_sybil' == True
    # And 'attack_type' might be 'None', 'backdoor', or 'label_flip' for the underlying local model behavior
    df_sybil = df_rounds_all[df_rounds_all['is_sybil'] == True].copy()

    if not df_sybil.empty:
        sybil_amp_factors = sorted(df_sybil['sybil_amp_factor'].unique())

        for i, dataset in enumerate(datasets_to_plot):
            for j, amp_factor in enumerate(sybil_amp_factors):
                # Consider Sybil with no underlying poisoning first
                filter_crit_sybil_no_local_atk = {'dataset_name': dataset, 'sybil_amp_factor': amp_factor,
                                                  'attack_type': 'None'}
                amp_factor_str = f"amp{str(amp_factor).replace('.', 'p')}"

                plot_metric_over_rounds(
                    df=df_sybil, metric_col='perf_global_accuracy', group_by_col='aggregation_method',
                    filter_dict=filter_crit_sybil_no_local_atk,
                    title=f'Accuracy under Sybil (No Local Atk, {dataset}, {amp_factor_str})',
                    output_path=sec4_dir / f"fig4_{chr(97 + i)}_{chr(97 + j)}_acc_sybil_no_local_{dataset}_{amp_factor_str}.png",
                    rename_groups=AGG_NAME_MAPPING
                )

                # If Sybil combined with backdoor
                filter_crit_sybil_backdoor = {'dataset_name': dataset, 'sybil_amp_factor': amp_factor,
                                              'attack_type': 'backdoor'}
                if not df_sybil.query(" and ".join(f"`{k}`=='{v}'" if isinstance(v, str) else f"`{k}`=={v}" for k, v in
                                                   filter_crit_sybil_backdoor.items()), engine='python').empty:
                    plot_metric_over_rounds(
                        df=df_sybil, metric_col='perf_global_attack_success_rate', group_by_col='aggregation_method',
                        filter_dict=filter_crit_sybil_backdoor,
                        title=f'ASR under Sybil+Backdoor ({dataset}, {amp_factor_str})',
                        output_path=sec4_dir / f"fig4_{chr(97 + i)}_{chr(97 + j)}_asr_sybil_backdoor_{dataset}_{amp_factor_str}.png",
                        rename_groups=AGG_NAME_MAPPING
                    )
        # Final Accuracy/ASR bar plots for Sybil
        plot_final_metric_comparison_bar(
            df=df_sybil[df_sybil['attack_type'] == 'None'], metric_col='perf_global_accuracy',
            main_group_col='aggregation_method',
            hue_col='dataset_name',  # or hue_col='sybil_amp_factor'
            filter_dict=None,  # Use all sybil_df with no local attack
            title='Final Accuracy under Sybil Attack (No Local Poisoning)',
            output_path=sec4_dir / "fig4_z_final_acc_sybil.png",
            rename_groups=AGG_NAME_MAPPING
        )
    else:
        logging.warning("No Sybil attack data found for Section 4 plots.")

    # == Subsection 5: Impact of Data Distribution and Discovery ==
    sec5_dir = output_dir_paper / "5_data_discovery"
    sec5_dir.mkdir(parents=True, exist_ok=True)
    # Assuming 'discovery_quality' and 'buyer_data_mode' are columns
    df_discovery = df_rounds_all[
        df_rounds_all['data_split_mode'] == 'discovery'].copy()  # Or however you identify these runs

    if not df_discovery.empty:
        discovery_qualities = sorted(df_discovery['discovery_quality'].unique())
        # Fig 5a: Effect of discovery_quality on accuracy for key aggregators.
        plot_final_metric_comparison_bar(
            df=df_discovery, metric_col='perf_global_accuracy', main_group_col='aggregation_method',
            hue_col='discovery_quality',  # This makes sense
            filter_dict={'buyer_data_mode': 'unbiased'},  # Example: focus on unbiased buyer data for this plot
            title='Impact of Discovery Quality on Final Accuracy (Unbiased Buyer)',
            output_path=sec5_dir / "fig5_a_discovery_quality_effect.png",
            rename_groups=AGG_NAME_MAPPING
        )
        # Fig 5b: Effect of buyer_data_mode
        if 'buyer_data_mode' in df_discovery.columns:  # Check if you varied this
            plot_final_metric_comparison_bar(
                df=df_discovery, metric_col='perf_global_accuracy', main_group_col='aggregation_method',
                hue_col='buyer_data_mode',
                filter_dict={'discovery_quality': 0.3},  # Example: fix one quality
                title='Impact of Buyer Data Mode on Final Accuracy (Discovery Quality 0.3)',
                output_path=sec5_dir / "fig5_b_buyer_data_mode_effect.png",
                rename_groups=AGG_NAME_MAPPING
            )
    else:
        logging.warning("No discovery split data found for Section 5 plots.")

    # == Subsection 6: Privacy Leakage Analysis (Gradient Inversion) ==
    sec6_dir = output_dir_paper / "6_privacy_gia"
    sec6_dir.mkdir(parents=True, exist_ok=True)

    if df_gia_all is not None and not df_gia_all.empty:
        # Ensure GIA metrics are numeric
        gia_metric_cols_for_plot = ['metric_psnr', 'metric_ssim', 'metric_lpips', 'metric_label_acc']
        for col in gia_metric_cols_for_plot:
            if col in df_gia_all.columns:
                df_gia_all[col] = pd.to_numeric(df_gia_all[col], errors='coerce')

        # Assuming GIA is run at specific rounds, bar plots of average metrics are common
        # Filter for GIA performed runs (if 'gia_performed' was in df_rounds_all and merged, or check df_gia_all directly)

        # Fig 6z: Final PSNR/SSIM bar plot comparison.
        # Here 'main_group_col' could be 'aggregation_method' if you ran GIA under different defenses
        # or 'dataset_name' if you compare GIA across datasets for a fixed defense (e.g., FedAvg)
        plot_final_metric_comparison_bar(  # This function needs adaptation if using df_gia_all directly
            # as it expects 'round_number' to get final values.
            # For GIA, usually average over all GIA attempts.
            df=df_gia_all, metric_col='metric_psnr', main_group_col='aggregation_method',
            hue_col='dataset_name',
            title='Average GIA PSNR Comparison',
            output_path=sec6_dir / "fig6_z_avg_psnr_gia.png",
            rename_groups=AGG_NAME_MAPPING  # For aggregation_method if used
        )
        plot_final_metric_comparison_bar(
            df=df_gia_all, metric_col='metric_ssim', main_group_col='aggregation_method',
            hue_col='dataset_name',
            title='Average GIA SSIM Comparison',
            output_path=sec6_dir / "fig6_z_avg_ssim_gia.png",
            rename_groups=AGG_NAME_MAPPING
        )
        if 'metric_label_acc' in df_gia_all.columns:
            plot_final_metric_comparison_bar(
                df=df_gia_all, metric_col='metric_label_acc', main_group_col='aggregation_method',
                hue_col='dataset_name',
                title='Average GIA Reconstructed Label Accuracy',
                output_path=sec6_dir / "fig6_opt_label_acc_gia.png",
                rename_groups=AGG_NAME_MAPPING
            )

        # Qualitative Examples for GIA:
        # You need to define base_data_path_gia correctly based on how paths are stored in df_gia_all
        # Example: if df_gia_all has a 'run_save_path_prefix' and 'reconstructed_image_file' is relative to that
        # num_qual_examples = 3
        # if 'metric_psnr' in df_gia_all.columns:
        #     best_psnr_gia = df_gia_all.nlargest(num_qual_examples, 'metric_psnr')
        #     logging.info(f"Generating GIA visualizations for best PSNR ({len(best_psnr_gia)} cases)")
        #     for _, row in best_psnr_gia.iterrows():
        #         run_base_for_files = Path(row['run_save_path_prefix']) # Assuming you added this
        #         # Ensure privacy_attack_path used by GIA is reconstructed or stored
        #         # This is tricky, base_data_path_gia might need to be constructed per row
        #         # or assumed to be relative to a single higher-level path.
        #         # For now, assuming base_data_path_gia passed to this main function is correct root
        #         display_gia_instance(row, base_data_path_gia, sec6_dir / "qualitative_examples")
    else:
        logging.warning("No GIA data found for Section 6 plots.")

    # == Subsection 7: Marketplace-Specific Analysis ==
    sec7_dir = output_dir_paper / "7_marketplace_analysis"
    sec7_dir.mkdir(parents=True, exist_ok=True)
    if df_rounds_all is not None:
        logging.info("\n--- Generating Marketplace-Specific Analysis ---")
        # Example: Number of sellers selected over rounds
        if 'num_sellers_selected' in df_rounds_all.columns:
            for dataset in datasets_to_plot:
                plot_metric_over_rounds(
                    df=df_rounds_all, metric_col='num_sellers_selected', group_by_col='aggregation_method',
                    filter_dict={'dataset_name': dataset, 'attack_type': 'None'},  # Example: for baselines
                    title=f'Number of Selected Sellers ({dataset}, Baseline)',
                    output_path=sec7_dir / f"fig7_num_selected_{dataset}_baseline.png",
                    rename_groups=AGG_NAME_MAPPING
                )
        # Add more plots for diversity, Gini, etc. This requires more specific data processing.
        # For Gini of selection frequencies:
        # 1. For each experiment_id and run_seed_id:
        # 2. Count how many times each seller_id was in 'selected_sellers' list across all rounds.
        # 3. Calculate Gini coefficient for these counts.
        # 4. Plot average Gini per aggregation_method. (This is more involved)

    logging.info(f"Paper figure generation complete. Check folder: {output_dir_paper}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures from aggregated experiment results.")
    parser.add_argument(
        "base_results_dir", type=Path,
        help="Base directory where all experiment results are stored (e.g., './experiment_results_revised')."
    )
    parser.add_argument(
        "--output_paper_figs_dir", type=Path, default=Path("./paper_figures"),
        help="Directory to save the generated paper figures."
    )
    # parser.add_argument( # If GIA files are not directly under base_results_dir
    #     "--gia_data_root", type=Path,
    #     help="Root path for GIA tensor files if not easily inferred from base_results_dir and CSV paths."
    # )
    args = parser.parse_args()

    # 1. Load and aggregate all data
    # This function needs to be robust from analyze_experiment_results.py
    # For now, assume it's available and works:
    # from analyze_experiment_results import load_all_results
    df_rounds, df_gia = load_all_results(args.base_results_dir)

    if df_rounds is None and df_gia is None:
        logging.error("No data loaded from results directory. Exiting.")
    else:
        generate_paper_figures(df_rounds, df_gia, args.output_paper_figs_dir)
