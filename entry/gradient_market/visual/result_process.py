# analyze_fl_results_full_with_mal_selection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
import warnings
import logging
import re # Import regular expressions for parsing adv_rate
from typing import Dict, List, Optional, Tuple, Any, Union

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Plotting Style ---
sns.set_theme(style="whitegrid", palette="viridis")


# ======================================================================
# == Helper Functions (load_all_results, safe_literal_eval) ==
# ======================================================================
# --- (Keep implementations from previous answer) ---
def safe_literal_eval(val): # Condensed version
    if pd.isna(val) or val is None: return None
    if isinstance(val, (list, dict)): return val
    try:
        if isinstance(val, str) and (('{' in val and '}' in val) or ('[' in val and ']' in val)): return ast.literal_eval(val)
        else:
            try: return float(val)
            except (ValueError, TypeError): pass
            return val
    except: return None

def load_all_results(base_dir: str, csv_filename: str = "round_results.csv") -> Dict[str, List[pd.DataFrame]]:
    base_path = Path(base_dir); all_results = {};
    if not base_path.is_dir(): logging.error(f"Base directory '{base_dir}' not found."); return all_results
    logging.info(f"Loading results from: {base_path}")
    for experiment_path in sorted(base_path.iterdir()):
        if experiment_path.is_dir():
            experiment_name = experiment_path.name; run_dfs = []
            logging.info(f"  Loading Experiment: {experiment_name}")
            for run_path in sorted(experiment_path.glob("run_*")):
                if run_path.is_dir():
                    csv_file = run_path / csv_filename
                    if csv_file.is_file():
                        try: df = pd.read_csv(csv_file);
                        except Exception as e: logging.error(f"    Error loading {csv_file}: {e}"); continue
                        if not df.empty: df['run_id'] = run_path.name; df['experiment_setup'] = experiment_name; run_dfs.append(df)
            if run_dfs: all_results[experiment_name] = run_dfs
    if not all_results: logging.error(f"No experiments loaded from {base_dir}.")
    return all_results

# ======================================================================
# == Preprocessing Function (Modified) ==
# ======================================================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug(f"Preprocessing DataFrame with columns: {df.columns.tolist()}")
    eval_columns = ["selected_sellers", "outlier_sellers", "perf_global", "perf_local", "selection_rate_info", "defense_metrics", "selection_scores", "gradient_inversion_log", "selected_seller_indices", "outlier_seller_indices", "seller_ids_all"] # Added list cols
    numeric_list_cols = ["selected_sellers", "outlier_sellers", "selected_seller_indices", "outlier_seller_indices"] # Treat ids as maybe non-numeric if strings
    dict_cols = ["perf_global", "perf_local", "selection_rate_info", "defense_metrics", "selection_scores", "gradient_inversion_log"]
    str_list_cols = ["seller_ids_all"] # List of strings

    # Apply literal eval first
    for col in df.columns:
        if col in eval_columns:
            logging.debug(f"  Applying safe_literal_eval to '{col}'")
            df[col] = df[col].apply(safe_literal_eval)

    # Ensure correct empty types and list contents after eval
    for col in df.columns:
         if col in eval_columns:
            if col in numeric_list_cols: df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            elif col in dict_cols: df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
            elif col in str_list_cols: df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])


    # --- Extract nested performance metrics ---
    if 'perf_global' in df.columns:
        df['global_acc'] = df['perf_global'].apply(lambda x: x.get('accuracy') if isinstance(x, dict) else np.nan)
        df['global_loss'] = df['perf_global'].apply(lambda x: x.get('loss') if isinstance(x, dict) else np.nan)
        df['global_asr'] = df['perf_global'].apply(lambda x: x.get('attack_success_rate') if isinstance(x, dict) else np.nan)

    # --- Extract Selection Rate Info ---
    if 'selection_rate_info' in df.columns:
        df['selection_fpr'] = df['selection_rate_info'].apply(lambda x: x.get('false_positive_rate', x.get('FPR')) if isinstance(x, dict) else np.nan)
        df['selection_fnr'] = df['selection_rate_info'].apply(lambda x: x.get('false_negative_rate', x.get('FNR')) if isinstance(x, dict) else np.nan)

    # --- Convert numeric columns ---
    numeric_cols_to_convert = ['round_number', 'round_duration_sec', 'num_sellers_selected', 'avg_client_train_loss', 'comm_up_avg_bytes', 'comm_up_max_bytes', 'comm_down_bytes', 'client_time_avg_ms', 'client_time_max_ms', 'server_agg_time_ms', 'attack_victim_idx_targeted', 'attack_victim_score']
    for col in numeric_cols_to_convert:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'round_number' in df.columns:
        df = df.dropna(subset=['round_number'])
        if not df.empty: df['round_number'] = df['round_number'].astype(int)

    # --- Parse experiment setup for easier filtering ---
    adv_rate = np.nan # Default
    num_sellers = 10 # Default, might need adjustment or parsing
    if 'experiment_setup' in df.columns and isinstance(df['experiment_setup'].iloc[0], str):
        exp_name = df['experiment_setup'].iloc[0] # Use first row's name
        df['exp_type'] = exp_name.split('_')[0] if len(exp_name.split('_')) > 0 else 'unknown'
        df['exp_dataset'] = exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else 'unknown'
        df['exp_aggregator'] = exp_name.split('_')[2] if len(exp_name.split('_')) > 2 else 'unknown'
        # Extract adversary rate (example using regex)
        match = re.search(r'_adv(\d+)pct', exp_name)
        if match:
             adv_rate = int(match.group(1)) / 100.0
        df['exp_adv_rate'] = adv_rate # Store parsed rate
        # Add more parsing if needed (e.g., quality 'q0.7')
    else:
        df['exp_type'] = 'unknown'
        df['exp_adv_rate'] = adv_rate

    # --- *** NEW: Calculate Malicious Selection Rate *** ---
    # Requires 'selected_seller_indices' and knowledge of which indices are malicious
    df['malicious_selection_rate'] = np.nan # Initialize column

    # Apply row-wise calculation
    def calculate_mal_select_rate(row):
        selected_indices = row.get('selected_seller_indices')
        exp_adv_rate = row.get('exp_adv_rate')
        seller_ids = row.get('seller_ids_all')

        # Basic checks
        if not isinstance(selected_indices, list) or pd.isna(exp_adv_rate) or exp_adv_rate == 0 or not isinstance(seller_ids, list) or not seller_ids:
            return 0.0 # Return 0 if no adversaries or no selection info

        num_total_sellers = len(seller_ids)
        num_malicious = int(round(num_total_sellers * exp_adv_rate)) # Calculate expected number of malicious

        # ** ASSUMPTION: Malicious clients are the first 'num_malicious' indices **
        malicious_indices_set = set(range(num_malicious))

        if not selected_indices: # Handle case where no sellers were selected
             return 0.0

        num_malicious_selected = sum(1 for idx in selected_indices if idx in malicious_indices_set)
        num_total_selected = len(selected_indices)

        return num_malicious_selected / num_total_selected if num_total_selected > 0 else 0.0

    if 'selected_seller_indices' in df.columns and 'exp_adv_rate' in df.columns and 'seller_ids_all' in df.columns:
         df['malicious_selection_rate'] = df.apply(calculate_mal_select_rate, axis=1)
    else:
        logging.warning("Could not calculate malicious_selection_rate. Required columns: 'selected_seller_indices', 'exp_adv_rate', 'seller_ids_all'")


    logging.debug("Preprocessing finished.")
    return df

# ======================================================================
# == Plotting Functions (plot_metric_comparison, plot_final_round_comparison) ==
# ======================================================================
# --- (Keep implementations from previous answer) ---
def plot_metric_comparison(results_dict, metric_column, title=None, xlabel="Communication Round", ylabel=None, confidence_interval='sd', use_seaborn=True, save_path=None, figsize=(10, 6), legend_loc='best', ylim=None, filter_metric_nan=True, **kwargs):
    # ... (implementation from previous answer) ...
    if not results_dict: print("Error: No results data for plotting."); return
    all_runs_list = []; valid_experiments = []
    for experiment_name, run_dfs in results_dict.items():
        if not run_dfs: continue
        first_df_processed = run_dfs[0]
        if metric_column not in first_df_processed.columns: logging.warning(f"Metric '{metric_column}' not found for exp '{experiment_name}'. Skipping."); continue
        if filter_metric_nan and first_df_processed[metric_column].isnull().all(): logging.warning(f"Metric '{metric_column}' is all NaN for exp '{experiment_name}'. Skipping."); continue
        combined_exp_df = pd.concat(run_dfs, ignore_index=True); all_runs_list.append(combined_exp_df); valid_experiments.append(experiment_name)
    if not all_runs_list: logging.error(f"No valid data found for metric '{metric_column}' across experiments."); return
    all_runs_df = pd.concat(all_runs_list, ignore_index=True)
    if filter_metric_nan: original_rows = len(all_runs_df); all_runs_df = all_runs_df.dropna(subset=[metric_column]);
    if all_runs_df.empty: logging.error(f"No non-NaN data left for metric '{metric_column}' after filtering."); return
    plt.figure(figsize=figsize); plot_ylabel = ylabel if ylabel is not None else metric_column.replace('_', ' ').title(); plot_title = title if title is not None else f"{plot_ylabel} vs. {xlabel}"
    palette = sns.color_palette("viridis", n_colors=len(valid_experiments))
    if use_seaborn:
        try: sns.lineplot(data=all_runs_df, x="round_number", y=metric_column, hue="experiment_setup", hue_order=sorted(valid_experiments), palette=palette, errorbar=confidence_interval, linewidth=2.5, alpha=0.9, **kwargs); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(title='Experiment Setup', loc=legend_loc, fontsize=9)
        except Exception as e: logging.error(f"Error during seaborn plot: {e}"); plt.close(); return
    else: logging.error("Manual matplotlib plotting not fully implemented here, using seaborn is recommended."); plt.close(); return
    plt.title(plot_title, fontsize=16, pad=15); plt.xlabel(xlabel, fontsize=13); plt.ylabel(plot_ylabel, fontsize=13);
    if ylim: plt.ylim(ylim)
    plt.tight_layout()
    if save_path: save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True);
    try: plt.savefig(save_path, dpi=300); logging.info(f"Plot saved: {save_path}")
    except Exception as e: logging.error(f"Error saving plot {save_path}: {e}")
    plt.close()

def plot_final_round_comparison(results_dict, metric_column, title=None, ylabel=None, higher_is_better=True, save_path=None, figsize=(10, 6), **kwargs):
    # ... (implementation from previous answer) ...
    final_metrics = {}; final_errors = {}
    for exp_name, run_dfs in results_dict.items():
        if not run_dfs: continue; first_df = run_dfs[0]
        if metric_column not in first_df.columns: logging.warning(f"Metric '{metric_column}' missing for {exp_name}, skipping final comparison."); continue
        final_vals_for_exp = []
        for df in run_dfs:
            df = df.dropna(subset=[metric_column, 'round_number'])
            if not df.empty: final_round_val = df.loc[df['round_number'].idxmax(), metric_column]
            if not pd.isna(final_round_val): final_vals_for_exp.append(final_round_val)
        if final_vals_for_exp: final_metrics[exp_name] = np.mean(final_vals_for_exp); final_errors[exp_name] = np.std(final_vals_for_exp) / np.sqrt(len(final_vals_for_exp)) if len(final_vals_for_exp) > 0 else 0
    if not final_metrics: logging.error(f"No final round data found for metric '{metric_column}'."); return
    exp_names = sorted(list(final_metrics.keys())); mean_values = [final_metrics[name] for name in exp_names]; std_errors = [final_errors[name] for name in exp_names]
    plt.figure(figsize=figsize); palette = sns.color_palette("viridis", n_colors=len(exp_names))
    best_val = max(mean_values) if higher_is_better else min(mean_values)
    colors = [palette[exp_names.index(name)] if final_metrics[name] != best_val else sns.color_palette("Reds", 1)[0] for name in exp_names] # Consistent color mapping
    bars = plt.bar(exp_names, mean_values, yerr=std_errors, capsize=5, color=colors, **kwargs)
    plot_ylabel = ylabel if ylabel is not None else metric_column.replace('_', ' ').title(); plot_title = title if title is not None else f"Comparison of Final Round {plot_ylabel}"
    plt.title(plot_title, fontsize=16, pad=15); plt.ylabel(plot_ylabel, fontsize=13); plt.xticks(rotation=30, ha='right', fontsize=10); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=9, fontweight='bold')
    if save_path: save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True);
    try: plt.savefig(save_path, dpi=300); logging.info(f"Plot saved: {save_path}")
    except Exception as e: logging.error(f"Error saving plot {save_path}: {e}")
    plt.close()


# ======================================================================
# == Main Analysis Execution ==
# ======================================================================
if __name__ == "__main__":
    # --- Configuration ---
    RESULTS_BASE_DIR = "./experiment_results" # <<<--- CHANGE THIS PATH!!!
    OUTPUT_PLOT_DIR = Path("./analysis_plots_final_v2")
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load & Preprocess ---
    all_results_raw = load_all_results(RESULTS_BASE_DIR)
    all_results_processed = {}
    if all_results_raw:
        for exp_name, run_dfs_raw in all_results_raw.items():
            processed_runs = []
            logging.info(f"Preprocessing experiment: {exp_name}")
            for i, df_raw in enumerate(run_dfs_raw):
                try: df_processed = preprocess_data(df_raw.copy());
                except Exception as e: logging.error(f"  Error preprocessing run {i} for {exp_name}: {e}", exc_info=True); continue
                if not df_processed.empty: processed_runs.append(df_processed)
            if processed_runs: all_results_processed[exp_name] = processed_runs
    else: exit("No results loaded. Exiting.")

    if not all_results_processed: exit("No data survived preprocessing. Exiting.")

    logging.info("\n--- Starting Analysis & Plotting ---")

    # === Define Experiment Groups for Analysis ===
    exp_groups = {
        "Baseline (CIFAR)": [k for k in all_results_processed if k.startswith('baseline_cifar_')],
        "Baseline (FMNIST)": [k for k in all_results_processed if k.startswith('baseline_fmnist_')],
        "Backdoor (CIFAR, 10% Adv)": [k for k in all_results_processed if k.startswith('backdoor_cifar_adv10pct')],
        "Backdoor (CIFAR, 30% Adv)": [k for k in all_results_processed if k.startswith('backdoor_cifar_adv30pct')],
        "LabelFlip (FMNIST, 10% Adv)": [k for k in all_results_processed if k.startswith('label_flip_fmnist_adv10pct')],
        "LabelFlip (FMNIST, 30% Adv)": [k for k in all_results_processed if k.startswith('label_flip_fmnist_adv30pct')],
        "Sybil (CIFAR, 30% Adv)": [k for k in all_results_processed if k.startswith('sybil_cifar_adv30pct')], # Example specific rate
        "Discovery (CIFAR, Q=0.7)": [k for k in all_results_processed if k.startswith('discovery_cifar_') and '_q0.7_' in k],
        "Privacy (CIFAR, Q=0.7)": [k for k in all_results_processed if k.startswith('privacy_discovery_cifar_') and '_q0.7_' in k], # Check your actual naming
        "Privacy (FMNIST, Q=0.3)": [k for k in all_results_processed if k.startswith('privacy_discovery_fmnist_') and '_q0.3_' in k], # Check your actual naming
    }

    # --- Generate Plots for Each Group ---
    for group_name, exp_keys in exp_groups.items():
        logging.info(f"\n--- Analyzing Group: {group_name} ---")
        group_results = {k: all_results_processed[k] for k in exp_keys if k in all_results_processed}
        if not group_results:
            logging.warning(f"No results found for group '{group_name}'. Skipping plots.")
            continue

        group_plot_dir = OUTPUT_PLOT_DIR / group_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('=', '')
        group_plot_dir.mkdir(parents=True, exist_ok=True)

        # Common Performance Plots
        plot_metric_comparison(group_results, "global_acc", title=f"{group_name} - Accuracy", ylabel="Accuracy", ylim=(0, 1.0), save_path=group_plot_dir / "global_accuracy.png")
        plot_metric_comparison(group_results, "global_loss", title=f"{group_name} - Loss", ylabel="Loss", save_path=group_plot_dir / "global_loss.png")
        plot_final_round_comparison(group_results, "global_acc", title=f"{group_name} - Final Accuracy", ylabel="Accuracy", save_path=group_plot_dir / "final_accuracy_bar.png")

        # Attack Specific Plots (ASR) - Plot only if the metric likely exists for the group
        if "Backdoor" in group_name or "Sybil" in group_name:
            plot_metric_comparison(group_results, "global_asr", title=f"{group_name} - ASR", ylabel="ASR", ylim=(-0.05, 1.05), save_path=group_plot_dir / "global_asr.png")
            plot_final_round_comparison(group_results, "global_asr", title=f"{group_name} - Final ASR", ylabel="ASR", higher_is_better=False, save_path=group_plot_dir / "final_asr_bar.png")

        # Selection Plots (Always potentially relevant)
        plot_metric_comparison(group_results, "num_sellers_selected", title=f"{group_name} - # Selected Sellers", ylabel="# Selected", save_path=group_plot_dir / "num_selected.png")
        plot_metric_comparison(group_results, "selection_fpr", title=f"{group_name} - Selection FPR", ylabel="FPR", ylim=(-0.05, 1.05), save_path=group_plot_dir / "selection_fpr.png")
        plot_metric_comparison(group_results, "selection_fnr", title=f"{group_name} - Selection FNR", ylabel="FNR", ylim=(-0.05, 1.05), save_path=group_plot_dir / "selection_fnr.png")

        # *** NEW: Malicious Selection Rate Plot ***
        # Plot only if the group involves adversaries (adv rate > 0 likely)
        if "Backdoor" in group_name or "LabelFlip" in group_name or "Sybil" in group_name:
             # Check if the column was successfully created
             if 'malicious_selection_rate' in list(group_results.values())[0][0].columns:
                 plot_metric_comparison(group_results, "malicious_selection_rate",
                                        title=f"{group_name} - Malicious Selection Rate",
                                        ylabel="Fraction of Selected who are Malicious", ylim=(-0.05, 1.05),
                                        save_path=group_plot_dir / "malicious_selection_rate.png")
                 plot_final_round_comparison(group_results, "malicious_selection_rate",
                                            title=f"{group_name} - Final Malicious Selection Rate",
                                            ylabel="Malicious Rate among Selected", higher_is_better=False, # Lower is better
                                            save_path=group_plot_dir / "final_malicious_selection_rate_bar.png")
             else:
                  logging.warning(f"Column 'malicious_selection_rate' not found for group '{group_name}'. Skipping plot.")


        # Privacy Specific Plots - Plot only if metrics likely exist
        if "Privacy" in group_name:
             # Check if the columns were successfully created
             first_df = list(group_results.values())[0][0] if group_results else None
             if first_df is not None and 'attack_psnr' in first_df.columns:
                 plot_metric_comparison(group_results, "attack_psnr", title=f"{group_name} - Attack PSNR", ylabel="PSNR (dB)", filter_metric_nan=True, save_path=group_plot_dir / "attack_psnr.png")
                 plot_metric_comparison(group_results, "attack_ssim", title=f"{group_name} - Attack SSIM", ylabel="SSIM", ylim=(0,1.05), filter_metric_nan=True, save_path=group_plot_dir / "attack_ssim.png")
                 plot_final_round_comparison(group_results, "attack_psnr", title=f"{group_name} - Final Attack PSNR", ylabel="PSNR (dB)", higher_is_better=True, save_path=group_plot_dir / "final_attack_psnr_bar.png") # Higher PSNR -> worse for attacker
             else:
                  logging.warning(f"Attack metric columns (e.g., 'attack_psnr') not found for group '{group_name}'. Skipping privacy plots.")


    logging.info("\n--- Analysis Script Finished ---")
    logging.info(f"Plots saved in subdirectories under: {OUTPUT_PLOT_DIR}")