import glob
import json
import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_distribution_similarity(buyer_distribution, seller_distribution):
    # Ensure distributions are valid (non-empty and contain numbers)
    if not buyer_distribution or not seller_distribution:
        return 0.0
    try:
        buyer_dist_array = np.array(list(buyer_distribution.values()), dtype=float)
        seller_dist_array = np.array(list(seller_distribution.values()), dtype=float)

        # Handle potential zero vectors to avoid division by zero
        norm_buyer = np.linalg.norm(buyer_dist_array)
        norm_seller = np.linalg.norm(seller_dist_array)

        if norm_buyer == 0 or norm_seller == 0:
            return 0.0  # Or handle as appropriate, e.g., if one is zero and other isn't, similarity is 0

        similarity = np.dot(buyer_dist_array, seller_dist_array) / (norm_buyer * norm_seller)
        return similarity
    except Exception as e:
        print(f"Error in calculate_distribution_similarity: {e}")
        print(f"Buyer dist: {buyer_distribution}, Seller dist: {seller_distribution}")
        return 0.0  # Return a default value or raise


def calculate_gini(payments):
    """Calculate the Gini coefficient of a numpy array."""
    # ensure payments is a numpy array
    payments = np.asarray(payments, dtype=float)
    # Values must be non-negative
    if np.any(payments < 0):
        # Or handle as an error, Gini is typically for non-negative quantities like income
        payments[payments < 0] = 0  # Cap at zero for simplicity
        # raise ValueError("Payments Gini coefficient requires non-negative values.")
    # Array must not be empty
    if payments.size == 0:
        return 0.0  # Or handle as an error / NaN
    # Values cannot all be zero
    if not np.any(payments):
        return 0.0  # Perfect equality if all payments are zero

    # Sort payments in ascending order
    sorted_payments = np.sort(payments)
    n = len(payments)
    cum_payments = np.cumsum(sorted_payments, dtype=float)
    # Gini coefficient
    gini = (n + 1 - 2 * np.sum(cum_payments) / cum_payments[-1]) / n
    return gini


def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run,
                              target_accuracy_for_coc=0.8):
    """
    Process a single experiment file and extract metrics, incorporating data distribution similarity and payment simulation.
    """
    try:
        experiment_data = torch.load(file_path, map_location='cpu', weights_only=False)
        data_stats = load_json(data_statistics_path)

        buyer_distribution = data_stats['buyer_stats']['class_distribution']
        seller_distributions = data_stats['seller_stats']  # Dict of seller_id_str -> stats

        if not experiment_data:
            print(f"Warning: No round records found in {file_path}")
            return [], {}

        processed_data = []
        num_total_sellers = market_params["N_CLIENTS"]  # Get total number of sellers from market_params
        num_adversaries = int(num_total_sellers * adv_rate)

        # --- Payment/Incentive Simulation Variables ---
        total_payments_per_seller = {str(i): 0 for i in range(num_total_sellers)}  # Store total payment for each seller
        cost_of_convergence = None
        target_accuracy_reached_round = -1
        cumulative_cost_for_coc = 0
        # ---
        # For "No Attack" runs, we'll store selection counts for different hypothetical adversary group sizes
        # Key: hypothetical_adv_rate_for_baseline (e.g., 0.1, 0.2), Value: list of per-round selection rates for that group
        baseline_designated_group_selection_rates_this_run = {}
        hypothetical_adv_rates_for_baselines = [0.1, 0.2, 0.3, 0.4]
        baseline_designated_group_selection_rates_summary_collector = {}
        run_attack_method = attack_params.get('ATTACK_METHOD', 'None')  # Or 'single', etc.
        for i, record in enumerate(experiment_data):
            round_num = record.get('round_number', i)

            selected_clients = record.get("used_sellers", [])  # These are seller IDs (strings)

            adversary_selections = [cid for cid in selected_clients if int(cid) < num_adversaries]
            benign_selections = [cid for cid in selected_clients if int(cid) >= num_adversaries]

            # --- Payment Calculation for the current round ---
            cost_per_round = len(selected_clients)  # Since payment_per_selected_gradient is 1
            for seller_id in selected_clients:
                if str(seller_id) in total_payments_per_seller:  # Ensure seller_id is a string for dict key
                    total_payments_per_seller[str(seller_id)] += 1  # Payment of 1
                else:
                    # This case should ideally not happen if total_payments_per_seller is initialized correctly
                    print(f"Warning: Selected seller_id {seller_id} not in payment tracking. Initializing.")
                    total_payments_per_seller[str(seller_id)] = 1

            selected_clients_int = [int(cid) for cid in selected_clients]  # Ensure integer IDs if not already
            benign_selected_in_round = [cid_str for cid_str in selected_clients if int(cid_str) >= num_adversaries]
            malicious_selected_in_round = [cid_str for cid_str in selected_clients if int(cid_str) < num_adversaries]

            # 2. Calculate Benign Seller Selection Rate for the round
            num_total_benign_sellers = num_total_sellers - num_adversaries
            if num_total_benign_sellers > 0 and selected_clients:  # Avoid division by zero
                round_benign_selection_rate = len(benign_selected_in_round) / num_total_benign_sellers
            else:
                round_benign_selection_rate = 0.0 if selected_clients else np.nan  # Or just 0.0

            current_attack_method = attack_params.get('ATTACK_METHOD', 'None')  # Get from params like 'single', 'None'

            malicious_selected_in_round_actual = []
            benign_selected_in_round_actual = []
            if current_attack_method != 'None' and current_attack_method != 'No Attack':  # If an attack is active
                for cid_str in selected_clients:
                    if int(cid_str) < num_adversaries:
                        malicious_selected_in_round_actual.append(cid_str)
                    else:
                        benign_selected_in_round_actual.append(cid_str)
            else:  # No active attack in this run
                benign_selected_in_round_actual = list(selected_clients)

            # --- NEW: Calculate Baseline Selection Rates for Designated Groups (if this is a "No Attack" run) ---
            if current_attack_method == 'None' or current_attack_method == 'No Attack':
                for hypo_adv_rate in hypothetical_adv_rates_for_baselines:
                    num_hypo_designated_malicious = int(num_total_sellers * hypo_adv_rate)
                    if num_hypo_designated_malicious == 0: continue

                    selected_from_hypo_group_count = 0
                    for cid_str in selected_clients:
                        if int(cid_str) < num_hypo_designated_malicious:
                            selected_from_hypo_group_count += 1

                    # Rate: count selected from group / size of group
                    rate_for_hypo_group_this_round = selected_from_hypo_group_count / len(
                        selected_clients) if selected_clients else 0.0

                    hypo_adv_rate_key = f"{hypo_adv_rate:.1f}"  # e.g., "0.1"
                    if hypo_adv_rate_key not in baseline_designated_group_selection_rates_this_run:
                        baseline_designated_group_selection_rates_this_run[hypo_adv_rate_key] = []
                    baseline_designated_group_selection_rates_this_run[hypo_adv_rate_key].append(
                        rate_for_hypo_group_this_round)

            # 3. Calculate Gini Coefficient for payments ONLY to BENIGN sellers in this round
            # This assumes you have payment information per round or can infer it.
            # If payment is uniform (e.g., 1 per selected seller):
            payments_to_benign_this_round = {seller_id_str: 0 for seller_id_str in
                                             map(str, range(num_adversaries, num_total_sellers))}
            for seller_id_str in benign_selected_in_round:
                payments_to_benign_this_round[seller_id_str] = 1  # Or actual payment value

            all_benign_payments_array = np.array(list(payments_to_benign_this_round.values()))
            round_benign_gini_coefficient = calculate_gini(all_benign_payments_array)  # Use your existing Gini function
            round_data = {'run': cur_run, 'round': round_num, **attack_params, **market_params,
                          'n_selected_clients': len(selected_clients), 'selected_clients': selected_clients,
                          'adversary_selection_rate': len(adversary_selections) / len(
                              selected_clients) if selected_clients else 0,
                          'benign_selection_rate': len(benign_selections) / len(
                              selected_clients) if selected_clients else 0, 'cost_per_round': cost_per_round,
                          'benign_selection_rate_in_round': round_benign_selection_rate,
                          'benign_gini_coefficient_in_round': round_benign_gini_coefficient,
                          }

            for hypo_adv_rate in hypothetical_adv_rates_for_baselines:
                hypo_adv_rate_key_str = f"{hypo_adv_rate:.1f}"
                round_data_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate_key_str}_ROUND'  # New key for per-round log

                if run_attack_method == 'None' or run_attack_method == 'No Attack':
                    num_hypo_designated_malicious = int(num_total_sellers * hypo_adv_rate)
                    if num_hypo_designated_malicious == 0:
                        round_data[round_data_key] = 0.0  # Or np.nan if preferred for no group
                        # Also collect for summary (will be averaged later)
                        if hypo_adv_rate_key_str not in baseline_designated_group_selection_rates_summary_collector:
                            baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str] = []
                        baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str].append(0.0)
                        continue

                    selected_from_hypo_group_count = 0
                    for cid_str in selected_clients:
                        if int(cid_str) < num_hypo_designated_malicious:
                            selected_from_hypo_group_count += 1

                    rate_for_hypo_group_this_round = selected_from_hypo_group_count / len(
                        selected_clients) if selected_clients else 0.0
                    round_data[round_data_key] = rate_for_hypo_group_this_round

                    # Collect for summary average calculation
                    if hypo_adv_rate_key_str not in baseline_designated_group_selection_rates_summary_collector:
                        baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str] = []
                    baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str].append(
                        rate_for_hypo_group_this_round)
                else:  # If it's an ATTACK run, these per-round baseline metrics are not applicable
                    round_data[round_data_key] = np.nan  # Store NaN in the per-round log
            # Calculate distribution similarities

            similarities = []
            for cid_ in selected_clients:  # cid from selected_clients is already a string
                cid_str = str(cid_)
                if cid_str in seller_distributions:
                    seller_dist = seller_distributions[cid_str]['class_distribution']
                    similarities.append(calculate_distribution_similarity(buyer_distribution, seller_dist))
                else:
                    print(f"Warning: Seller {cid_str} not found in data_statistics for similarity calculation.")
                    print(seller_distributions.keys())
                    similarities.append(0)  # Or handle as NaN or skip

            round_data['avg_selected_data_distribution_similarity'] = np.mean(similarities) if similarities else 0

            un_selected_similarities = []
            all_seller_ids_str = [str(k) for k in
                                  range(num_total_sellers)]  # Generate all possible seller IDs as strings
            unselected_seller_ids_str = [sid for sid in all_seller_ids_str if sid not in selected_clients]

            for cid_str in unselected_seller_ids_str:
                if cid_str in seller_distributions:
                    seller_dist = seller_distributions[cid_str]['class_distribution']
                    un_selected_similarities.append(calculate_distribution_similarity(buyer_distribution, seller_dist))
                else:
                    # This might happen if N_CLIENTS in market_params is larger than actual sellers in data_stats
                    # Or if seller IDs are not contiguous from 0.
                    # print(f"Warning: Unselected seller {cid_str} not found in data_statistics for similarity.")
                    pass  # Or append a default value if needed

            round_data['avg_unselected_data_distribution_similarity'] = np.mean(
                un_selected_similarities) if un_selected_similarities else 0

            final_perf = record.get('final_perf_global', {})
            current_accuracy = final_perf.get('acc')
            round_data['main_acc'] = current_accuracy
            round_data['main_loss'] = final_perf.get('loss')

            # --- Cost of Convergence (CoC) Calculation ---
            if current_accuracy is not None and cost_of_convergence is None:
                cumulative_cost_for_coc += cost_per_round
                if current_accuracy >= target_accuracy_for_coc:
                    cost_of_convergence = cumulative_cost_for_coc
                    target_accuracy_reached_round = round_num
            elif cost_of_convergence is None:  # If accuracy is None but CoC not yet met
                cumulative_cost_for_coc += cost_per_round

            poison_metrics = record.get('extra_info', {}).get('poison_metrics', {})
            round_data.update({
                'clean_acc': poison_metrics.get('clean_accuracy'),
                'triggered_acc': poison_metrics.get('triggered_accuracy'),
                'asr': poison_metrics.get('attack_success_rate')
            })

            processed_data.append(round_data)

        sorted_records = sorted(processed_data, key=lambda x: x['round'])

        if sorted_records:
            asr_values = [r.get('asr') or 0 for r in sorted_records]
            final_record = sorted_records[-1]

            # --- Gini Coefficient Calculation ---
            # Ensure all potential sellers are included, even if they received 0 payment
            all_seller_payments = [total_payments_per_seller.get(str(i), 0) for i in range(num_total_sellers)]
            payment_gini_coefficient = calculate_gini(np.array(all_seller_payments))

            # If target accuracy was never reached, CoC remains None or could be set to total cost
            final_cumulative_cost = sum(r['cost_per_round'] for r in sorted_records)
            if cost_of_convergence is None:
                print(
                    f"Warning: Target accuracy {target_accuracy_for_coc} for CoC not reached in {file_path}. CoC will be total cost or NaN.")
                # Option 1: Set CoC to total cost if not met (indicates high cost)
                # cost_of_convergence = final_cumulative_cost
                # Option 2: Leave as None or set to a specific indicator like np.nan
                cost_of_convergence = np.nan  # Or final_cumulative_cost if you prefer that interpretation

            summary = {"run": cur_run, **market_params, **attack_params,
                       'MAX_ASR': max(asr_values) if asr_values else 0, 'FINAL_ASR': final_record.get('asr'),
                       'FINAL_MAIN_ACC': final_record.get('main_acc'), 'FINAL_CLEAN_ACC': final_record.get('clean_acc'),
                       'FINAL_TRIGGERED_ACC': final_record.get('triggered_acc'),
                       'AVG_SELECTED_DISTRIBUTION_SIMILARITY': np.mean(
                           [r['avg_selected_data_distribution_similarity'] for r in sorted_records if
                            'avg_selected_data_distribution_similarity' in r]),
                       'AVG_UNSELECTED_DISTRIBUTION_SIMILARITY': np.mean(
                           [r['avg_unselected_data_distribution_similarity'] for r in sorted_records if
                            'avg_unselected_data_distribution_similarity' in r]),
                       'AVG_ADVERSARY_SELECTION_RATE': np.mean(
                           [r['adversary_selection_rate'] for r in sorted_records if 'adversary_selection_rate' in r]),
                       'AVG_BENIGN_SELECTION_RATE': np.mean(
                           [r['benign_selection_rate'] for r in sorted_records if 'benign_selection_rate' in r]),
                       'AVG_COST_PER_ROUND': np.mean(
                           [r['cost_per_round'] for r in sorted_records if 'cost_per_round' in r]),
                       'COST_OF_CONVERGENCE': cost_of_convergence, 'TARGET_ACC_FOR_COC': target_accuracy_for_coc,
                       'COC_TARGET_REACHED_ROUND': target_accuracy_reached_round,
                       'PAYMENT_GINI_COEFFICIENT': payment_gini_coefficient, 'TOTAL_COST': final_cumulative_cost,
                       'TOTAL_ROUNDS': len(sorted_records), 'AVG_BENIGN_SELLER_SELECTION_RATE': np.mean(
                    [r['benign_selection_rate_in_round'] for r in sorted_records if
                     'benign_selection_rate_in_round' in r and pd.notna(r['benign_selection_rate_in_round'])]),
                       'AVG_BENIGN_PAYMENT_GINI': np.mean(
                           [r['benign_gini_coefficient_in_round'] for r in sorted_records if
                            'benign_gini_coefficient_in_round' in r and pd.notna(
                                r['benign_gini_coefficient_in_round'])])}
            for hypo_adv_rate_key_str, rates_list in baseline_designated_group_selection_rates_summary_collector.items():
                summary_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate_key_str}'
                if rates_list: # If list has rates (i.e., was a No Attack run and clients were selected)
                    summary[summary_key] = np.mean(rates_list)
                else: # Could be an attack run, or No Attack run with no selections/no group
                    summary[summary_key] = np.nan

            # Ensure all potential baseline columns exist in all summaries, even if NaN (for Attack runs)
            if run_attack_method != 'None' and run_attack_method != 'No Attack':
                for hypo_adv_rate in hypothetical_adv_rates_for_baselines:
                    summary_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate:.1f}'
                    if summary_key not in summary: # If not added because it was an attack run
                        summary[summary_key] = np.nan
            return processed_data, summary
        return [], {}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        traceback.print_exc()
        return [], {}


# --- Helper function for Gini, already defined above ---

# ... (rest of your script: get_save_path, load_attack_params, average_dicts, process_all_experiments, analyze_client_level_selection, main block)
# Make sure to update `average_dicts` if you want to average the new economic metrics.
# For Gini, CoC, it might make more sense to report the individual run values or their distribution rather than a simple mean.

def average_dicts(dict_list):
    if not dict_list:
        return {}

    averaged_dict = {}
    # Attempt to get keys from the first dictionary, handle if it's empty
    if not dict_list[0]: return {}
    keys = dict_list[0].keys()

    for key in keys:
        values = [d.get(key) for d in dict_list if
                  d is not None and d.get(key) is not None]  # Filter out None values for the key

        if not values:  # If no valid values for this key, skip or set to None
            averaged_dict[key] = None
            continue

        if key == "run":  # Or other non-numeric identifiers
            averaged_dict[key] = values[0]  # Keep the first (assuming they should be identical or run is an ID)
            continue
        # Add other string/categorical keys that should not be averaged
        if key in ['AGGREGATION_METHOD', 'DATA_SPLIT_MODE', 'ATTACK_METHOD', 'IS_SYBIL', 'CHANGE_BASE', 'TRIGGER_MODE',
                   'buyer_data_mode', 'TARGET_ACC_FOR_COC']:
            averaged_dict[key] = values[0]  # Assume these are constant per group being averaged
            continue

        if isinstance(values[0], (int, float, np.number)):
            averaged_dict[key] = np.nanmean(values)  # Use nanmean to ignore NaNs (e.g., for CoC if not reached)
        else:
            # For lists or other types, you might want different aggregation
            # For now, just take the first if not numeric
            averaged_dict[key] = values[0]
    return averaged_dict


# def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run):
#     """
#     Process a single experiment file and extract metrics, incorporating data distribution similarity.
#
#     Args:
#         file_path: Path to the market_log.ckpt file
#         attack_params: Dictionary containing attack parameters
#         market_params: Dictionary containing market parameters
#         data_statistics_path: Path to the data_statistics.json file
#         adv_rate: Proportion of sellers considered adversaries
#
#     Returns:
#         processed_data: List of dictionaries with processed round data
#         summary_data: Dictionary with summary metrics
#     """
#     try:
#         experiment_data = torch.load(file_path, map_location='cpu', weights_only=False)
#         data_stats = load_json(data_statistics_path)
#
#         buyer_distribution = data_stats['buyer_stats']['class_distribution']
#         seller_distributions = data_stats['seller_stats']
#
#         if not experiment_data:
#             print(f"Warning: No round records found in {file_path}")
#             return [], {}
#
#         processed_data = []
#         num_adversaries = int(len(seller_distributions) * adv_rate)
#
#         for i, record in enumerate(experiment_data):
#             round_num = record.get('round_number', i)
#
#             selected_clients = record.get("used_sellers", [])
#             adversary_selections = [cid for cid in selected_clients if int(cid) < num_adversaries]
#             benign_selections = [cid for cid in selected_clients if int(cid) >= num_adversaries]
#
#             round_data = {
#                 'run': cur_run,
#                 'round': round_num,
#                 **attack_params,
#                 **market_params,
#                 'n_selected_clients': len(selected_clients),
#                 'selected_clients': selected_clients,
#                 'adversary_selection_rate': len(adversary_selections) / len(
#                     selected_clients) if selected_clients else 0,
#                 'benign_selection_rate': len(benign_selections) / len(selected_clients) if selected_clients else 0
#             }
#             similarities = [
#                 calculate_distribution_similarity(buyer_distribution,
#                                                   seller_distributions[str(cid)]['class_distribution'])
#                 for cid in selected_clients
#             ]
#
#             round_data['avg_selected_data_distribution_similarity'] = np.mean(similarities) if similarities else 0
#
#             un_selected_similarities = [
#                 calculate_distribution_similarity(buyer_distribution,
#                                                   seller_distributions[str(cid)]['class_distribution'])
#                 for cid in range(market_params["N_CLIENTS"]) if cid not in selected_clients
#             ]
#             round_data['avg_unselected_data_distribution_similarity'] = np.mean(
#                 un_selected_similarities) if un_selected_similarities else 0
#
#             final_perf = record.get('final_perf_global', {})
#             round_data['main_acc'] = final_perf.get('acc')
#             round_data['main_loss'] = final_perf.get('loss')
#
#             poison_metrics = record.get('extra_info', {}).get('poison_metrics', {})
#             round_data.update({
#                 'clean_acc': poison_metrics.get('clean_accuracy'),
#                 'triggered_acc': poison_metrics.get('triggered_accuracy'),
#                 'asr': poison_metrics.get('attack_success_rate')
#             })
#
#             processed_data.append(round_data)
#
#         sorted_records = sorted(processed_data, key=lambda x: x['round'])
#
#         if sorted_records:
#             asr_values = [r.get('asr') or 0 for r in sorted_records]
#             final_record = sorted_records[-1]
#
#             summary = {
#                 "run": cur_run,
#                 **market_params,
#                 **attack_params,
#                 'MAX_ASR': max(asr_values),
#                 'FINAL_ASR': final_record.get('asr'),
#                 'FINAL_MAIN_ACC': final_record.get('main_acc'),
#                 'FINAL_CLEAN_ACC': final_record.get('clean_acc'),
#                 'FINAL_TRIGGERED_ACC': final_record.get('triggered_acc'),
#                 'AVG_SELECTED_DISTRIBUTION_SIMILARITY': np.mean(
#                     [r['avg_selected_data_distribution_similarity'] for r in sorted_records]),
#                 'AVG_UNSELECTED_DISTRIBUTION_SIMILARITY': np.mean(
#                     [r['avg_unselected_data_distribution_similarity'] for r in sorted_records]),
#                 'AVG_ADVERSARY_SELECTION_RATE': np.mean([r['adversary_selection_rate'] for r in sorted_records]),
#                 'AVG_BENIGN_SELECTION_RATE': np.mean([r['benign_selection_rate'] for r in sorted_records]),
#                 'TOTAL_ROUNDS': len(sorted_records)
#             }
#
#             return processed_data, summary
#
#         return [], {}
#
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         traceback.print_exc()
#         return [], {}


def get_save_path(n_sellers, local_epoch, local_lr, gradient_manipulation_mode,
                  sybil_mode=False, is_sybil="False", data_split_mode='iid',
                  aggregation_method='fedavg', dataset_name='cifar10',
                  poison_strength=None, trigger_rate=None, trigger_type=None,
                  adv_rate=None, change_base="True", trigger_attack_mode="", exp_name="", discovery_quality=0.1,
                  buyer_data_mode=""):
    """
    Construct a save path based on the experiment parameters.

    Args:
        n_sellers: Number of sellers
        local_epoch: Number of local epochs
        local_lr: Local learning rate
        gradient_manipulation_mode: Type of attack ("None", "cmd", "single")
        sybil_mode: Mode of sybil attack
        is_sybil: Whether sybil attack is used
        data_split_mode: Data split mode
        aggregation_method: Aggregation method used
        dataset_name: Name of the dataset
        poison_strength: Strength of poisoning (for "cmd")
        trigger_rate: Rate of trigger insertion
        trigger_type: Type of trigger used
        adv_rate: Rate of adversaries

    Returns:
        A string representing the path.
    """
    # Use is_sybil flag or, if not true, use sybil_mode
    sybil_str = is_sybil

    if aggregation_method == "martfl":
        base_dir = Path(
            "./results") / exp_name / f"backdoor_trigger_{trigger_attack_mode}" / f"is_sybil_{sybil_str}" / f"is_iid_{data_split_mode}" / f"buyer_data_{buyer_data_mode}" / f"{aggregation_method}_{change_base}" / dataset_name
    else:
        base_dir = Path(
            "./results") / exp_name / f"backdoor_trigger_{trigger_attack_mode}" / f"is_sybil_{sybil_str}" / f"is_iid_{data_split_mode}" / f"buyer_data_{buyer_data_mode}" / aggregation_method / dataset_name

    if gradient_manipulation_mode == "None":
        subfolder = "no_attack"
        param_str = f"n_seller_{n_sellers}_local_epoch_{local_epoch}_local_lr_{local_lr}"
    elif gradient_manipulation_mode == "cmd":
        subfolder = f"backdoor_mode_{gradient_manipulation_mode}_strength_{poison_strength}_trigger_rate_{trigger_rate}_trigger_type_{trigger_type}"
        param_str = f"n_seller_{n_sellers}_adv_rate_{adv_rate}_local_epoch_{local_epoch}_local_lr_{local_lr}"
    elif gradient_manipulation_mode == "single":
        subfolder = f"backdoor_mode_{gradient_manipulation_mode}_trigger_rate_{trigger_rate}_trigger_type_{trigger_type}"
        param_str = f"n_seller_{n_sellers}_adv_rate_{adv_rate}_local_epoch_{local_epoch}_local_lr_{local_lr}"
    else:
        raise NotImplementedError(f"No such attack type: {gradient_manipulation_mode}")
    if data_split_mode == "discovery":
        discovery_str = f"discovery_quality_{discovery_quality}"
        save_path = base_dir / discovery_str / subfolder / param_str
    # Construct the full save path
    else:
        # Construct the full save path
        save_path = base_dir / subfolder / param_str
    return str(save_path)


def load_attack_params(path):
    with open(os.path.join(path, "attack_params.json"), 'r') as f:
        return json.load(f)


def average_dicts(dict_list):
    if not dict_list:
        return {}

    averaged_dict = {}
    keys = dict_list[0].keys()

    for key in keys:
        values = [d[key] for d in dict_list]
        if key == "run":
            averaged_dict[key] = max(values)  # Keep the first non-numeric value (assuming all are identical)
            continue
        if isinstance(values[0], (int, float, np.number)):
            averaged_dict[key] = np.mean(values)
        else:
            averaged_dict[key] = values[0]  # Keep the first non-numeric value (assuming all are identical)

    return averaged_dict


def process_all_experiments(output_dir='./processed_data', local_epoch=2,
                            aggregation_methods=['martfl', 'fedavg'], exp_name=""):
    """
    Process all experiment files for multiple aggregation methods.

    Args:
        output_dir: Directory to save processed data
        local_epoch: Local epoch setting used in experiments
        aggregation_methods: List of aggregation methods to process
    """
    all_processed_data = []
    all_summary_data_avg = []
    all_summary_data = []
    trigger_type = "blended_patch"
    dataset_name = "FMNIST"
    n_sellers = 30

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each aggregation method
    for aggregation_method in aggregation_methods:
        print(f"\nProcessing experiments for {aggregation_method}...")
        for data_split_mode in ["discovery"]:
            for grad_mode in ['single', "None"]:
                for trigger_attack_mode in ['static', 'dynamic']:
                    for trigger_rate in [0.1, 0.5]:
                        for is_sybil in ["False", "mimic"]:
                            for adv_rate in [0.1, 0.2, 0.3, 0.4]:
                                for change_base in ["True", "False"]:
                                    for discovery_quality in ["0.1", "1.0", "10.0"]:
                                        for buyer_data_mode in ["random", "biased"]:
                                            if aggregation_method == "fedavg" and change_base == "True":
                                                continue

                                            base_save_path = get_save_path(
                                                n_sellers=n_sellers,
                                                adv_rate=adv_rate,
                                                local_epoch=local_epoch,
                                                local_lr=1e-2,
                                                gradient_manipulation_mode=grad_mode,
                                                poison_strength=0,
                                                trigger_type=trigger_type,
                                                is_sybil=is_sybil,
                                                trigger_rate=trigger_rate,
                                                aggregation_method=aggregation_method,
                                                data_split_mode=data_split_mode,
                                                change_base=change_base,
                                                dataset_name=dataset_name,
                                                trigger_attack_mode=trigger_attack_mode,
                                                exp_name=exp_name,
                                                discovery_quality=discovery_quality,
                                                buyer_data_mode=buyer_data_mode
                                            )

                                            # Find all runs
                                            run_paths = sorted(glob.glob(f"{base_save_path}/run_*"))
                                            if not run_paths:
                                                print(f"No runs found in: {base_save_path}")
                                                continue

                                            aggregated_processed_data = []
                                            aggregated_summaries = []
                                            params = load_attack_params(base_save_path)
                                            run_cnt = 0
                                            for run_path in run_paths:
                                                file_path = os.path.join(run_path, "market_log.ckpt")
                                                data_statistics_path = os.path.join(run_path, "data_statistics.json")
                                                if not os.path.exists(file_path):
                                                    print(f"File not found: {file_path}")
                                                    continue

                                                print(f"Processing: {file_path}")

                                                # Load params from attack_params.json

                                                attack_params = {
                                                    'ATTACK_METHOD': params["local_attack_params"][
                                                        "gradient_manipulation_mode"],
                                                    'TRIGGER_RATE': params["local_attack_params"]["trigger_rate"],
                                                    'IS_SYBIL': params["sybil_params"]["sybil_mode"] if
                                                    params["sybil_params"][
                                                        "is_sybil"] else "False",
                                                    'ADV_RATE': adv_rate if params["sybil_params"]["adv_rate"] == 0 else
                                                    params["sybil_params"]["adv_rate"],
                                                    'CHANGE_BASE': change_base,
                                                    'TRIGGER_MODE': params["sybil_params"]["trigger_mode"],
                                                    "benign_rounds": params["sybil_params"]["benign_rounds"],
                                                    "trigger_mode": params["sybil_params"]["trigger_mode"],

                                                }
                                                if data_split_mode == "discovery":
                                                    market_params = {
                                                        'AGGREGATION_METHOD': aggregation_method,
                                                        'DATA_SPLIT_MODE': data_split_mode,
                                                        "discovery_quality": params["dm_params"]["discovery_quality"],
                                                        "buyer_data_mode": params["dm_params"]["buyer_data_mode"],
                                                        'N_CLIENTS': n_sellers
                                                    }

                                                else:
                                                    market_params = {
                                                        'AGGREGATION_METHOD': aggregation_method,
                                                        'DATA_SPLIT_MODE': data_split_mode,
                                                        'N_CLIENTS': n_sellers
                                                    }

                                                processed_data, summary = process_single_experiment(
                                                    file_path,
                                                    attack_params,
                                                    market_params,
                                                    data_statistics_path=data_statistics_path,
                                                    adv_rate=adv_rate,
                                                    cur_run=run_cnt

                                                )
                                                run_cnt += 1
                                                aggregated_processed_data.extend(processed_data)
                                                if summary:
                                                    aggregated_summaries.append(summary)

                                                # Aggregate numeric fields safely:

                                            if aggregated_summaries:
                                                avg_summary = average_dicts(aggregated_summaries)
                                                print(avg_summary)
                                                all_summary_data_avg.append(avg_summary)
                                            all_processed_data.extend(aggregated_processed_data)
                                            all_summary_data.extend(aggregated_summaries)
    # Convert to DataFrames
    all_rounds_df = pd.DataFrame(all_processed_data)
    summary_df_avg = pd.DataFrame(all_summary_data_avg)
    summary_data = pd.DataFrame(all_summary_data)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    all_rounds_csv = f"{output_dir}/all_rounds.csv"
    summary_csv_avg = f"{output_dir}/summary_avg.csv"
    summary_csv = f"{output_dir}/summary.csv"

    all_rounds_df.to_csv(all_rounds_csv, index=False)
    print(f"Saved all rounds data to {all_rounds_csv}")

    summary_df_avg.to_csv(summary_csv_avg, index=False)
    print(f"Saved summary data to {summary_csv_avg}")

    summary_data.to_csv(summary_csv, index=False)
    print(f"Saved summary data to {summary_csv}")

    # Return DataFrames for further analysis if desired
    return all_rounds_df, summary_df_avg


def analyze_client_level_selection(processed_data, seller_raw_data_stats):
    """
    Analyze individual client-level selection behavior and its relation to the sellers' raw data distribution.

    Args:
        processed_data (list of dict): Processed round data from process_single_experiment.
            Each dictionary should contain keys such as 'selected_clients', 'round', etc.
        seller_raw_data_stats (dict): Dictionary mapping seller IDs to raw data distribution metrics.
            For example: {
                'seller1': {'distribution_similarity': 0.95, 'kl_divergence': 0.1, ...},
                'seller2': {'distribution_similarity': 0.80, 'kl_divergence': 0.3, ...},
                ...
            }

    Returns:
        analysis_results (dict): A dictionary containing aggregated insights, including:
            - Total rounds
            - Per-seller selection counts and frequency (percentage)
            - Basic statistics (mean, variance) of selection frequency
            - Pearson correlation between a chosen raw distribution metric and selection frequency
            - Optionally, plots for further visual analysis.
    """
    # Total rounds in the experiment
    total_rounds = len(processed_data)

    # Gather list of all seller IDs from seller_raw_data_stats
    all_seller_ids = list(seller_raw_data_stats.keys())

    # Initialize counts for each seller (even if they never appear as selected)
    seller_selection_counts = {seller: 0 for seller in all_seller_ids}

    # Iterate over each round record and update selection counts.
    for record in processed_data:
        selected = record.get('selected_clients', [])
        for cid in selected:
            if cid in seller_selection_counts:
                seller_selection_counts[cid] += 1
            else:
                seller_selection_counts[cid] = 1

    # Compute selection frequency (rate per seller)
    seller_selection_freq = {seller: count / total_rounds
                             for seller, count in seller_selection_counts.items()}

    # Compute summary statistics for selection frequency
    freq_values = np.array(list(seller_selection_freq.values()))
    mean_freq = np.mean(freq_values)
    var_freq = np.var(freq_values)

    # Now, correlate selection frequency with a raw data distribution metric.
    # We assume that each seller has a metric "distribution_similarity"
    # (if not, we can convert from "kl_divergence": lower divergence -> higher similarity).
    metric_list = []
    selection_freq_list = []
    for seller in all_seller_ids:
        stats_dict = seller_raw_data_stats.get(seller, {})
        # Use 'distribution_similarity' if available; else derive one from 'kl_divergence'
        if "distribution_similarity" in stats_dict:
            metric = stats_dict["distribution_similarity"]
        elif "kl_divergence" in stats_dict:
            # A simple conversion: similarity = exp(-KL divergence)
            metric = np.exp(-stats_dict["kl_divergence"])
        else:
            continue  # skip if no metric is provided
        metric_list.append(metric)
        selection_freq_list.append(seller_selection_freq.get(seller, 0))

    if len(metric_list) > 1:
        corr, p_val = stats.pearsonr(metric_list, selection_freq_list)
    else:
        corr, p_val = None, None

    # Optionally, produce a scatter plot showing the relation between raw distribution metric and selection frequency.
    plt.figure(figsize=(6, 4))
    plt.scatter(metric_list, selection_freq_list, alpha=0.7)
    plt.xlabel('Raw Data Distribution Similarity')
    plt.ylabel('Selection Frequency')
    plt.title('Per-Seller Selection Frequency vs. Raw Data Distribution')
    plt.grid(True)
    plt.show()

    # You might also want to plot a histogram of selection frequencies:
    plt.figure(figsize=(6, 4))
    plt.hist(freq_values, bins=10, alpha=0.8, edgecolor='black')
    plt.xlabel('Selection Frequency')
    plt.ylabel('Number of Sellers')
    plt.title('Histogram of Seller Selection Frequencies')
    plt.grid(True)
    plt.show()

    # Compile analysis results into a dictionary
    analysis_results = {
        "total_rounds": total_rounds,
        "seller_selection_counts": seller_selection_counts,
        "seller_selection_frequency": seller_selection_freq,
        "mean_selection_frequency": mean_freq,
        "variance_selection_frequency": var_freq,
        "raw_metric": metric_list,
        "selection_freq_list": selection_freq_list,
        "pearson_correlation": corr,
        "p_value": p_val,
    }

    print("Total Rounds:", total_rounds)
    print("Mean Selection Frequency:", mean_freq)
    print("Variance in Selection Frequency:", var_freq)
    if corr is not None:
        print("Correlation between raw data similarity and selection frequency:", corr)
        print("P-value:", p_val)
    else:
        print("Not enough data to compute correlation.")

    return analysis_results


# Example usage:
# processed_data = process_single_experiment(file_path, attack_params, aggregation_method)[0]
# seller_raw_data_stats = load_seller_distribution_stats()  # Your function to load distribution stats.
# analysis_results = analyze_client_level_selection(processed_data, seller_raw_data_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process federated learning backdoor attack logs")
    parser.add_argument("--output_dir", default="./processed_data", help="Output directory for processed data")
    parser.add_argument("--local_epoch", type=int, default=2, help="Local epoch setting used in experiments")
    parser.add_argument("--aggregation_methods", nargs='+', default=['martfl'],
                        help="List of aggregation methods to process")
    parser.add_argument("--exp_name", type=str, default="experiment_20250306_170329", help="experiment name")

    args = parser.parse_args()

    # Process all experiments
    all_rounds_df, summary_df = process_all_experiments(
        output_dir=args.output_dir,
        local_epoch=args.local_epoch,
        aggregation_methods=args.aggregation_methods,
        exp_name=args.exp_name
    )

    # Print summary statistics
    if not summary_df.empty:
        print("\nSummary Statistics:")
        print(f"Total experiments processed: {len(summary_df)}")
        print(f"Average Final ASR: {summary_df['FINAL_ASR'].mean():.4f}")
        print(f"Average Main Accuracy: {summary_df['FINAL_MAIN_ACC'].mean():.4f}")

        # Group by aggregation method
        for agg_method in summary_df['AGGREGATION_METHOD'].unique():
            agg_data = summary_df[summary_df['AGGREGATION_METHOD'] == agg_method]
            print(f"\nAggregation Method: {agg_method}")
            print(f"  Average ASR: {agg_data['FINAL_ASR'].mean():.4f}")
            print(f"  Average Main Accuracy: {agg_data['FINAL_MAIN_ACC'].mean():.4f}")

            # Group by gradient mode within each aggregation method
            for grad_mode in agg_data['ATTACK_METHOD'].unique():
                grad_data = agg_data[agg_data['ATTACK_METHOD'] == grad_mode]
                print(f"    Gradient Mode: {grad_mode}")
                print(f"      Average ASR: {grad_data['FINAL_ASR'].mean():.4f}")
                print(f"      Average Main Accuracy: {grad_data['FINAL_MAIN_ACC'].mean():.4f}")
