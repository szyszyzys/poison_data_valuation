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


# Assume average_dicts and process_single_experiment are defined elsewhere
# Example placeholder for average_dicts
def average_dicts(dict_list):
    if not dict_list:
        return {}
    # Simple averaging for numeric values, assumes all dicts have same keys
    # More robust implementation might be needed depending on dict contents
    avg_dict = {}
    keys = dict_list[0].keys()
    for key in keys:
        values = [d.get(key) for d in dict_list if isinstance(d.get(key), (int, float))]
        if values:
            avg_dict[key] = np.mean(values)
        else:
            # Keep non-numeric or missing values from the first dict (or handle differently)
            avg_dict[key] = dict_list[0].get(key)
    # Copy over the non-numeric identifying keys from the first dict
    for key in keys:
        if key not in avg_dict:  # If it wasn't numeric and averaged
            avg_dict[key] = dict_list[0].get(key)

    return avg_dict


# Assume process_single_experiment exists and has this signature:
# def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run):
#     # ... processes the data ...
#     # returns processed_data (list of dicts), summary (dict or None)
#     pass # Placeholder
def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run,
                              target_accuracy_for_coc=0.8, convergence_milestones=None):
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
        if convergence_milestones is None:
            convergence_milestones = [0.70, 0.75, 0.8, 0.85, 0.9]  # Default to a single 80% milestone if not provided
        milestone_convergence_info = {acc: None for acc in convergence_milestones}
        cumulative_cost_for_milestones_tracker = 0  # Single cumulative cost tracker for all milestones
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
            cumulative_cost_for_milestones_tracker += cost_per_round  # Update before checking milestones
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

            # --- Accuracy, ASR, and Convergence Milestone Check ---
            final_perf = record.get('final_perf_global', {})
            current_accuracy = final_perf.get('acc')
            round_data['main_acc'] = current_accuracy
            round_data['main_loss'] = final_perf.get('loss')
            poison_metrics = record.get('extra_info', {}).get('poison_metrics', {})
            round_data['clean_acc'] = poison_metrics.get('clean_accuracy')
            round_data['triggered_acc'] = poison_metrics.get('triggered_accuracy')
            round_data['asr'] = poison_metrics.get('attack_success_rate')

            if current_accuracy is not None:
                for target_acc in convergence_milestones:
                    if milestone_convergence_info[target_acc] is None and current_accuracy >= target_acc:
                        milestone_convergence_info[target_acc] = {
                            'round': round_num + 1,  # Record 1-indexed round
                            'cost': cumulative_cost_for_milestones_tracker
                        }

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
                if rates_list:  # If list has rates (i.e., was a No Attack run and clients were selected)
                    summary[summary_key] = np.mean(rates_list)
                else:  # Could be an attack run, or No Attack run with no selections/no group
                    summary[summary_key] = np.nan

            # Add convergence milestone info to summary
            for target_acc_ms, info_ms in milestone_convergence_info.items():
                acc_label = f"{int(target_acc_ms * 100)}"
                summary[f'ROUNDS_TO_{acc_label}ACC'] = info_ms['round'] if info_ms else np.nan
                summary[f'COST_TO_{acc_label}ACC'] = info_ms['cost'] if info_ms else np.nan

            # Ensure all potential baseline columns exist in all summaries, even if NaN (for Attack runs)
            if run_attack_method != 'None' and run_attack_method != 'No Attack':
                for hypo_adv_rate in hypothetical_adv_rates_for_baselines:
                    summary_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate:.1f}'
                    if summary_key not in summary:  # If not added because it was an attack run
                        summary[summary_key] = np.nan
            return processed_data, summary
        return [], {}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        traceback.print_exc()
        return [], {}


import os
import json
import glob
import pandas as pd
import numpy as np  # For average_dicts if used
import traceback  # For detailed error printing


# --- Placeholder for process_single_experiment and average_dicts ---
# You need to have these functions defined or imported.
# For this example, I'll create dummy versions.

def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run,
                              target_accuracy_for_coc=0.8, convergence_milestones=None):
    """
    DUMMY process_single_experiment. Replace with your actual function.
    """
    # print(f"    [DUMMY] Processing run {cur_run} from {file_path} with params: A={attack_params}, M={market_params}")
    if not os.path.exists(file_path):
        # print(f"    [DUMMY] File not found: {file_path}")
        return [], {}

    # Simulate some data processing
    round_data_example = {
        'run': cur_run, 'round': 0, **attack_params, **market_params,
        'main_acc': np.random.rand(), 'asr': np.random.rand() * adv_rate,
        'cost_per_round': 10, 'adversary_selection_rate': adv_rate * np.random.rand(),
        # Add all columns expected by all_rounds_df
    }
    # Add milestone data
    if convergence_milestones is None:
        convergence_milestones = [0.7, 0.8, 0.9]
    for acc_thresh in convergence_milestones:
        round_data_example[f'ROUNDS_TO_{int(acc_thresh * 100)}ACC'] = np.random.randint(10, 50) if round_data_example[
                                                                                                       'main_acc'] > acc_thresh else np.nan
        round_data_example[f'COST_TO_{int(acc_thresh * 100)}ACC'] = round_data_example[
                                                                        f'ROUNDS_TO_{int(acc_thresh * 100)}ACC'] * 10 if pd.notna(
            round_data_example[f'ROUNDS_TO_{int(acc_thresh * 100)}ACC']) else np.nan

    summary_example = {
        'run': cur_run, **market_params, **attack_params,  # market_params and attack_params are dicts
        'FINAL_MAIN_ACC': round_data_example['main_acc'],
        'FINAL_ASR': round_data_example['asr'],
        'AVG_ADVERSARY_SELECTION_RATE': round_data_example['adversary_selection_rate'],
        'COST_OF_CONVERGENCE': np.random.randint(100, 500) if round_data_example[
                                                                  'main_acc'] > target_accuracy_for_coc else np.nan,
        'TOTAL_COST': 500,
        'exp_path': os.path.dirname(os.path.dirname(file_path))  # Grandparent dir for exp path
    }
    # Add milestone data to summary
    for acc_thresh in convergence_milestones:
        summary_example[f'ROUNDS_TO_{int(acc_thresh * 100)}ACC'] = round_data_example[
            f'ROUNDS_TO_{int(acc_thresh * 100)}ACC']
        summary_example[f'COST_TO_{int(acc_thresh * 100)}ACC'] = round_data_example[
            f'COST_TO_{int(acc_thresh * 100)}ACC']

    return [round_data_example], summary_example


def average_dicts(list_of_dicts):
    """
    DUMMY average_dicts. Averages numeric values for common keys.
    Non-numeric values from the first dict are preserved if keys match.
    """
    if not list_of_dicts:
        return {}

    # Initialize with non-numeric data from the first dictionary
    # and prepare for summing numeric data.
    avg_dict = {}
    sum_dict = {}
    count_dict = {}

    first_dict = list_of_dicts[0]
    for key, value in first_dict.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):  # Exclude bools from averaging
            sum_dict[key] = 0
            count_dict[key] = 0
        else:
            avg_dict[key] = value  # Preserve non-numeric or bool

    # Aggregate sums and counts for numeric values
    for d in list_of_dicts:
        for key, value in d.items():
            if key in sum_dict:  # If it's a key we decided to average
                if pd.notna(value) and isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                    sum_dict[key] += value
                    count_dict[key] += 1

    # Calculate averages
    for key in sum_dict:
        if count_dict[key] > 0:
            avg_dict[key] = sum_dict[key] / count_dict[key]
        else:
            avg_dict[key] = np.nan  # Or some other placeholder like None or first_dict[key]

    return avg_dict


# --- End of Placeholder Functions ---


def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run,
                              target_accuracy_for_coc=0.8, convergence_milestones=None):
    if convergence_milestones is None: convergence_milestones = [0.7, 0.8, 0.9]
    if not os.path.exists(file_path): return [], {}

    # Use ATTACK_METHOD from attack_params to simulate different behaviors
    current_attack_type = attack_params.get('ATTACK_METHOD', 'None')
    sim_asr = 0.0
    if current_attack_type == 'Backdoor':
        sim_asr = np.random.rand() * adv_rate * 0.8  # Higher ASR for backdoor
    elif current_attack_type == 'LabelFlipping':
        sim_asr = np.random.rand() * adv_rate * 0.5  # Lower ASR for label flipping

    round_data_example = {
        'run': cur_run, 'round': 0, **attack_params, **market_params,
        'main_acc': np.random.uniform(0.3, 0.9), 'asr': sim_asr,
        'cost_per_round': 10, 'adversary_selection_rate': adv_rate * np.random.uniform(0.2, 0.8),
    }
    for acc_thresh in convergence_milestones:
        reached = round_data_example['main_acc'] > acc_thresh
        round_data_example[f'ROUNDS_TO_{int(acc_thresh * 100)}ACC'] = np.random.randint(10, 50) if reached else np.nan
        round_data_example[f'COST_TO_{int(acc_thresh * 100)}ACC'] = round_data_example[
                                                                        f'ROUNDS_TO_{int(acc_thresh * 100)}ACC'] * 10 if reached else np.nan

    summary_example = {
        'run': cur_run, **market_params, **attack_params,
        'FINAL_MAIN_ACC': round_data_example['main_acc'], 'FINAL_ASR': round_data_example['asr'],
        'AVG_ADVERSARY_SELECTION_RATE': round_data_example['adversary_selection_rate'],
        'COST_OF_CONVERGENCE': np.random.randint(100, 500) if round_data_example[
                                                                  'main_acc'] > target_accuracy_for_coc else np.nan,
        'TOTAL_COST': 500, 'exp_path': os.path.dirname(os.path.dirname(file_path))
    }
    for acc_thresh in convergence_milestones:
        summary_example[f'ROUNDS_TO_{int(acc_thresh * 100)}ACC'] = round_data_example[
            f'ROUNDS_TO_{int(acc_thresh * 100)}ACC']
        summary_example[f'COST_TO_{int(acc_thresh * 100)}ACC'] = round_data_example[
            f'COST_TO_{int(acc_thresh * 100)}ACC']
    return [round_data_example], summary_example


def average_dicts(list_of_dicts):
    if not list_of_dicts: return {}
    avg_dict, sum_dict, count_dict = {}, {}, {}
    first_dict = list_of_dicts[0]
    for key, value in first_dict.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            sum_dict[key], count_dict[key] = 0, 0
        else:
            avg_dict[key] = value
    for d in list_of_dicts:
        for key, value in d.items():
            if key in sum_dict and pd.notna(value) and isinstance(value, (int, float, np.number)) and not isinstance(
                    value, bool):
                sum_dict[key] += value
                count_dict[key] += 1
    for key in sum_dict:
        avg_dict[key] = sum_dict[key] / count_dict[key] if count_dict[key] > 0 else np.nan
    return avg_dict


# --- End of Placeholder Functions ---

def process_all_experiments_revised(base_results_dir='./experiment_results_revised',
                                    output_dir='./processed_data_revised',
                                    filter_params=None,
                                    verbose=True):
    all_processed_data = []
    all_summary_data_avg = []
    all_summary_data = []

    os.makedirs(output_dir, exist_ok=True)
    print(f"Scanning for experiments in: {os.path.abspath(base_results_dir)}")

    experiment_found_flag = False
    processed_experiment_count = 0

    for root, dirs, files in os.walk(base_results_dir):
        if 'experiment_params.json' in files:
            experiment_found_flag = True
            exp_params_path = os.path.join(root, 'experiment_params.json')
            # (JSON loading and initial full_config same as your code)
            try:
                with open(exp_params_path, 'r') as f:
                    params = json.load(f)
            except Exception as e:
                print(f"  ERROR loading {exp_params_path}: {e}. Skipping."); continue
            full_config = params.get('full_config', params)

            def get_param(config_dict, path, default_val='N/A'):
                # (get_param helper same as your code)
                current = config_dict
                for key in path.split('.'):
                    if not isinstance(current, dict) or key not in current: return default_val
                    current = current.get(key)
                return current

            # --- Parameter Extraction (Focus on ATTACK_METHOD) ---
            try:
                aggregation_method = get_param(full_config, 'aggregation_method', 'N/A')
                adv_rate = get_param(full_config, 'data_split.adv_rate', 0.0)
                # ... (other param extractions as in your code) ...
                dataset_name = get_param(full_config, 'dataset_name', 'UnknownDataset')
                data_split_mode = get_param(full_config, 'data_split.data_split_mode', 'N/A')
                discovery_quality_raw = get_param(full_config, 'data_split.dm_params.discovery_quality', 'N/A')
                discovery_quality = str(discovery_quality_raw) if data_split_mode == "discovery" else 'N/A'
                buyer_data_mode = get_param(full_config, 'data_split.dm_params.buyer_data_mode',
                                            'N/A') if data_split_mode == "discovery" else 'N/A'
                n_sellers = get_param(full_config, 'data_split.num_sellers', 100)
                local_epoch = get_param(full_config, 'training.local_training_params.local_epochs', 1)
                is_sybil_bool = get_param(full_config, 'sybil.is_sybil', False)
                sybil_mode_from_config = get_param(full_config, 'sybil.sybil_mode', 'default')
                is_sybil_str = sybil_mode_from_config if is_sybil_bool else "False"
                trigger_attack_mode = get_param(full_config, 'sybil.trigger_mode', 'always_on')
                change_base_bool = get_param(full_config, 'federated_learning.change_base', False)
                change_base = str(change_base_bool)
                poison_strength = get_param(full_config, 'attack.poison_strength', 0.0)  # Extracted earlier

                attack_enabled = get_param(full_config, 'attack.enabled', False)
                # Determine ATTACK_METHOD based on gradient_manipulation_mode
                # Standardize 'None' or empty to 'NoAttack' for clarity
                raw_attack_mode = get_param(full_config, 'attack.gradient_manipulation_mode', 'None')
                if not attack_enabled or raw_attack_mode.lower() in ['none', '']:
                    attack_method_final = 'NoAttack'  # Standardized name for no attack
                    trigger_rate_final = 0.0  # Ensure trigger rate is 0 if no attack
                    adv_rate_final = 0.0  # Often adv_rate is considered 0 for NoAttack scenarios
                else:
                    attack_method_final = raw_attack_mode  # e.g., "Backdoor", "LabelFlipping"
                    trigger_rate_final = get_param(full_config, 'attack.poison_rate', 0.0)
                    adv_rate_final = adv_rate  # Use the configured adv_rate for active attacks

                if verbose and processed_experiment_count < 1:
                    print(f"    Extracted Params: AGG={aggregation_method}, ADV_RATE_CONFIG={adv_rate}, "
                          f"ATTACK_METHOD_FINAL={attack_method_final}, TRIGGER_RATE_FINAL={trigger_rate_final}, ADV_RATE_FINAL={adv_rate_final} "
                          f"IS_SYBIL={is_sybil_str}, DQ={discovery_quality}, BM={buyer_data_mode}, DS={dataset_name}")

            except Exception as e:
                print(f"  ERROR: Extracting parameters from {exp_params_path}: {e}. Skipping.")
                traceback.print_exc()
                continue

            # --- Apply Filters (Ensure ATTACK_METHOD_FINAL is used for filtering if 'attack_method' is a filter key) ---
            if filter_params:
                skip_experiment = False
                param_map_for_filter = {
                    'aggregation_method': aggregation_method, 'adv_rate': adv_rate_final,
                    # Use adv_rate_final for filtering
                    'attack_method': attack_method_final,  # Use standardized attack_method for filtering
                    'trigger_rate': trigger_rate_final,  # Use trigger_rate_final for filtering
                    'is_sybil': is_sybil_str, 'change_base': change_base, 'dataset_name': dataset_name,
                    'discovery_quality': discovery_quality, 'buyer_data_mode': buyer_data_mode,
                    'data_split_mode': data_split_mode
                }
                # (Filter logic same as your code, ensure key 'attack_method' in filter_params matches 'attack_method_final')
                for key, allowed_values in filter_params.items():
                    if key not in param_map_for_filter:
                        print(f"    Warning: Filter key '{key}' not recognized. Skipping this filter.")
                        continue
                    actual_value = param_map_for_filter[key]
                    if key in ['adv_rate', 'trigger_rate'] and not isinstance(actual_value, (int, float)):
                        try:
                            actual_value = float(actual_value)
                        except ValueError:
                            pass

                    if isinstance(actual_value, (int, float)):
                        typed_allowed_values = [float(v) if isinstance(v, (int, float)) or (
                                isinstance(v, str) and v.replace('.', '', 1).isdigit()) else v for v in
                                                allowed_values]
                        allowed_values_to_check = typed_allowed_values
                    else:
                        allowed_values_to_check = allowed_values
                    if actual_value not in allowed_values_to_check:
                        if verbose: print(
                            f"    Skipping experiment {root} due to filter: {key}='{actual_value}' (not in {allowed_values_to_check})")
                        skip_experiment = True;
                        break
                if skip_experiment: continue

            if verbose: print(f"  Processing experiment: {root}")

            # --- Construct parameter dicts for process_single_experiment ---
            attack_params_dict = {
                'ATTACK_METHOD': attack_method_final,  # Use the standardized/determined attack method
                'TRIGGER_RATE': trigger_rate_final,  # Use the determined trigger rate
                'IS_SYBIL': is_sybil_str,
                'ADV_RATE': adv_rate_final,  # Pass the determined adv_rate
                'CHANGE_BASE': change_base,
                'TRIGGER_MODE': trigger_attack_mode,
                "poison_strength": poison_strength,
            }
            market_params_dict = {
                'AGGREGATION_METHOD': aggregation_method, 'DATA_SPLIT_MODE': data_split_mode,
                'N_CLIENTS': n_sellers, 'LOCAL_EPOCH': local_epoch, 'DATASET': dataset_name,
            }
            if data_split_mode == "discovery":
                market_params_dict["discovery_quality"] = discovery_quality
                market_params_dict["buyer_data_mode"] = buyer_data_mode

            # (Run processing loop same as your code, using market_log_file = os.path.join(run_dir_path, "market_log.ckpt"))
            # ...
            # Make sure to use market_log_file as you had it: "market_log_final.ckpt" or "market_log.ckpt"
            run_paths = sorted(glob.glob(os.path.join(root, "run_*")))
            if not run_paths:  # (Same as before)
                if verbose: print(f"    No 'run_*' subdirectories found in: {root}"); continue
            if verbose: print(f"    Found {len(run_paths)} potential run directories.")

            current_exp_processed_data, current_exp_summaries = [], []
            for run_idx, run_dir_path in enumerate(run_paths):
                if not os.path.isdir(run_dir_path): continue
                market_log_file = os.path.join(run_dir_path, "market_log_final.ckpt")  # OR "market_log_final.ckpt"
                data_stats_file = os.path.join(run_dir_path, "data_statistics.json")
                if not os.path.exists(market_log_file):
                    if verbose: print(
                        f"      Log file not found in: {run_dir_path} (expected {os.path.basename(market_log_file)})")
                    continue
                data_stats_path_to_pass = data_stats_file if os.path.exists(data_stats_file) else None
                try:
                    processed_run_data, summary_run = process_single_experiment(
                        file_path=market_log_file, attack_params=attack_params_dict,
                        market_params=market_params_dict, data_statistics_path=data_stats_path_to_pass,
                        adv_rate=adv_rate_final, cur_run=run_idx  # Pass adv_rate_final
                    )
                    if processed_run_data: current_exp_processed_data.extend(processed_run_data)
                    if summary_run: summary_run['exp_path'] = root; current_exp_summaries.append(summary_run)
                except Exception as e:
                    print(f"    ERROR processing run {run_dir_path}: {e}"); traceback.print_exc()

            # --- Aggregate results (ensure ATTACK_METHOD, ADV_RATE, TRIGGER_RATE are correctly propagated to avg summary) ---
            if current_exp_summaries:
                processed_experiment_count += 1
                all_summary_data.extend(current_exp_summaries)
                try:
                    avg_summary_for_exp = average_dicts(current_exp_summaries)
                    # Explicitly set/override key identifiers from the experiment-level derived values
                    avg_summary_for_exp.update({
                        'AGGREGATION_METHOD': aggregation_method,
                        'DATASET': dataset_name,
                        'ATTACK_METHOD': attack_method_final,  # Crucial for distinguishing attacks
                        'ADV_RATE': adv_rate_final,  # Crucial
                        'TRIGGER_RATE': trigger_rate_final,  # Crucial
                        'IS_SYBIL': is_sybil_str,
                        'CHANGE_BASE': change_base,
                        'DATA_SPLIT_MODE': data_split_mode,
                        'discovery_quality': discovery_quality,
                        'buyer_data_mode': buyer_data_mode,
                        'N_CLIENTS': n_sellers,  # From market_params
                        'LOCAL_EPOCH': local_epoch,  # From market_params
                        'poison_strength': poison_strength,  # From attack_params
                        'exp_path': root
                    })
                    # Remove any run-specific fields that don't make sense in an averaged summary
                    avg_summary_for_exp.pop('run', None)  # 'run' (cur_run) is run-specific

                    all_summary_data_avg.append(avg_summary_for_exp)
                    if verbose: print(f"    Aggregated {len(current_exp_summaries)} runs for experiment {root}.")
                except Exception as e:
                    print(f"    ERROR averaging summaries for {root}: {e}");
                    traceback.print_exc()
            if current_exp_processed_data: all_processed_data.extend(current_exp_processed_data)

    # (Rest of the saving logic is the same)
    if not experiment_found_flag: print("No 'experiment_params.json' files found.")
    if processed_experiment_count == 0: print("Found experiment params, but no runs yielded summaries.")
    if not all_processed_data and not all_summary_data_avg and not all_summary_data:
        print("No data processed. Output files will be empty.");
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_all_rounds = pd.DataFrame(all_processed_data) if all_processed_data else pd.DataFrame()
    df_summary_avg = pd.DataFrame(all_summary_data_avg) if all_summary_data_avg else pd.DataFrame()
    df_summary_individual_runs = pd.DataFrame(all_summary_data) if all_summary_data else pd.DataFrame()
    # (Saving to CSV logic same as your code)
    all_rounds_csv, summary_csv_avg, summary_csv_individual = (os.path.join(output_dir, f) for f in
                                                               ["all_rounds.csv", "summary_avg.csv",
                                                                "summary_individual_runs.csv"])
    if not df_all_rounds.empty:
        df_all_rounds.to_csv(all_rounds_csv, index=False); print(f"Saved {len(df_all_rounds)} rows to {all_rounds_csv}")
    else:
        print("No all_rounds data.")
    if not df_summary_avg.empty:
        df_summary_avg.to_csv(summary_csv_avg, index=False); print(
            f"Saved {len(df_summary_avg)} rows to {summary_csv_avg}")
    else:
        print("No avg summary data.")
    if not df_summary_individual_runs.empty:
        df_summary_individual_runs.to_csv(summary_csv_individual, index=False); print(
            f"Saved {len(df_summary_individual_runs)} rows to {summary_csv_individual}")
    else:
        print("No individual run summary data.")
    return df_all_rounds, df_summary_avg, df_summary_individual_runs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process federated learning backdoor attack logs")
    parser.add_argument("--output_dir", default="./processed_data", help="Output directory for processed data")
    parser.add_argument("--result_path", default="./experiment_results_revised",
                        help="Output directory for processed data")

    args = parser.parse_args()

    all_df, avg_df, summary_df = process_all_experiments_revised(
        base_results_dir=args.result_path,
        output_dir=args.output_dir
    )

    print("\n--- Processing Complete ---")
    if all_df is not None:
        print("\nAll Rounds DataFrame Head:")
        print(all_df.head())
    if avg_df is not None:
        print("\nAverage Summary DataFrame:")
        print(avg_df)
    if summary_df is not None:
        print("\nIndividual Run Summary DataFrame Head:")
        print(summary_df.head())
