import glob
import json
import os
import traceback

import numpy as np
import pandas as pd
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


def process_all_experiments_revised(base_results_dir='./experiment_results_revised',
                                    output_dir='./processed_data_revised',
                                    filter_params=None):
    """
    Process all experiment results stored with experiment_params.json.

    Args:
        base_results_dir: Root directory containing experiment results.
        output_dir: Directory to save processed data CSVs.
        filter_params (dict, optional): A dictionary of parameters to filter experiments.
                                         Example: {'aggregation_method': ['fedavg', 'fltrust'], 'adv_rate': [0.1, 0.2]}
                                         Only experiments matching ALL specified criteria will be processed.
                                         If None, all found experiments are processed.
    """
    all_processed_data = []
    all_summary_data_avg = []
    all_summary_data = []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning for experiments in: {base_results_dir}")

    # Walk through the base directory
    for root, dirs, files in os.walk(base_results_dir):
        if 'experiment_params.json' in files:
            exp_params_path = os.path.join(root, 'experiment_params.json')
            print(f"\nFound experiment parameters: {exp_params_path}")

            try:
                with open(exp_params_path, 'r') as f:
                    params = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON file: {exp_params_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error loading {exp_params_path}: {e}. Skipping.")
                continue

            # Extract parameters from the 'full_config' dictionary
            # Use .get() for safety in case keys are missing
            full_config = params.get('full_config', {})
            if not full_config:
                print(f"Warning: 'full_config' not found in {exp_params_path}. Trying top-level keys.")
                full_config = params  # Try using the top-level dict if full_config is missing

            # --- Extract required parameters ---
            try:
                aggregation_method = full_config.get('aggregation_method', 'N/A')
                data_split_config = full_config.get('data_split', {})
                adv_rate = data_split_config.get('adv_rate', 0.0)
                n_sellers = data_split_config.get('num_sellers', 'N/A')
                data_split_mode = data_split_config.get('data_split_mode', 'N/A')

                training_config = full_config.get('training', {})
                local_training_params = training_config.get('local_training_params', {})
                local_epoch = local_training_params.get('local_epochs', 'N/A')
                local_lr = local_training_params.get('learning_rate', 'N/A')  # Assuming you might need this later

                attack_config = full_config.get('attack', {})
                attack_enabled = attack_config.get('enabled', False)
                gradient_manipulation_mode = attack_config.get('gradient_manipulation_mode',
                                                               'None') if attack_enabled else 'None'
                trigger_type = attack_config.get('trigger_type', 'N/A')
                poison_strength = attack_config.get('poison_strength', 0) if attack_enabled else 0
                trigger_rate = attack_config.get('poison_rate', 0.0) if attack_enabled else 0.0

                sybil_config = full_config.get('sybil', {})
                is_sybil_bool = sybil_config.get('is_sybil', False)
                # Replicate original logic for IS_SYBIL string representation
                is_sybil_str = sybil_config.get('sybil_mode', 'default') if is_sybil_bool else "False"
                # trigger_attack_mode (originally sybil.trigger_mode)
                trigger_attack_mode = sybil_config.get('trigger_mode', 'always_on')  # Map sybil trigger_mode

                federated_config = full_config.get('federated_learning', {})
                change_base = str(federated_config.get('change_base', False))  # Convert bool to string like original

                dataset_name = full_config.get('dataset_name', 'N/A')
                exp_name = full_config.get('exp_name', '')  # Get exp_name if needed

                dm_params = data_split_config.get('dm_params', {})
                discovery_quality = str(
                    dm_params.get('discovery_quality', 'N/A')) if data_split_mode == "discovery" else 'N/A'
                buyer_data_mode = dm_params.get('buyer_data_mode', 'N/A') if data_split_mode == "discovery" else 'N/A'

            except KeyError as e:
                print(f"Missing expected key {e} in {exp_params_path}. Skipping experiment.")
                continue
            except Exception as e:
                print(f"Error extracting parameters from {exp_params_path}: {e}. Skipping experiment.")
                continue

            # --- Apply Filters (Optional) ---
            if filter_params:
                skip_experiment = False
                for key, allowed_values in filter_params.items():
                    # Extract the actual value from the loaded params based on the key
                    actual_value = None
                    if key == 'aggregation_method':
                        actual_value = aggregation_method
                    elif key == 'adv_rate':
                        actual_value = adv_rate
                    elif key == 'is_sybil':
                        actual_value = is_sybil_str  # Use the string representation
                    elif key == 'trigger_rate':
                        actual_value = trigger_rate
                    elif key == 'change_base':
                        actual_value = change_base
                    # Add more elif conditions for other parameters you want to filter on
                    else:
                        print(f"Warning: Unknown filter key '{key}'. Skipping this filter.")
                        continue

                    if actual_value not in allowed_values:
                        print(
                            f"Skipping experiment {root} due to filter: {key}={actual_value} (not in {allowed_values})")
                        skip_experiment = True
                        break  # No need to check other filters for this experiment
                if skip_experiment:
                    continue

            # --- Construct parameter dicts for process_single_experiment ---
            # These are based on the structure expected by your original call
            attack_params = {
                'ATTACK_METHOD': gradient_manipulation_mode,
                'TRIGGER_RATE': trigger_rate,
                'IS_SYBIL': is_sybil_str,
                'ADV_RATE': adv_rate,  # Included here as per original structure, though also passed separately
                'CHANGE_BASE': change_base,  # Included here as per original structure
                'TRIGGER_MODE': trigger_attack_mode,  # Mapped from sybil.trigger_mode
                # 'benign_rounds': sybil_config.get('benign_rounds'), # Add if needed by process_single_experiment
                "poison_strength": poison_strength,  # Add if needed
            }

            if data_split_mode == "discovery":
                market_params = {
                    'AGGREGATION_METHOD': aggregation_method,
                    'DATA_SPLIT_MODE': data_split_mode,
                    "discovery_quality": discovery_quality,
                    "buyer_data_mode": buyer_data_mode,
                    'N_CLIENTS': n_sellers,
                    'LOCAL_EPOCH': local_epoch,  # Adding info that might be useful
                    'DATASET': dataset_name,
                }
            else:
                market_params = {
                    'AGGREGATION_METHOD': aggregation_method,
                    'DATA_SPLIT_MODE': data_split_mode,
                    'N_CLIENTS': n_sellers,
                    'LOCAL_EPOCH': local_epoch,
                    'DATASET': dataset_name,
                }

            # --- Find and Process Runs ---
            run_paths = sorted(glob.glob(os.path.join(root, "run_*")))
            if not run_paths:
                print(f"No 'run_*' directories found in: {root}")
                continue

            print(f"Processing {len(run_paths)} runs for experiment: {root}")
            aggregated_processed_data = []
            aggregated_summaries = []
            run_cnt = 0
            for run_path in run_paths:
                if not os.path.isdir(run_path):  # Skip if it's not a directory
                    continue

                file_path = os.path.join(run_path, "market_log.ckpt")
                data_statistics_path = os.path.join(run_path, "data_statistics.json")  # Assumed name

                if not os.path.exists(file_path):
                    # print(f"File not found: {file_path}") # Reduce verbosity
                    continue

                # print(f"Processing: {file_path}") # Reduce verbosity

                try:
                    # Call the existing processing function
                    processed_data, summary = process_single_experiment(
                        file_path=file_path,
                        attack_params=attack_params,
                        market_params=market_params,
                        data_statistics_path=data_statistics_path if os.path.exists(data_statistics_path) else None,
                        adv_rate=adv_rate,  # Pass adv_rate separately as original code did
                        cur_run=run_cnt
                    )
                    run_cnt += 1
                    if processed_data:
                        aggregated_processed_data.extend(processed_data)
                    if summary:
                        aggregated_summaries.append(summary)
                except Exception as e:
                    print(f"Error processing run {run_path}: {e}")
                    # Decide if you want to continue with other runs or skip the experiment

            # --- Aggregate results for this experiment ---
            if aggregated_summaries:
                try:
                    # Ensure identifying parameters are consistent before averaging
                    # (e.g., copy from the first summary to the avg_summary)
                    base_summary_info = aggregated_summaries[0]  # Get first summary as template
                    avg_summary = average_dicts(aggregated_summaries)
                    # Add back identifying information that might be lost in averaging
                    for key, val in base_summary_info.items():
                        if key not in avg_summary or not isinstance(val, (int, float)):
                            avg_summary[key] = val
                    # Explicitly add key identifiers from params
                    avg_summary.update({
                        'AGGREGATION_METHOD': aggregation_method,
                        'ADV_RATE': adv_rate,
                        'IS_SYBIL': is_sybil_str,
                        'TRIGGER_RATE': trigger_rate,
                        'CHANGE_BASE': change_base,
                        'DATA_SPLIT_MODE': data_split_mode,
                        'discovery_quality': discovery_quality,
                        'buyer_data_mode': buyer_data_mode,
                        'exp_path': root  # Add experiment path for reference
                    })

                    print(
                        f"Avg Summary for {root}: { {k: v for k, v in avg_summary.items() if isinstance(v, (int, float, str))} }")  # Print concise summary
                    all_summary_data_avg.append(avg_summary)
                except Exception as e:
                    print(f"Error averaging summaries for {root}: {e}")

            all_processed_data.extend(aggregated_processed_data)
            all_summary_data.extend(aggregated_summaries)  # Still collect individual run summaries

    # --- Save final aggregated results ---
    if not all_processed_data:
        print("No data processed.")
        return None, None, None

    # Convert to DataFrames
    try:
        all_rounds_df = pd.DataFrame(all_processed_data)
    except Exception as e:
        print(f"Error creating all_rounds DataFrame: {e}")
        all_rounds_df = pd.DataFrame()  # Create empty df

    try:
        summary_df_avg = pd.DataFrame(all_summary_data_avg)
    except Exception as e:
        print(f"Error creating summary_avg DataFrame: {e}")
        summary_df_avg = pd.DataFrame()

    try:
        summary_data_df = pd.DataFrame(all_summary_data)
    except Exception as e:
        print(f"Error creating summary DataFrame: {e}")
        summary_data_df = pd.DataFrame()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    all_rounds_csv = os.path.join(output_dir, "all_rounds.csv")
    summary_csv_avg = os.path.join(output_dir, "summary_avg.csv")
    summary_csv = os.path.join(output_dir, "summary.csv")

    if not all_rounds_df.empty:
        all_rounds_df.to_csv(all_rounds_csv, index=False)
        print(f"Saved all rounds data to {all_rounds_csv}")
    else:
        print("No all_rounds data to save.")

    if not summary_df_avg.empty:
        summary_df_avg.to_csv(summary_csv_avg, index=False)
        print(f"Saved average summary data to {summary_csv_avg}")
    else:
        print("No average summary data to save.")

    if not summary_data_df.empty:
        summary_data_df.to_csv(summary_csv, index=False)
        print(f"Saved individual run summary data to {summary_csv}")
    else:
        print("No individual run summary data to save.")

    # Return DataFrames for further analysis if desired
    return all_rounds_df, summary_df_avg, summary_data_df


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
