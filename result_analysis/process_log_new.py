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

            selected_clients = (
                    record.get("used_sellers")  # old name
                    or record.get("selected_sellers")  # new name
                    or []
            )

            perf_global = (
                    record.get("final_perf_global")  # old name
                    or record.get("perf_global")  # new name
                    or {}
            )

            poison_metrics = (
                    record.get("extra_info", {}).get("poison_metrics")  # old structure
                    or {  # new structure: ASR is inside perf_global
                        "attack_success_rate": perf_global.get("attack_success_rate"),
                        "clean_accuracy": None,  # you removed these in new logs
                        "triggered_accuracy": None,
                    }
            )

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
            # final_perf = record.get('final_perf_global', {})
            # current_accuracy = final_perf.get('acc')
            # round_data['main_acc'] = current_accuracy
            # round_data['main_loss'] = final_perf.get('loss')
            current_accuracy = perf_global.get('accuracy')  # instead of final_perf['acc']
            round_data['main_acc'] = current_accuracy
            round_data['main_loss'] = perf_global.get('loss')
            round_data['asr'] = poison_metrics.get('attack_success_rate')

            poison_metrics = record.get('extra_info', {}).get('poison_metrics', {})
            round_data['clean_acc'] = poison_metrics.get('clean_accuracy')
            round_data['triggered_acc'] = poison_metrics.get('triggered_accuracy')

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
                       'MAX_ASR': max(asr_values) if asr_values else 0,
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
                                r['benign_gini_coefficient_in_round'])]),
                       'FINAL_MAIN_ACC': final_record.get('main_acc'),
                       'FINAL_CLEAN_ACC': final_record.get('clean_acc'),  # will be None for new logs
                       'FINAL_TRIGGERED_ACC': final_record.get('triggered_acc'),  # None for new logs
                       'FINAL_ASR': final_record.get('asr'),
                       }
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


# Assume process_single_experiment exists and has this signature:
# def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run):
#     # ... processes the data ...
#     # returns processed_data (list of dicts), summary (dict or None)
#     pass # Placeholder
# def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run,
#                               target_accuracy_for_coc=0.8, convergence_milestones=None):
#     """
#     Process a single experiment file and extract metrics, incorporating data distribution similarity and payment simulation.
#     """
#     try:
#         experiment_data = torch.load(file_path, map_location='cpu', weights_only=False)
#         data_stats = load_json(data_statistics_path)
#
#         buyer_distribution = data_stats['buyer_stats']['class_distribution']
#         seller_distributions = data_stats['seller_stats']  # Dict of seller_id_str -> stats
#
#         if not experiment_data:
#             print(f"Warning: No round records found in {file_path}")
#             return [], {}
#         if convergence_milestones is None:
#             convergence_milestones = [0.70, 0.75, 0.8, 0.85, 0.9]  # Default to a single 80% milestone if not provided
#         milestone_convergence_info = {acc: None for acc in convergence_milestones}
#         cumulative_cost_for_milestones_tracker = 0  # Single cumulative cost tracker for all milestones
#         processed_data = []
#         num_total_sellers = market_params["N_CLIENTS"]  # Get total number of sellers from market_params
#         num_adversaries = int(num_total_sellers * adv_rate)
#
#         # --- Payment/Incentive Simulation Variables ---
#         total_payments_per_seller = {str(i): 0 for i in range(num_total_sellers)}  # Store total payment for each seller
#         cost_of_convergence = None
#         target_accuracy_reached_round = -1
#         cumulative_cost_for_coc = 0
#         # ---
#         # For "No Attack" runs, we'll store selection counts for different hypothetical adversary group sizes
#         # Key: hypothetical_adv_rate_for_baseline (e.g., 0.1, 0.2), Value: list of per-round selection rates for that group
#         baseline_designated_group_selection_rates_this_run = {}
#         hypothetical_adv_rates_for_baselines = [0.1, 0.2, 0.3, 0.4]
#         baseline_designated_group_selection_rates_summary_collector = {}
#         run_attack_method = attack_params.get('ATTACK_METHOD', 'None')  # Or 'single', etc.
#         for i, record in enumerate(experiment_data):
#             round_num = record.get('round_number', i)
#
#             selected_clients = record.get("used_sellers", [])  # These are seller IDs (strings)
#
#             adversary_selections = [cid for cid in selected_clients if int(cid) < num_adversaries]
#             benign_selections = [cid for cid in selected_clients if int(cid) >= num_adversaries]
#
#             # --- Payment Calculation for the current round ---
#             cost_per_round = len(selected_clients)  # Since payment_per_selected_gradient is 1
#             for seller_id in selected_clients:
#                 if str(seller_id) in total_payments_per_seller:  # Ensure seller_id is a string for dict key
#                     total_payments_per_seller[str(seller_id)] += 1  # Payment of 1
#                 else:
#                     # This case should ideally not happen if total_payments_per_seller is initialized correctly
#                     print(f"Warning: Selected seller_id {seller_id} not in payment tracking. Initializing.")
#                     total_payments_per_seller[str(seller_id)] = 1
#             cumulative_cost_for_milestones_tracker += cost_per_round  # Update before checking milestones
#             selected_clients_int = [int(cid) for cid in selected_clients]  # Ensure integer IDs if not already
#             benign_selected_in_round = [cid_str for cid_str in selected_clients if int(cid_str) >= num_adversaries]
#             malicious_selected_in_round = [cid_str for cid_str in selected_clients if int(cid_str) < num_adversaries]
#
#             # 2. Calculate Benign Seller Selection Rate for the round
#             num_total_benign_sellers = num_total_sellers - num_adversaries
#             if num_total_benign_sellers > 0 and selected_clients:  # Avoid division by zero
#                 round_benign_selection_rate = len(benign_selected_in_round) / num_total_benign_sellers
#             else:
#                 round_benign_selection_rate = 0.0 if selected_clients else np.nan  # Or just 0.0
#
#             current_attack_method = attack_params.get('ATTACK_METHOD', 'None')  # Get from params like 'single', 'None'
#
#             malicious_selected_in_round_actual = []
#             benign_selected_in_round_actual = []
#             if current_attack_method != 'None' and current_attack_method != 'No Attack':  # If an attack is active
#                 for cid_str in selected_clients:
#                     if int(cid_str) < num_adversaries:
#                         malicious_selected_in_round_actual.append(cid_str)
#                     else:
#                         benign_selected_in_round_actual.append(cid_str)
#             else:  # No active attack in this run
#                 benign_selected_in_round_actual = list(selected_clients)
#
#             # 3. Calculate Gini Coefficient for payments ONLY to BENIGN sellers in this round
#             # This assumes you have payment information per round or can infer it.
#             # If payment is uniform (e.g., 1 per selected seller):
#             payments_to_benign_this_round = {seller_id_str: 0 for seller_id_str in
#                                              map(str, range(num_adversaries, num_total_sellers))}
#             for seller_id_str in benign_selected_in_round:
#                 payments_to_benign_this_round[seller_id_str] = 1  # Or actual payment value
#
#             all_benign_payments_array = np.array(list(payments_to_benign_this_round.values()))
#             round_benign_gini_coefficient = calculate_gini(all_benign_payments_array)  # Use your existing Gini function
#             round_data = {'run': cur_run, 'round': round_num, **attack_params, **market_params,
#                           'n_selected_clients': len(selected_clients), 'selected_clients': selected_clients,
#                           'adversary_selection_rate': len(adversary_selections) / len(
#                               selected_clients) if selected_clients else 0,
#                           'benign_selection_rate': len(benign_selections) / len(
#                               selected_clients) if selected_clients else 0, 'cost_per_round': cost_per_round,
#                           'benign_selection_rate_in_round': round_benign_selection_rate,
#                           'benign_gini_coefficient_in_round': round_benign_gini_coefficient,
#                           }
#
#             for hypo_adv_rate in hypothetical_adv_rates_for_baselines:
#                 hypo_adv_rate_key_str = f"{hypo_adv_rate:.1f}"
#                 round_data_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate_key_str}_ROUND'  # New key for per-round log
#
#                 if run_attack_method == 'None' or run_attack_method == 'No Attack':
#                     num_hypo_designated_malicious = int(num_total_sellers * hypo_adv_rate)
#                     if num_hypo_designated_malicious == 0:
#                         round_data[round_data_key] = 0.0  # Or np.nan if preferred for no group
#                         # Also collect for summary (will be averaged later)
#                         if hypo_adv_rate_key_str not in baseline_designated_group_selection_rates_summary_collector:
#                             baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str] = []
#                         baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str].append(0.0)
#                         continue
#
#                     selected_from_hypo_group_count = 0
#                     for cid_str in selected_clients:
#                         if int(cid_str) < num_hypo_designated_malicious:
#                             selected_from_hypo_group_count += 1
#
#                     rate_for_hypo_group_this_round = selected_from_hypo_group_count / len(
#                         selected_clients) if selected_clients else 0.0
#                     round_data[round_data_key] = rate_for_hypo_group_this_round
#
#                     # Collect for summary average calculation
#                     if hypo_adv_rate_key_str not in baseline_designated_group_selection_rates_summary_collector:
#                         baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str] = []
#                     baseline_designated_group_selection_rates_summary_collector[hypo_adv_rate_key_str].append(
#                         rate_for_hypo_group_this_round)
#                 else:  # If it's an ATTACK run, these per-round baseline metrics are not applicable
#                     round_data[round_data_key] = np.nan  # Store NaN in the per-round log
#             # Calculate distribution similarities
#
#             # --- Accuracy, ASR, and Convergence Milestone Check ---
#             final_perf = record.get('final_perf_global', {})
#             current_accuracy = final_perf.get('acc')
#             round_data['main_acc'] = current_accuracy
#             round_data['main_loss'] = final_perf.get('loss')
#             poison_metrics = record.get('extra_info', {}).get('poison_metrics', {})
#             round_data['clean_acc'] = poison_metrics.get('clean_accuracy')
#             round_data['triggered_acc'] = poison_metrics.get('triggered_accuracy')
#             round_data['asr'] = poison_metrics.get('attack_success_rate')
#
#             if current_accuracy is not None:
#                 for target_acc in convergence_milestones:
#                     if milestone_convergence_info[target_acc] is None and current_accuracy >= target_acc:
#                         milestone_convergence_info[target_acc] = {
#                             'round': round_num + 1,  # Record 1-indexed round
#                             'cost': cumulative_cost_for_milestones_tracker
#                         }
#
#             similarities = []
#             for cid_ in selected_clients:  # cid from selected_clients is already a string
#                 cid_str = str(cid_)
#                 if cid_str in seller_distributions:
#                     seller_dist = seller_distributions[cid_str]['class_distribution']
#                     similarities.append(calculate_distribution_similarity(buyer_distribution, seller_dist))
#                 else:
#                     print(f"Warning: Seller {cid_str} not found in data_statistics for similarity calculation.")
#                     print(seller_distributions.keys())
#                     similarities.append(0)  # Or handle as NaN or skip
#
#             round_data['avg_selected_data_distribution_similarity'] = np.mean(similarities) if similarities else 0
#
#             un_selected_similarities = []
#             all_seller_ids_str = [str(k) for k in
#                                   range(num_total_sellers)]  # Generate all possible seller IDs as strings
#             unselected_seller_ids_str = [sid for sid in all_seller_ids_str if sid not in selected_clients]
#
#             for cid_str in unselected_seller_ids_str:
#                 if cid_str in seller_distributions:
#                     seller_dist = seller_distributions[cid_str]['class_distribution']
#                     un_selected_similarities.append(calculate_distribution_similarity(buyer_distribution, seller_dist))
#                 else:
#                     # This might happen if N_CLIENTS in market_params is larger than actual sellers in data_stats
#                     # Or if seller IDs are not contiguous from 0.
#                     # print(f"Warning: Unselected seller {cid_str} not found in data_statistics for similarity.")
#                     pass  # Or append a default value if needed
#
#             round_data['avg_unselected_data_distribution_similarity'] = np.mean(
#                 un_selected_similarities) if un_selected_similarities else 0
#
#             final_perf = record.get('final_perf_global', {})
#             current_accuracy = final_perf.get('acc')
#             round_data['main_acc'] = current_accuracy
#             round_data['main_loss'] = final_perf.get('loss')
#
#             # --- Cost of Convergence (CoC) Calculation ---
#             if current_accuracy is not None and cost_of_convergence is None:
#                 cumulative_cost_for_coc += cost_per_round
#                 if current_accuracy >= target_accuracy_for_coc:
#                     cost_of_convergence = cumulative_cost_for_coc
#                     target_accuracy_reached_round = round_num
#             elif cost_of_convergence is None:  # If accuracy is None but CoC not yet met
#                 cumulative_cost_for_coc += cost_per_round
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
#             # --- Gini Coefficient Calculation ---
#             # Ensure all potential sellers are included, even if they received 0 payment
#             all_seller_payments = [total_payments_per_seller.get(str(i), 0) for i in range(num_total_sellers)]
#             payment_gini_coefficient = calculate_gini(np.array(all_seller_payments))
#
#             # If target accuracy was never reached, CoC remains None or could be set to total cost
#             final_cumulative_cost = sum(r['cost_per_round'] for r in sorted_records)
#             if cost_of_convergence is None:
#                 print(
#                     f"Warning: Target accuracy {target_accuracy_for_coc} for CoC not reached in {file_path}. CoC will be total cost or NaN.")
#                 # Option 1: Set CoC to total cost if not met (indicates high cost)
#                 # cost_of_convergence = final_cumulative_cost
#                 # Option 2: Leave as None or set to a specific indicator like np.nan
#                 cost_of_convergence = np.nan  # Or final_cumulative_cost if you prefer that interpretation
#
#             summary = {"run": cur_run, **market_params, **attack_params,
#                        'MAX_ASR': max(asr_values) if asr_values else 0, 'FINAL_ASR': final_record.get('asr'),
#                        'FINAL_MAIN_ACC': final_record.get('main_acc'), 'FINAL_CLEAN_ACC': final_record.get('clean_acc'),
#                        'FINAL_TRIGGERED_ACC': final_record.get('triggered_acc'),
#                        'AVG_SELECTED_DISTRIBUTION_SIMILARITY': np.mean(
#                            [r['avg_selected_data_distribution_similarity'] for r in sorted_records if
#                             'avg_selected_data_distribution_similarity' in r]),
#                        'AVG_UNSELECTED_DISTRIBUTION_SIMILARITY': np.mean(
#                            [r['avg_unselected_data_distribution_similarity'] for r in sorted_records if
#                             'avg_unselected_data_distribution_similarity' in r]),
#                        'AVG_ADVERSARY_SELECTION_RATE': np.mean(
#                            [r['adversary_selection_rate'] for r in sorted_records if 'adversary_selection_rate' in r]),
#                        'AVG_BENIGN_SELECTION_RATE': np.mean(
#                            [r['benign_selection_rate'] for r in sorted_records if 'benign_selection_rate' in r]),
#                        'AVG_COST_PER_ROUND': np.mean(
#                            [r['cost_per_round'] for r in sorted_records if 'cost_per_round' in r]),
#                        'COST_OF_CONVERGENCE': cost_of_convergence, 'TARGET_ACC_FOR_COC': target_accuracy_for_coc,
#                        'COC_TARGET_REACHED_ROUND': target_accuracy_reached_round,
#                        'PAYMENT_GINI_COEFFICIENT': payment_gini_coefficient, 'TOTAL_COST': final_cumulative_cost,
#                        'TOTAL_ROUNDS': len(sorted_records), 'AVG_BENIGN_SELLER_SELECTION_RATE': np.mean(
#                     [r['benign_selection_rate_in_round'] for r in sorted_records if
#                      'benign_selection_rate_in_round' in r and pd.notna(r['benign_selection_rate_in_round'])]),
#                        'AVG_BENIGN_PAYMENT_GINI': np.mean(
#                            [r['benign_gini_coefficient_in_round'] for r in sorted_records if
#                             'benign_gini_coefficient_in_round' in r and pd.notna(
#                                 r['benign_gini_coefficient_in_round'])])}
#             for hypo_adv_rate_key_str, rates_list in baseline_designated_group_selection_rates_summary_collector.items():
#                 summary_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate_key_str}'
#                 if rates_list:  # If list has rates (i.e., was a No Attack run and clients were selected)
#                     summary[summary_key] = np.mean(rates_list)
#                 else:  # Could be an attack run, or No Attack run with no selections/no group
#                     summary[summary_key] = np.nan
#
#             # Add convergence milestone info to summary
#             for target_acc_ms, info_ms in milestone_convergence_info.items():
#                 acc_label = f"{int(target_acc_ms * 100)}"
#                 summary[f'ROUNDS_TO_{acc_label}ACC'] = info_ms['round'] if info_ms else np.nan
#                 summary[f'COST_TO_{acc_label}ACC'] = info_ms['cost'] if info_ms else np.nan
#
#             # Ensure all potential baseline columns exist in all summaries, even if NaN (for Attack runs)
#             if run_attack_method != 'None' and run_attack_method != 'No Attack':
#                 for hypo_adv_rate in hypothetical_adv_rates_for_baselines:
#                     summary_key = f'NO_ATTACK_DESIG_MAL_SEL_RATE_{hypo_adv_rate:.1f}'
#                     if summary_key not in summary:  # If not added because it was an attack run
#                         summary[summary_key] = np.nan
#             return processed_data, summary
#         return [], {}
#
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         traceback.print_exc()
#         return [], {}


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

def process_all_experiments_revised(
        base_results_dir: str = "./experiment_results_revised",
        output_dir: str = "./processed_data_revised",
        filter_params: dict | None = None,
        verbose: bool = True,
):
    """
    Scan `base_results_dir` recursively, load each `experiment_params.json`,
    and aggregate per‑round & per‑run summaries so they match the format
    produced by the *old* `process_all_experiments` function.

    The only substantive differences from your previous draft are:

    1. `attack_params_dict`
       ────────────────
       • Renamed LOCAL_POISON_RATE → **TRIGGER_RATE**
       • Added **CHANGE_BASE**, **benign_rounds**, **trigger_mode**
       • Ensured ADV_RATE reflects the *effective* attack rate that is also
         forwarded as the `adv_rate` argument to `process_single_experiment`.

    2. `market_params_dict`
       ────────────────
       • Field names identical to the old code
       • Discovery‑specific keys inserted only when `DATA_SPLIT_MODE == "discovery"`

    3. The `adv_rate` forwarded to `process_single_experiment`
       is exactly the same value stored in `ATTACK_PARAMS['ADV_RATE']`.
    """

    all_processed_data, all_summary_data, all_summary_data_avg = [], [], []
    os.makedirs(output_dir, exist_ok=True)
    print(f"Scanning for experiments in: {os.path.abspath(base_results_dir)}")

    experiment_found_flag, processed_experiment_count = False, 0

    # ------------------------------------------------------------------ #
    #                           MAIN SCAN LOOP                           #
    # ------------------------------------------------------------------ #
    for root, _, files in os.walk(base_results_dir):
        if "experiment_params.json" not in files:
            continue

        experiment_found_flag = True
        exp_params_path = os.path.join(root, "experiment_params.json")
        if verbose:
            print(f"\nFound experiment parameters: {exp_params_path}")

        try:
            with open(exp_params_path, "r") as f:
                params = json.load(f)
        except Exception as e:
            print(f"  ERROR loading {exp_params_path}: {e}. Skipping.")
            continue

        # Prefer the flattened copy under "full_config" if present
        full_cfg = params.get("full_config", params)

        # --------------------------- helpers --------------------------- #
        def _get(cfg: dict, dotted: str, default="N/A"):
            cur = cfg
            for k in dotted.split("."):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        # ----------------------- pull parameters ---------------------- #
        # ─ market‑level
        agg_method = _get(full_cfg, "aggregation_method")
        data_split_mode = _get(full_cfg, "data_split.data_split_mode")
        n_clients = _get(full_cfg, "data_split.num_sellers", 100)
        local_epoch = _get(full_cfg, "training.local_training_params.local_epochs", 1)
        dataset_name = _get(full_cfg, "dataset_name", "UnknownDataset")

        # discovery‑specific
        discovery_quality = str(_get(full_cfg, "data_split.dm_params.discovery_quality", "N/A"))
        buyer_data_mode = _get(full_cfg, "data_split.dm_params.buyer_data_mode", "N/A")

        # ─ attack / sybil
        attack_enabled = _get(full_cfg, "attack.enabled", False)
        gradient_manip_mode = _get(full_cfg, "attack.gradient_manipulation_mode", "None")
        poison_rate = _get(full_cfg, "attack.poison_rate", 0.0)
        attack_objective = _get(full_cfg, "attack.attack_type", "backdoor")
        benign_rounds = _get(full_cfg, "sybil.benign_rounds", 0)
        trigger_mode = _get(full_cfg, "sybil.trigger_mode", "fixed")

        is_sybil_bool = _get(full_cfg, "sybil.is_sybil", False)
        sybil_mode = _get(full_cfg, "sybil.sybil_mode", "False")
        adv_rate_sybil_override = _get(full_cfg, "sybil.adv_rate", 0.0)
        adv_rate_split = _get(full_cfg, "data_split.adv_rate", 0.0)

        # match the old code’s logic for ADV_RATE
        effective_adv_rate = adv_rate_sybil_override if is_sybil_bool else adv_rate_split

        # Determine ATTACK_METHOD exactly like before
        if not attack_enabled or gradient_manip_mode.lower() in ("none", ""):
            attack_method = "None"  # <- must be the string your downstream code expects
            trigger_rate = 0.0
        else:
            attack_method = gradient_manip_mode  # e.g. "single"
            trigger_rate = poison_rate

        # ----------------------- optional filters --------------------- #
        if filter_params:
            filt_map = {
                "aggregation_method": agg_method,
                "adv_rate": effective_adv_rate,
                "attack_method": attack_method,
                "trigger_rate": trigger_rate,
                "is_sybil": sybil_mode if is_sybil_bool else "False",
                "attack_objective": attack_objective,
                "dataset_name": dataset_name,
                "discovery_quality": discovery_quality,
                "buyer_data_mode": buyer_data_mode,
                "data_split_mode": data_split_mode,
            }
            skip = False
            for k, allowed in filter_params.items():
                if str(filt_map.get(k)) not in {str(v) for v in allowed}:
                    skip = True
                    break
            if skip:
                if verbose:
                    print(f"    Skipped due to filter mismatch: {root}")
                continue

        if verbose:
            print(f"  Processing experiment: {root}")

        # -------------------- 1️⃣  attack_params ----------------------- #
        # attack_params_dict = {
        #     "ATTACK_METHOD": attack_method,
        #     "TRIGGER_RATE": trigger_rate,
        #     "IS_SYBIL": sybil_mode if is_sybil_bool else "False",
        #     "ADV_RATE": effective_adv_rate,
        #     "CHANGE_BASE": _get(full_cfg, "data_split.change_base", "False"),
        #     "TRIGGER_MODE": trigger_mode,
        #     "benign_rounds": benign_rounds,
        #     "trigger_mode": trigger_mode,  # (exact duplicate key kept for B/C)
        # }
        attack_params_dict = {
            "ATTACK_METHOD": attack_method,  # now "None" for baselines
            "TRIGGER_RATE": trigger_rate,
            "IS_SYBIL": sybil_mode if is_sybil_bool else "False",
            "ADV_RATE": effective_adv_rate,
            "CHANGE_BASE": _get(full_cfg, "data_split.change_base", "False"),
            "TRIGGER_MODE": trigger_mode,
            "benign_rounds": benign_rounds,
            "trigger_mode": trigger_mode,
            "attack_objective": attack_objective  # kept for backward compatibility
        }

        # -------------------- 2️⃣  market_params ----------------------- #
        market_params_dict = {
            "AGGREGATION_METHOD": agg_method,
            "DATA_SPLIT_MODE": data_split_mode,
            "N_CLIENTS": n_clients,
            "LOCAL_EPOCH": local_epoch,
            "DATASET": dataset_name,
        }
        if data_split_mode == "discovery":
            market_params_dict["discovery_quality"] = discovery_quality
            market_params_dict["buyer_data_mode"] = buyer_data_mode

        # ----------------------- find & process runs ------------------ #
        run_dirs = sorted(glob.glob(os.path.join(root, "run_*")))
        if not run_dirs:
            if verbose:
                print(f"    No 'run_*' directories in {root}")
            continue
        if verbose:
            print(f"    Found {len(run_dirs)} run(s).")

        cur_exp_processed, cur_exp_summaries = [], []
        for run_idx, run_dir in enumerate(run_dirs):
            log_file = os.path.join(run_dir, "market_log_final.ckpt")
            stats_file = os.path.join(run_dir, "data_statistics.json")
            if not os.path.exists(log_file):
                if verbose:
                    print(f"      Missing log: {log_file}")
                continue
            stats_path = stats_file if os.path.exists(stats_file) else None
            try:
                pd_run, summary = process_single_experiment(
                    file_path=log_file,
                    attack_params=attack_params_dict,
                    market_params=market_params_dict,
                    data_statistics_path=stats_path,
                    adv_rate=effective_adv_rate,  # ← identical to ATTACK_PARAMS['ADV_RATE']
                    cur_run=run_idx,
                )
                if pd_run:
                    cur_exp_processed.extend(pd_run)
                if summary:
                    cur_exp_summaries.append(summary)
            except Exception as e:
                print(f"    ERROR processing {run_dir}: {e}")
                traceback.print_exc()

        # ------------- aggregate summaries (per experiment) ----------- #
        if cur_exp_summaries:
            processed_experiment_count += 1
            all_summary_data.extend(cur_exp_summaries)

            avg_summary = average_dicts(cur_exp_summaries)
            # inject identifying keys (same keys as old code produced)
            avg_summary.update({
                "AGGREGATION_METHOD": agg_method,
                "DATASET": dataset_name,
                "ATTACK_METHOD": attack_method,
                "ADV_RATE": effective_adv_rate,
                "TRIGGER_RATE": trigger_rate,
                "IS_SYBIL": attack_params_dict["IS_SYBIL"],
                "DATA_SPLIT_MODE": data_split_mode,
                "discovery_quality": discovery_quality,
                "buyer_data_mode": buyer_data_mode,
                "N_CLIENTS": n_clients,
                "LOCAL_EPOCH": local_epoch,
                "exp_path": root,
                "attack_objective": attack_objective,
            })
            avg_summary.pop("run", None)  # per‑run field not relevant here
            all_summary_data_avg.append(avg_summary)

        if cur_exp_processed:
            all_processed_data.extend(cur_exp_processed)

    # ------------------------------------------------------------------ #
    #                          write CSV outputs                         #
    # ------------------------------------------------------------------ #
    if not experiment_found_flag:
        print("No 'experiment_params.json' files found.")
    if processed_experiment_count == 0:
        print("Found experiments, but no runs produced summaries.")

    df_all = pd.DataFrame(all_processed_data)
    df_avg = pd.DataFrame(all_summary_data_avg)
    df_runs = pd.DataFrame(all_summary_data)

    out_all, out_avg, out_indv = [
        os.path.join(output_dir, f) for f in
        ("all_rounds.csv", "summary_avg.csv", "summary_individual_runs.csv")
    ]

    if not df_all.empty:
        df_all.to_csv(out_all, index=False)
        print(f"Saved {len(df_all)} rows → {out_all}")
    else:
        print("No all_rounds data.")

    if not df_avg.empty:
        df_avg.to_csv(out_avg, index=False)
        print(f"Saved {len(df_avg)} rows → {out_avg}")
    else:
        print("No summary_avg data.")

    if not df_runs.empty:
        df_runs.to_csv(out_indv, index=False)
        print(f"Saved {len(df_runs)} rows → {out_indv}")
    else:
        print("No per‑run summary data.")

    return df_all, df_avg, df_runs


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
