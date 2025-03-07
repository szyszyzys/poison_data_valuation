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
    buyer_dist_array = np.array(list(buyer_distribution.values()))
    seller_dist_array = np.array(list(seller_distribution.values()))
    similarity = np.dot(buyer_dist_array, seller_dist_array) / (
            np.linalg.norm(buyer_dist_array) * np.linalg.norm(seller_dist_array)
    )
    return similarity


def process_single_experiment(file_path, attack_params, market_params, data_statistics_path, adv_rate, cur_run):
    """
    Process a single experiment file and extract metrics, incorporating data distribution similarity.

    Args:
        file_path: Path to the market_log.ckpt file
        attack_params: Dictionary containing attack parameters
        market_params: Dictionary containing market parameters
        data_statistics_path: Path to the data_statistics.json file
        adv_rate: Proportion of sellers considered adversaries

    Returns:
        processed_data: List of dictionaries with processed round data
        summary_data: Dictionary with summary metrics
    """
    try:
        experiment_data = torch.load(file_path, map_location='cpu')
        data_stats = load_json(data_statistics_path)

        buyer_distribution = data_stats['buyer_stats']['class_distribution']
        seller_distributions = data_stats['seller_stats']

        if not experiment_data:
            print(f"Warning: No round records found in {file_path}")
            return [], {}

        processed_data = []
        num_adversaries = int(len(seller_distributions) * adv_rate)

        for i, record in enumerate(experiment_data):
            round_num = record.get('round_number', i)

            selected_clients = record.get("used_sellers", [])
            adversary_selections = [cid for cid in selected_clients if int(cid) < num_adversaries]
            benign_selections = [cid for cid in selected_clients if int(cid) >= num_adversaries]

            round_data = {
                'run': cur_run,
                'round': round_num,
                **attack_params,
                **market_params,
                'n_selected_clients': len(selected_clients),
                'selected_clients': selected_clients,
                'adversary_selection_rate': len(adversary_selections) / len(
                    selected_clients) if selected_clients else 0,
                'benign_selection_rate': len(benign_selections) / len(selected_clients) if selected_clients else 0
            }
            similarities = [
                calculate_distribution_similarity(buyer_distribution,
                                                  seller_distributions[str(cid)]['class_distribution'])
                for cid in selected_clients
            ]

            round_data['avg_distribution_similarity'] = np.mean(similarities) if similarities else 0

            final_perf = record.get('final_perf_global', {})
            round_data['main_acc'] = final_perf.get('acc')
            round_data['main_loss'] = final_perf.get('loss')

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

            summary = {
                "run": cur_run,
                **market_params,
                **attack_params,
                'MAX_ASR': max(asr_values),
                'FINAL_ASR': final_record.get('asr'),
                'FINAL_MAIN_ACC': final_record.get('main_acc'),
                'FINAL_CLEAN_ACC': final_record.get('clean_acc'),
                'FINAL_TRIGGERED_ACC': final_record.get('triggered_acc'),
                'AVG_DISTRIBUTION_SIMILARITY': np.mean([r['avg_distribution_similarity'] for r in sorted_records]),
                'AVG_ADVERSARY_SELECTION_RATE': np.mean([r['adversary_selection_rate'] for r in sorted_records]),
                'AVG_BENIGN_SELECTION_RATE': np.mean([r['benign_selection_rate'] for r in sorted_records]),
                'TOTAL_ROUNDS': len(sorted_records)
            }

            return processed_data, summary

        return [], {}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        traceback.print_exc()
        return [], {}


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
            for grad_mode in ['single']:
                for trigger_attack_mode in ['static', 'dynamic']:
                    for trigger_rate in [0.1]:
                        for is_sybil in ["False", "mimic"]:
                            for adv_rate in [0.2, 0.3]:
                                for change_base in ["True", "False"]:
                                    for discovery_quality in [0.1, 1, 10]:
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
                                                    'ADV_RATE': params["sybil_params"]["adv_rate"],
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
                                                        "buyer_data_mode": params["dm_params"]["buyer_data_mode"]}
                                                else:
                                                    market_params = {
                                                        'AGGREGATION_METHOD': aggregation_method,
                                                        'DATA_SPLIT_MODE': data_split_mode,
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
                                                aggregated_processed_data.append(processed_data)
                                                if summary:
                                                    aggregated_summaries.append(summary)

                                                # Aggregate numeric fields safely:

                                            if aggregated_summaries:
                                                avg_summary = average_dicts(aggregated_summaries)
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
