import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch


def process_single_experiment(file_path, attack_params, aggregation_method):
    """
    Process a single experiment file and extract metrics.

    Args:
        file_path: Path to the market_log.ckpt file
        attack_params: Dictionary containing attack parameters
        aggregation_method: The aggregation method used in the experiment

    Returns:
        processed_data: List of dictionaries with processed round data
        summary_data: Dictionary with summary metrics
    """
    try:
        # Load the experiment data
        experiment_data = torch.load(file_path, map_location=torch.device('cpu'))

        # Extract round records
        round_records = experiment_data
        if not round_records:
            print(f"Warning: No round records found in {file_path}")
            return [], {}

        processed_data = []

        for i, record in enumerate(round_records):
            round_num = record.get('round_number', i)

            # Extract basic round info
            round_data = {
                'round': round_num,
                'AGGREGATION_METHOD': aggregation_method,  # Add aggregation method
                **attack_params  # Include attack parameters
            }

            # Extract performance metrics
            if 'final_perf_global' in record and record['final_perf_global'] is not None:
                round_data['main_acc'] = record['final_perf_global'].get('acc', None)
                round_data['main_loss'] = record['final_perf_global'].get('loss', None)

            # Extract backdoor attack metrics
            extra_info = record.get('extra_info', {})
            if 'poison_metrics' in extra_info:
                poison_metrics = extra_info['poison_metrics']
                round_data['clean_acc'] = poison_metrics.get('clean_accuracy', None)
                round_data['triggered_acc'] = poison_metrics.get('triggered_accuracy', None)
                round_data['asr'] = poison_metrics.get('attack_success_rate', None)

            # Extract selection rates
            if 'selection_rate_info' in record and record['selection_rate_info']:
                selection_info = record.get("selection_rate_info", {})
                round_data['malicious_rate'] = selection_info.get('malicious_rate', None)
                round_data['benign_rate'] = selection_info.get('benign_rate', None)
                round_data['avg_malicious_rate'] = selection_info.get('avg_malicious_rate', None)
                round_data['avg_benign_rate'] = selection_info.get('avg_benign_rate', None)

            if 'martfl_baseline_id' in record:
                round_baseline_id = record.get("martfl_baseline_id", None)

                round_data["baseline_client_id"] = round_baseline_id
                if round_baseline_id:
                    round_data["malicious_baseline"] = isinstance(round_baseline_id, str)
            round_data['selected_clients'] = record["used_sellers"]
            round_data['outlier_clients'] = record["outlier_ids"]
            round_data['n_selected_clients'] = record["num_sellers_selected"]
            processed_data.append(round_data)

        # Calculate summary metrics
        if processed_data:
            # Sort records by round
            sorted_records = sorted(processed_data, key=lambda x: x['round'])

            # Calculate summary metrics
            summary = {
                'AGGREGATION_METHOD': aggregation_method,  # Add aggregation method
                **attack_params,

                # Calculate max ASR achieved during training
                'MAX_ASR': max([r.get('asr', 0) or 0 for r in sorted_records]),

                # Calculate final metrics (last round)
                'FINAL_ASR': sorted_records[-1].get('asr', None),
                'FINAL_MAIN_ACC': sorted_records[-1].get('main_acc', None),
                'FINAL_CLEAN_ACC': sorted_records[-1].get('clean_acc', None),
                'FINAL_TRIGGERED_ACC': sorted_records[-1].get('triggered_acc', None),

                # Calculate ASR at different stages of training
                'ASR_25PCT': next(
                    (r.get('asr', None) for r in sorted_records if r['round'] >= len(sorted_records) * 0.25), None),
                'ASR_50PCT': next(
                    (r.get('asr', None) for r in sorted_records if r['round'] >= len(sorted_records) * 0.5), None),
                'ASR_75PCT': next(
                    (r.get('asr', None) for r in sorted_records if r['round'] >= len(sorted_records) * 0.75), None),

                # Calculate rounds to reach specific ASR thresholds
                'ROUNDS_TO_50PCT_ASR': next((r['round'] for r in sorted_records if (r.get('asr', 0) or 0) >= 0.5),
                                            float('inf')),
                'ROUNDS_TO_75PCT_ASR': next((r['round'] for r in sorted_records if (r.get('asr', 0) or 0) >= 0.75),
                                            float('inf')),
                'ROUNDS_TO_90PCT_ASR': next((r['round'] for r in sorted_records if (r.get('asr', 0) or 0) >= 0.9),
                                            float('inf')),

                # Calculate average selection rates
                'AVG_MALICIOUS_RATE': np.mean([r.get('malicious_rate', 0) or 0 for r in sorted_records]),
                'AVG_BENIGN_RATE': np.mean([r.get('benign_rate', 0) or 0 for r in sorted_records]),

                # Calculate attack efficiency
                'ASR_PER_ADV': (sorted_records[-1].get('asr', 0) or 0) / attack_params['ADV_RATE'] if attack_params[
                                                                                                          'ADV_RATE'] > 0 else 0,

                # Calculate stealth (1 - abs difference between clean and final accuracy)
                'STEALTH': 1 - abs(
                    (sorted_records[-1].get('main_acc', 0) or 0) - (sorted_records[-1].get('clean_acc', 0) or 0)),

                # Total rounds
                'TOTAL_ROUNDS': len(sorted_records)
            }

            # Handle infinite values for CSV export
            for key, value in summary.items():
                if value == float('inf'):
                    summary[key] = -1  # Use -1 to represent "never reached threshold"

            return processed_data, summary
        else:
            return [], {}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print("Traceback details:")
        traceback.print_exc()
        return [], {}


def get_save_path(n_sellers, local_epoch, local_lr, gradient_manipulation_mode,
                  sybil_mode=False, is_sybil="False", data_split_mode='iid',
                  aggregation_method='fedavg', dataset_name='cifar10',
                  poison_strength=None, trigger_rate=None, trigger_type=None,
                  adv_rate=None, change_base="True", trigger_attack_mode=""):
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
            "./results") / f"backdoor_trigger_{trigger_attack_mode}" / f"is_sybil_{sybil_str}" / f"is_iid_{data_split_mode}" / f"{aggregation_method}_{change_base}" / dataset_name
    else:
        base_dir = Path(
            "./results") / f"backdoor_trigger_{trigger_attack_mode}" / f"is_sybil_{sybil_str}" / f"is_iid_{data_split_mode}" / aggregation_method / dataset_name

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

    # Construct the full save path
    save_path = base_dir / subfolder / param_str
    return str(save_path)


def process_all_experiments(output_dir='./processed_data', local_epoch=2,
                            aggregation_methods=['martfl', 'fedavg']):
    """
    Process all experiment files for multiple aggregation methods.

    Args:
        output_dir: Directory to save processed data
        local_epoch: Local epoch setting used in experiments
        aggregation_methods: List of aggregation methods to process
    """
    all_processed_data = []
    all_summary_data = []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each aggregation method
    for aggregation_method in aggregation_methods:
        print(f"\nProcessing experiments for {aggregation_method}...")
        # for data_split_mode in ["NonIID", "IID"]:
        for data_split_mode in ["dirichlet", "adversaryfirst"]:
            for grad_mode in ['single', 'None']:
                for trigger_attack_mode in ['static', 'dynamic']:
                    for trigger_rate in [0.25]:
                        for is_sybil in ["False", "mimic"]:
                            for adv_rate in [0.2, 0.3]:
                                for change_base in ["True", "False"]:
                                    if aggregation_method == "fedavg" and change_base == "True":
                                        continue
                                    # Get the file path
                                    save_path = get_save_path(
                                        n_sellers=30,

                                        adv_rate=adv_rate,
                                        local_epoch=local_epoch,
                                        local_lr=1e-2,
                                        gradient_manipulation_mode=grad_mode,
                                        poison_strength=0,
                                        trigger_type="blended_patch",
                                        is_sybil=is_sybil,
                                        trigger_rate=trigger_rate,
                                        aggregation_method=aggregation_method,
                                        data_split_mode=data_split_mode,
                                        change_base=change_base,
                                        dataset_name="FMNIST",
                                        trigger_attack_mode=trigger_attack_mode
                                    )

                                    # Construct the full file path
                                    file_path = f"{save_path}/market_log.ckpt"

                                    if not os.path.exists(file_path):
                                        print(f"File not found: {file_path}")
                                        continue

                                    print(f"Processing: {file_path}")

                                    # Extract attack parameters
                                    attack_params = {
                                        'ATTACK_METHOD': grad_mode,
                                        'TRIGGER_RATE': trigger_rate,
                                        'IS_SYBIL': is_sybil,
                                        'ADV_RATE': adv_rate,
                                        'CHANGE_BASE': change_base,
                                        'DATA_SPLIT_MODE': data_split_mode,
                                        'TRIGGER_MODE': trigger_attack_mode
                                    }

                                    # Process the file
                                    processed_data, summary = process_single_experiment(
                                        file_path,
                                        attack_params,
                                        aggregation_method
                                    )

                                    # Add to the overall data
                                    all_processed_data.extend(processed_data)
                                    if summary:
                                        all_summary_data.append(summary)

    # Convert to DataFrames
    all_rounds_df = pd.DataFrame(all_processed_data)
    summary_df = pd.DataFrame(all_summary_data)

    # Save to CSV
    if not all_rounds_df.empty:
        all_rounds_csv = f"{output_dir}/all_rounds.csv"
        all_rounds_df.to_csv(all_rounds_csv, index=False)
        print(f"Saved all rounds data to {all_rounds_csv}")

    if not summary_df.empty:
        summary_csv = f"{output_dir}/summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary data to {summary_csv}")

    return all_rounds_df, summary_df


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
    parser.add_argument("--aggregation_methods", nargs='+', default=['martfl', 'fedavg'],
                        help="List of aggregation methods to process")

    args = parser.parse_args()

    # Process all experiments
    all_rounds_df, summary_df = process_all_experiments(
        output_dir=args.output_dir,
        local_epoch=args.local_epoch,
        aggregation_methods=args.aggregation_methods
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
