import os
from pathlib import Path

import numpy as np
import pandas as pd
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
            if 'selection_rate_info' in record:
                selection_info = record['selection_rate_info']
                round_data['malicious_rate'] = selection_info.get('malicious_rate', None)
                round_data['benign_rate'] = selection_info.get('benign_rate', None)
                round_data['avg_malicious_rate'] = selection_info.get('avg_malicious_rate', None)
                round_data['avg_benign_rate'] = selection_info.get('avg_benign_rate', None)
            if 'martfl_baseline_id' in record:
                round_baseline_id = record['martfl_baseline_id']
                round_data["baseline_client_id"] = round_baseline_id
                if round_baseline_id:
                    round_data["malicious_baseline"] = isinstance(round_baseline_id, str)
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
        return [], {}


def get_save_path(n_sellers, local_epoch, local_lr, gradient_manipulation_mode,
                  sybil_mode=False, is_sybil=False, data_split_mode='iid',
                  aggregation_method='fedavg', dataset_name='cifar10',
                  poison_strength=None, trigger_rate=None, trigger_type=None,
                  adv_rate=None, change_base="True"):
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
    sybil_str = str(sybil_mode) if is_sybil else "False"

    if aggregation_method == "martfl":
        base_dir = Path(
            "./results") / "backdoor" / f"is_sybil_{sybil_str}" / f"is_iid_{data_split_mode}" / f"{aggregation_method}_{change_base}" / dataset_name
    else:
        base_dir = Path(
            "./results") / "backdoor" / f"is_sybil_{sybil_str}" / f"is_iid_{data_split_mode}" / aggregation_method / dataset_name

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
        for data_split_mode in ["NonIID", "IID"]:
            for grad_mode in ['single']:
                for trigger_rate in [0.25, 0.5]:
                    for is_sybil in [False, True]:
                        for adv_rate in [0.1, 0.2, 0.3, 0.4]:
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
                                    dataset_name="FMNIST"
                                )

                                # Construct the full file path
                                file_path = f"{save_path}/market_log.ckpt"

                                if not os.path.exists(file_path):
                                    print(f"File not found: {file_path}")
                                    continue

                                print(f"Processing: {file_path}")

                                # Extract attack parameters
                                attack_params = {
                                    'GRAD_MODE': grad_mode,
                                    'TRIGGER_RATE': trigger_rate,
                                    'IS_SYBIL': is_sybil,
                                    'ADV_RATE': adv_rate,
                                    'CHANGE_BASE': change_base,
                                    'IID': data_split_mode
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process federated learning backdoor attack logs")
    parser.add_argument("--output_dir", default="./processed_data", help="Output directory for processed data")
    parser.add_argument("--local_epoch", type=int, default=5, help="Local epoch setting used in experiments")
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
            for grad_mode in agg_data['GRAD_MODE'].unique():
                grad_data = agg_data[agg_data['GRAD_MODE'] == grad_mode]
                print(f"    Gradient Mode: {grad_mode}")
                print(f"      Average ASR: {grad_data['FINAL_ASR'].mean():.4f}")
                print(f"      Average Main Accuracy: {grad_data['FINAL_MAIN_ACC'].mean():.4f}")
