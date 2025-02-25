import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def process_single_experiment(file_path, attack_params):
    """
    Process a single experiment file and extract metrics.

    Args:
        file_path: Path to the market_log.ckpt file
        attack_params: Dictionary containing attack parameters

    Returns:
        processed_data: List of dictionaries with processed round data
        summary_data: Dictionary with summary metrics
    """
    try:
        # Load the experiment data
        experiment_data = torch.load(file_path, map_location=torch.device('cpu'))

        # Extract round records
        round_records = experiment_data.get('round_records', [])
        if not round_records:
            print(f"Warning: No round records found in {file_path}")
            return [], {}

        processed_data = []

        for i, record in enumerate(round_records):
            round_num = record.get('round_number', i)

            # Extract basic round info
            round_data = {
                'round': round_num,
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

            processed_data.append(round_data)

        # Calculate summary metrics
        if processed_data:
            # Sort records by round
            sorted_records = sorted(processed_data, key=lambda x: x['round'])

            # Calculate summary metrics
            summary = {
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
                'ASR_PER_ADV': (sorted_records[-1].get('asr', 0) or 0) / attack_params['N_ADV'] if attack_params[
                                                                                                       'N_ADV'] > 0 else 0,

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


def get_save_path(n_sellers, n_adversaries, local_epoch, local_lr, gradient_manipulation_mode,
                  poison_strength, trigger_type, is_sybil, trigger_rate,
                  aggregation_method='martfl', dataset_name='FMNIST', sybil_mode='mimic'):
    """
    Construct a save path based on the experiment parameters.
    This is a copy of your function.
    """
    # Use is_sybil flag or, if not true, use sybil_mode
    sybil_str = str(sybil_mode) if is_sybil else False
    base_dir = Path("./results") / f"is_sybil_{sybil_str}" / "backdoor" / aggregation_method / dataset_name

    if gradient_manipulation_mode == "None":
        subfolder = "no_attack"
        param_str = f"n_seller_{n_sellers}_local_epoch_{local_epoch}_local_lr_{local_lr}"
    elif gradient_manipulation_mode == "cmd":
        subfolder = f"backdoor_mode_{gradient_manipulation_mode}_strength_{poison_strength}_trigger_type_{trigger_type}"
        param_str = f"n_seller_{n_sellers}_n_adv_{n_adversaries}_local_epoch_{local_epoch}_local_lr_{local_lr}"
    elif gradient_manipulation_mode == "single":
        subfolder = f"backdoor_mode_{gradient_manipulation_mode}_trigger_rate_{trigger_rate}_trigger_type_{trigger_type}"
        param_str = f"n_seller_{n_sellers}_n_adv_{n_adversaries}_local_epoch_{local_epoch}_local_lr_{local_lr}"
    else:
        raise NotImplementedError(f"No such attack type: {gradient_manipulation_mode}")

    # Construct the full save path
    save_path = base_dir / subfolder / param_str
    return str(save_path)


def process_all_experiments(output_dir='./processed_data', local_epoch=5):
    """
    Process all experiment files using the same parameter combinations as in your loop.
    """
    all_processed_data = []
    all_summary_data = []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use the same parameter combinations as in your loop
    for aggregation_method in ["martfl", "fedavg"]:
        for grad_mode in ['cmd', 'single']:
            for trigger_rate in [0.1, 0.5, 0.7]:
                for poison_strength in [0.1, 0.5, 1.0]:
                    for is_sybil in [False, True]:
                        for n_adv in [1, 3, 5]:
                            # Skip invalid combinations
                            if grad_mode == 'single' and poison_strength != 0.1:
                                continue
                            if trigger_rate == 0.7 and grad_mode == 'cmd':
                                continue

                            # Get the file path
                            save_path = get_save_path(
                                n_sellers=10,
                                n_adversaries=n_adv,
                                local_epoch=local_epoch,
                                local_lr=1e-2,
                                gradient_manipulation_mode=grad_mode,
                                poison_strength=poison_strength,
                                trigger_type="blended_patch",
                                is_sybil=is_sybil,
                                trigger_rate=trigger_rate,
                                aggregation_method=aggregation_method
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
                                'POISON_STRENGTH': poison_strength,
                                'IS_SYBIL': is_sybil,
                                'N_ADV': n_adv
                            }

                            # Process the file
                            processed_data, summary = process_single_experiment(file_path, attack_params)

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
    # Process all experiments
    all_rounds_df, summary_df = process_all_experiments()

    # Print summary statistics
    if not summary_df.empty:
        print("\nSummary Statistics:")
        print(f"Total experiments processed: {len(summary_df)}")
        print(f"Average Final ASR: {summary_df['FINAL_ASR'].mean():.4f}")
        print(f"Average Main Accuracy: {summary_df['FINAL_MAIN_ACC'].mean():.4f}")

        # Group by attack type
        for grad_mode in summary_df['GRAD_MODE'].unique():
            grad_data = summary_df[summary_df['GRAD_MODE'] == grad_mode]
            print(f"\nGradient Mode: {grad_mode}")
            print(f"  Average ASR: {grad_data['FINAL_ASR'].mean():.4f}")
            print(f"  Average Main Accuracy: {grad_data['FINAL_MAIN_ACC'].mean():.4f}")

        # Group by Sybil mode
        for is_sybil in summary_df['IS_SYBIL'].unique():
            sybil_data = summary_df[summary_df['IS_SYBIL'] == is_sybil]
            print(f"\nSybil Mode: {is_sybil}")
            print(f"  Average ASR: {sybil_data['FINAL_ASR'].mean():.4f}")
            print(f"  Average Main Accuracy: {sybil_data['FINAL_MAIN_ACC'].mean():.4f}")
