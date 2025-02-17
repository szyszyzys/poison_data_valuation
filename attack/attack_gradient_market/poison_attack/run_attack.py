import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

def run_attack_experiment(dataset, attack_type):
def setup():
    """Setup marketplace with normal and adversarial sellers"""

    # Create marketplace
    marketplace = DataMarketplaceData()
    # Get data allocations

    allocations = data_manager.allocate_data_to_sellers(
        seller_configs,
        adversary_ratio=adversary_ratio
    )
    seller_dict = {}
    # Create and register sellers
    for config in seller_configs:
        seller_id = config['id']
        seller_data = allocations[seller_id]

        if config['type'] == 'adversary':
            seller = MaliciousDataSeller(
                seller_id=seller_id,
                dataset=seller_data['X'],
            )
        else:
            seller = BaseSeller(
                seller_id=seller_id,
                dataset=seller_data['X']
            )
        seller_dict[seller_id] = seller
        marketplace.register_seller(seller_id, seller)

    return marketplace, seller_dict


def parse_args():
    """
    Parse command-line arguments for the experiment configuration.

    :return: Dictionary containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run attack experiment with configurable parameters.")

    # Add arguments with default values from exp_config
    parser.add_argument("--dataset", type=str, default="cifar", help="fmnist cifar")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device,
                        help="Device to run on, e.g. 'cuda' or 'cpu' (default: cuda if available)")

    parser.add_argument("--attack_type", type=str, default="backdoor",
                        help="Current Attack Type")

    parser.add_argument("--seller_configs", type=str, default="adv1:adversary",
                        help="Comma-separated list of seller configurations (format: id:type, e.g., adv1:adversary,normal1:normal)")

    args = parser.parse_args()

    # Process seller_configs into a list of dictionaries
    seller_configs = []
    if args.seller_configs:
        for seller in args.seller_configs.split(","):
            seller_id, seller_type = seller.split(":")
            seller_configs.append({'id': seller_id, 'type': seller_type})

    # Return as a dictionary
    return {
        "num_seller": args.num_seller,
        "num_buyer": args.num_buyer,
        "adversary_ratio": args.adversary_ratio,
        "seller_configs": seller_configs
    }, args


if __name__ == "__main__":
    # Parse command-line arguments
    param_mapping, args = parse_args()
    print("start reconstruction attack")
    print(param_mapping)
    work_path = f"/result/gradient_market/"

    # config the buyer


    # config the clients


    # Run experiment with parsed arguments
    eval_results = run_attack_experiment(
        dataset=args.dataset,
        attack_type = args.attack_type,
        device=args.device,
        **param_mapping )

    # Plot and print results
    # plot_results(eval_results, eval_range)
    # print(eval_results)
