import argparse
import random

import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader

from attack.attack_gradient_market.poison_attack.attack_martfl import BackdoorImageGenerator
from marketplace.data_manager import DatasetManager
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller
from marketplace.seller.seller import BaseSeller
from marketplace.utils.gradient_market_utils.data_processor import get_data_set
from model.utils import get_model


def dataloader_to_tensors(dataloader):
    """
    Convert a DataLoader to tensors X (features) and y (labels).

    :param dataloader: PyTorch DataLoader object.
    :return: Tuple of torch.Tensors (X, y).
    """
    X_list, y_list = [], []

    for batch in dataloader:
        X_batch, y_batch = batch
        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all batches into single tensors
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y


def generate_attack_test_set(full_dataset, backdoor_generator, n_samples=1000):
    sample_indices = random.sample(range(len(full_dataset)), n_samples)
    subset_dataset = Subset(full_dataset, sample_indices)

    # ---------------------------
    # 2. Extract Images and Labels
    # ---------------------------
    # FashionMNIST images come in shape (1, H, W). For our backdoor generator,
    # assume we want images as (H, W, C). We can squeeze and then unsqueeze at the end.

    X_list, y_list = [], []
    for img, label in subset_dataset:
        # img is a torch.Tensor of shape (1, H, W); convert to (H, W, 1)
        img = img.permute(1, 2, 0)  # now shape (H, W, C)
        X_list.append(img)
        y_list.append(label)

    X = torch.stack(X_list)  # Shape: (10000, H, W, C)
    y = torch.tensor(y_list)  # Shape: (10000,)

    # ---------------------------
    # 3. Generate Poisoned Dataset
    # ---------------------------
    # Assuming your backdoor generator has a method generate_poisoned_dataset that takes
    # torch.Tensors in shape (N, H, W, C) and returns (X_poisoned, y_poisoned)
    # with the same shape for X_poisoned.

    # For example, if your backdoor generator is an instance of AdvancedBackdoorAttack:
    # backdoor_generator = AdvancedBackdoorAttack(trigger_pattern=..., target_label=..., alpha=0.1, ...)

    X_poisoned, y_poisoned = backdoor_generator.generate_poisoned_dataset(X, y, poison_rate=0.1)

    # ---------------------------
    # 4. Build DataLoaders
    # ---------------------------
    # Many PyTorch models expect image tensors to be in shape (N, C, H, W), so we permute.
    X = X.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    X_poisoned = X_poisoned.permute(0, 3, 1, 2)

    clean_dataset = TensorDataset(X, y)
    triggered_dataset = TensorDataset(X_poisoned, y_poisoned)

    batch_size = 64
    clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    triggered_loader = DataLoader(triggered_dataset, batch_size=batch_size, shuffle=True)
    return clean_loader, triggered_loader


def backdoor_attack(dataset_name, n_sellers, n_adversaries, model_structure,
                    global_rounds=100, backdoor_target_label=0, trigger_type: str = "blended_patch", exp_name="/"):
    # load the dataset
    save_path = f"/results/{exp_name}/"
    loss_fn = nn.CrossEntropyLoss()
    backdoor_generator = BackdoorImageGenerator(backdoor_target_label)

    # setup buyers, only one buyer per query. Set buyer cid as 0 for data split
    n_buyer = 1
    buyer_cid = 0

    # set up the data set for the participants
    client_loaders, full_dataset = get_data_set(dataset_name, num_clients=n_sellers + n_buyer, iid=True)

    # config the buyer
    buyer = GradientSeller(seller_id="buyer", local_data=client_loaders[buyer_cid].dataset, dataset_name=dataset_name,
                           save_path=save_path)

    # config the marketplace
    aggregator = Aggregator(save_path=save_path,
                            n_seller=n_sellers,
                            n_adversaries=n_adversaries,
                            model_structure=model_structure,
                            dataset_name=dataset_name,
                            quantization=False,
                            )
    marketplace = DataMarketplaceFederated(aggregator,
                                           selection_method="martfl")

    # config the seller and register to the marketplace
    for cid, loader in client_loaders.items():
        if cid == buyer_cid:
            continue
        if cid <= n_adversaries:
            cur_id = f"adv_{cid}"
            current_seller = AdvancedBackdoorAdversarySeller(seller_id=cur_id,
                                                             local_data=loader.dataset,
                                                             target_label=backdoor_target_label,
                                                             trigger_type=trigger_type, save_path=save_path,
                                                             backdoor_generator=backdoor_generator
                                                             )
        else:
            cur_id = f"seller_{cid}"
            current_seller = GradientSeller(seller_id=cur_id, local_data=loader.dataset,
                                            dataset_name=dataset_name, save_path=save_path)
        marketplace.register_seller(cur_id, current_seller)

    # config the attack test set.
    clean_loader, triggered_loader = generate_attack_test_set(full_dataset, backdoor_generator, 10000)

    # Start gloal round
    for gr in range(global_rounds):
        # compute the buyer gradient as the reference point
        buyer_gradient = buyer.get_gradient()
        # train the attack model
        marketplace.train_federated_round(round_number=gr,
                                          buyer_gradient=buyer_gradient,
                                          test_dataloader_buyer_local=client_loaders[buyer_cid],
                                          test_dataloader_global=clean_loader,
                                          clean_loader=clean_loader, triggered_loader=triggered_loader,
                                          loss_fn=loss_fn)

    # post fl process, test the final model.
    aggregator.global_model
    # record the result for each seller
    for s in seller:
        s.save_statistics()

    # record the attack result for the final round

    return eval_results


def setup(data_manager: DatasetManager, adversary_ratio=0.25, seller_configs=None):
    """Setup marketplace with normal and adversarial sellers"""

    # Create marketplace
    marketplace = DataMarketplaceData()
    # Get data allocations

    allocations = data_manager.allocate_data_to_sellers(
        seller_configs,
        adversary_ratio=adversary_ratio
    )
    seller_dict = {}
    print(f"Current seller info {seller_dict}")
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
    parser = argparse.ArgumentParser(description="Run Backdoor Attack Experiment")

    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., MNIST, CIFAR10)')
    parser.add_argument('--n_sellers', type=int, required=True, help='Number of sellers')
    parser.add_argument('--n_adversaries', type=int, required=True, help='Number of adversaries')

    # Optional arguments with defaults
    parser.add_argument('--global_rounds', type=int, default=100, help='Number of global training rounds')
    parser.add_argument('--backdoor_target_label', type=int, default=0, help='Target label for backdoor attack')
    parser.add_argument('--trigger_type', type=str, default="blended_patch", help='Type of backdoor trigger')
    parser.add_argument('--exp_name', type=str, default="/", help='Experiment name for logging')

    # Model architecture argument
    parser.add_argument('--model_arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'mlp'],
                        help='Model architecture (resnet18, resnet34, mlp)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    t_model = get_model(args.dataset_name)

    eval_results = backdoor_attack(
        dataset_name=args.dataset_name,
        n_sellers=args.n_sellers,
        n_adversaries=args.n_adversaries,
        model_structure=t_model,
        global_rounds=args.global_rounds,
        backdoor_target_label=args.backdoor_target_label,
        trigger_type=args.trigger_type,
        exp_name=args.exp_name
    )

    print("Evaluation Results:", eval_results)
