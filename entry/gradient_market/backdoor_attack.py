import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader

from attack.attack_gradient_market.poison_attack.attack_martfl import BackdoorImageGenerator
from general_utils.file_utils import save_history_to_json
from marketplace.market.markplace_gradient import DataMarketplaceFederated
from marketplace.market_mechanism.martfl import Aggregator
from marketplace.seller.gradient_seller import GradientSeller, AdvancedBackdoorAdversarySeller
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
                    global_rounds=100, backdoor_target_label=0, trigger_type: str = "blended_patch", save_path="/",
                    device='cpu', poison_strength=1, poison_test_sample=100):
    # load the dataset
    loss_fn = nn.CrossEntropyLoss()
    backdoor_generator = BackdoorImageGenerator(trigger_type="blended_patch", target_label=backdoor_target_label,
                                                channels=1)

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
                                                             backdoor_generator=backdoor_generator,
                                                             device=device,
                                                             poison_strength=poison_strength,
                                                             dataset_name=dataset_name
                                                             )
        else:
            cur_id = f"seller_{cid}"
            current_seller = GradientSeller(seller_id=cur_id, local_data=loader.dataset,
                                            dataset_name=dataset_name, save_path=save_path, device=device)
        marketplace.register_seller(cur_id, current_seller)

    # config the attack test set.
    clean_loader, triggered_loader = generate_attack_test_set(full_dataset, backdoor_generator, poison_test_sample)

    # Start gloal round
    fl_record_list = []
    for gr in range(global_rounds):
        # compute the buyer gradient as the reference point
        # todo new baseline selection
        buyer_gradient = buyer.get_gradient()
        # train the attack model
        round_record = marketplace.train_federated_round(round_number=gr,
                                                         buyer_gradient=buyer_gradient,
                                                         test_dataloader_buyer_local=client_loaders[buyer_cid],
                                                         test_dataloader_global=clean_loader,
                                                         clean_loader=clean_loader, triggered_loader=triggered_loader,
                                                         loss_fn=loss_fn)

    # post fl process, test the final model.
    save_history_to_json(marketplace.round_logs, f"{save_path}/market_log.json")
    # record the result for each seller
    all_sellers = marketplace.get_all_sellers
    for seller_id, seller in all_sellers.items():
        save_history_to_json(seller.get_federated_history, f"{save_path}/local_log_{seller_id}.json")

    # record the attack result for the final round

    return eval_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Backdoor Attack Experiment")

    # Required arguments
    parser.add_argument('--dataset_name', type=str, default='FMINIST',
                        help='Name of the dataset (e.g., MNIST, CIFAR10)')
    parser.add_argument('--n_sellers', type=int, default=10, help='Number of sellers')
    parser.add_argument('--n_adversaries', type=int, default=1, help='Number of adversaries')

    # Optional arguments with defaults
    parser.add_argument('--global_rounds', type=int, default=1, help='Number of global training rounds')
    parser.add_argument('--backdoor_target_label', type=int, default=0, help='Target label for backdoor attack')
    parser.add_argument('--trigger_type', type=str, default="blended_patch", help='Type of backdoor trigger')
    parser.add_argument('--exp_name', type=str, default="/", help='Experiment name for logging')
    parser.add_argument('--poison_test_sample', type=int, default=1, help='Number of global training rounds')

    parser.add_argument('--poison_strength', type=float, default=1, help='Strength of poisoning')

    # Model architecture argument
    parser.add_argument('--model_arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'mlp'],
                        help='Model architecture (resnet18, resnet34, mlp)')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1').")

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """Set the seed for random, numpy, and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures that CUDA selects deterministic algorithms when available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to: {seed}")


def get_device(args) -> str:
    """
    Returns a string representing the device based on available GPUs and args.gpu_ids.
    The --gpu_ids argument should be a comma-separated string of GPU indices (e.g., "0,1,2").
    """
    if torch.cuda.is_available():
        # Parse the gpu_ids argument into a list of integers.
        gpu_ids = [int(id_) for id_ in args.gpu_ids.split(',')]
        # Set CUDA_VISIBLE_DEVICES so that only these GPUs are visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        # Return the first GPU as the default device string.
        device_str = "cuda:0"
        print(f"[INFO] Using GPUs: {gpu_ids}. Default device set to {device_str}.")
    else:
        device_str = "cpu"
        print("[INFO] CUDA not available. Using CPU.")
    return device_str


if __name__ == "__main__":
    args = parse_args()
    t_model = get_model(args.dataset_name)
    print(f"start backdoor attack, current dataset: {args.dataset_name}, n_sellers: {args.n_sellers} ")
    set_seed(args.seed)
    device = get_device(args)
    save_path = f"/results/backdoor/{args.dataset_name}/n_seller_{args.n_sellers}_n_adv_{args.n_adversaries}_strength_{args.poison_strength}/"
    eval_results = backdoor_attack(
        dataset_name=args.dataset_name,
        n_sellers=args.n_sellers,
        n_adversaries=args.n_adversaries,
        model_structure=t_model,
        global_rounds=args.global_rounds,
        backdoor_target_label=args.backdoor_target_label,
        trigger_type=args.trigger_type,
        save_path=save_path,
        device=device,
        poison_strength=args.poison_strength,
        poison_test_sample=args.poison_test_sample
    )

    print("Evaluation Results:", eval_results)
