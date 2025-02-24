# !/usr/bin/env python3
"""
federated_data_split.py

This module provides functions to:
  - Load a dataset (example here uses FashionMNIST; you can easily modify for CIFAR-10).
  - Split the dataset indices among a specified number of clients.
    * IID: Random, equal split.
    * Non-IID: Using a Dirichlet distribution (controlled by parameter alpha).
  - Create a PyTorch DataLoader for each client's data.

Usage Example:
--------------
    from federated_data_split import load_fmnist_dataset, split_dataset, create_client_dataloaders

    # Load FashionMNIST training data
    dataset = load_fmnist_dataset(train=True, download=True)

    # Split the dataset among 10 clients in a non-IID manner (Dirichlet alpha=0.5)
    num_clients = 10
    splits = split_dataset(dataset, num_clients, iid=False, alpha=0.5)

    # Create DataLoaders for each client (e.g., batch_size=32)
    client_loaders = create_client_dataloaders(dataset, splits, batch_size=32, shuffle=True)

    # Now client_loaders is a dictionary: {client_id: DataLoader, ...}
    for cid, loader in client_loaders.items():
        print(f"Client {cid} has {len(loader.dataset)} samples.")
"""

import random
from collections import defaultdict

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_fmnist_dataset(train=True, download=True):
    """
    Load the FashionMNIST dataset with a basic transform.
    Returns the dataset object.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert images to tensor in [0,1]
        transforms.Normalize((0.5,), (0.5,))  # normalize to [-1,1]
    ])
    dataset = datasets.FashionMNIST(root="./data", train=train, transform=transform, download=download)
    return dataset


def load_cifar10_dataset(train=True, download=True):
    """
    Load the CIFAR-10 dataset with standard normalization.
    Returns the dataset object.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=download)
    return dataset


# def split_dataset_buyer_seller(dataset, buyer_count, num_sellers):
#     """
#     Split the dataset indices such that the buyer gets a fixed number of samples,
#     and the remaining samples are randomly split among the sellers.
#
#     Parameters:
#       dataset (Dataset): The full dataset.
#       buyer_count (int): Number of samples to assign to the buyer.
#       num_sellers (int): Number of seller clients.
#
#     Returns:
#       dict: A dictionary with keys 'buyer' and 'seller_i' for each seller,
#             mapping to lists of indices.
#     """
#     # Get all indices and shuffle them.
#     indices = np.arange(len(dataset))
#     np.random.shuffle(indices)
#
#     # Assign the first buyer_count indices to the buyer.
#     buyer_indices = indices[:buyer_count].tolist()
#     remaining_indices = indices[buyer_count:]
#
#     # Split the remaining indices among sellers (roughly equal splits)
#     seller_splits = np.array_split(remaining_indices, num_sellers)
#     splits = {"buyer": buyer_indices}
#     for i, seller_indices in enumerate(seller_splits):
#         splits[f"seller_{i}"] = seller_indices.tolist()
#     return splits


# def create_client_dataloaders(dataset, splits, batch_size=32, shuffle=True):
#     """
#     Given a dataset and a dictionary of splits (client_id: indices),
#     return a dictionary mapping client_id to a DataLoader for that client's data.
#     """
#     client_loaders = {}
#     for client_id, indices in splits.items():
#         subset = Subset(dataset, indices)
#         loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
#         client_loaders[client_id] = loader
#     return client_loaders


# Example get_data_set function that uses the buyer-seller split.
# def get_data_set(dataset_name, buyer_count, num_sellers, iid=True):
#     """
#     Load the dataset and split it between a buyer and several sellers.
#
#     Parameters:
#       dataset_name (str): Name of the dataset ("FMNIST", "CIFAR", etc.)
#       buyer_count (int): Number of samples to allocate to the buyer.
#       num_sellers (int): Number of seller clients.
#       iid (bool): If True, assume IID distribution (affects how splitting might be done).
#
#     Returns:
#       tuple: (client_loaders, full_dataset, test_set_loader)
#     """
#     # Load dataset (for example purposes, we'll assume these functions exist)
#     if dataset_name == "FMNIST":
#         from torchvision import datasets, transforms
#         transform = transforms.ToTensor()  # FMNIST images will be [0, 1] and shape (1, H, W)
#         dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#     elif dataset_name == "CIFAR":
#         from torchvision import datasets, transforms
#         transform = transforms.ToTensor()
#         dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     else:
#         raise NotImplementedError(f"No current dataset {dataset_name}")
#
#     # Create splits: assign buyer_count samples to buyer, rest to sellers.
#     splits = split_dataset_buyer_seller(dataset, buyer_count, num_sellers)
#     client_loaders = create_client_dataloaders(dataset, splits, batch_size=64, shuffle=True)
#
#     print("Client splits:")
#     for cid, loader in client_loaders.items():
#         print(f"  Client {cid}: {len(loader.dataset)} samples")
#
#     test_set_loader = DataLoader(test_set, batch_size=64, shuffle=False)
#     return client_loaders, dataset, test_set_loader


def split_dataset_buyer_seller(dataset, buyer_count, num_sellers, iid=True,
                               seller_label_distribution=None,
                               buyer_label_distribution=None):
    """
    Splits the dataset indices between one buyer and several sellers.

    Parameters:
      dataset: A PyTorch dataset object. It should have a `targets` attribute or be indexable as (image, label).
      buyer_count (int): Number of samples to allocate to the buyer.
      num_sellers (int): Number of seller clients.
      iid (bool): If True, splits are IID (random); if False, non-IID splits are produced.
      seller_label_distribution (dict, optional): Mapping from seller IDs (e.g. 0,1,2,...)
           to a list of labels that that seller should predominantly receive.
           For example, {0: [0, 1], 1: [2, 3]}.
      buyer_label_distribution (list, optional): A list of labels that the buyer should predominantly receive.
           For example, [0, 1] means the buyer’s data will be mostly from labels 0 and 1.

    Returns:
      buyer_indices (list): List of dataset indices assigned to the buyer.
      seller_splits (dict): Dictionary mapping seller IDs (0, 1, ..., num_sellers-1) to lists of dataset indices.
    """
    all_indices = list(range(len(dataset)))
    try:
        targets = dataset.targets
    except AttributeError:
        targets = [dataset[i][1] for i in range(len(dataset))]

    # --- Buyer Split ---
    if buyer_label_distribution is not None:
        # Only select indices whose label is in the buyer's designated labels.
        buyer_candidates = [i for i in all_indices if targets[i] in buyer_label_distribution]
    else:
        buyer_candidates = all_indices.copy()

    # If not enough candidate samples, sample from the whole dataset.
    if len(buyer_candidates) < buyer_count:
        buyer_indices = random.sample(all_indices, buyer_count)
    else:
        buyer_indices = random.sample(buyer_candidates, buyer_count)

    remaining_indices = list(set(all_indices) - set(buyer_indices))

    # --- Seller Splits ---
    if iid:
        # IID: simply shuffle and split remaining indices equally.
        random.shuffle(remaining_indices)
        seller_splits = {}
        per_seller = len(remaining_indices) // num_sellers
        for i in range(num_sellers):
            start = i * per_seller
            if i == num_sellers - 1:
                seller_splits[i] = remaining_indices[start:]
            else:
                seller_splits[i] = remaining_indices[start:start + per_seller]
    else:
        # Non-IID: use seller_label_distribution if provided; otherwise, assign two adjacent classes per seller.
        seller_splits = {i: [] for i in range(num_sellers)}
        if seller_label_distribution is None:
            unique_labels = sorted(list(set(targets)))
            num_classes = len(unique_labels)
            seller_label_distribution = {}
            for i in range(num_sellers):
                label1 = unique_labels[i % num_classes]
                label2 = unique_labels[(i + 1) % num_classes]
                seller_label_distribution[i] = [label1, label2]
        # Build inverse mapping from label to seller IDs.
        label_to_sellers = defaultdict(list)
        for seller, label_list in seller_label_distribution.items():
            for lab in label_list:
                label_to_sellers[lab].append(seller)
        # Assign each remaining index to a seller that is interested in its label.
        for idx in remaining_indices:
            lab = targets[idx]
            candidates = label_to_sellers.get(lab, [])
            if candidates:
                chosen = random.choice(candidates)
            else:
                chosen = random.choice(list(seller_splits.keys()))
            seller_splits[chosen].append(idx)

    return buyer_indices, seller_splits


def create_client_dataloaders(dataset, splits, batch_size=64, shuffle=True):
    """
    Create a dictionary of DataLoaders for each seller client.

    Parameters:
      dataset: The full dataset.
      splits:  A dictionary mapping seller IDs to lists of dataset indices.
      batch_size (int): Batch size for the DataLoader.
      shuffle (bool): Whether to shuffle the data.

    Returns:
      A dictionary mapping seller IDs to DataLoader objects.
    """
    loaders = {}
    for cid, indices in splits.items():
        subset = Subset(dataset, indices)
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loaders


def get_data_set(dataset_name, buyer_count, num_sellers, iid=True,
                 seller_label_distribution=None, buyer_label_distribution=None):
    """
    Load the dataset and split it between one buyer and several sellers.
    The buyer can have biased (non-IID) data if buyer_label_distribution is provided.

    Parameters:
      dataset_name (str): Name of the dataset ("FMNIST", "CIFAR", etc.)
      buyer_count (int): Number of samples to allocate to the buyer.
      num_sellers (int): Number of seller clients.
      iid (bool): If True, assume IID splits; if False, use non-IID splitting.
      seller_label_distribution (dict, optional): Mapping for seller non-IID splits.
      buyer_label_distribution (list, optional): List of labels for buyer's biased data.

    Returns:
      tuple: (buyer_loader, client_loaders, full_dataset, test_set_loader)
    """
    if dataset_name == "FMNIST":
        transform = transforms.ToTensor()  # Images: [0,1] shape (1, H, W)
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR":
        transform = transforms.ToTensor()
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")

    # Split the dataset into buyer and sellers.
    buyer_indices, seller_splits = split_dataset_buyer_seller(
        dataset, buyer_count, num_sellers, iid,
        seller_label_distribution=seller_label_distribution,
        buyer_label_distribution=buyer_label_distribution
    )

    # Create DataLoader for the buyer.
    buyer_loader = DataLoader(Subset(dataset, buyer_indices), batch_size=64, shuffle=True)
    # Create DataLoaders for sellers.
    client_loaders = create_client_dataloaders(dataset, seller_splits, batch_size=64, shuffle=True)

    print("Buyer split:")
    print(f"  Buyer: {len(buyer_indices)} samples")
    print("Seller splits:")
    for cid, loader in client_loaders.items():
        print(f"  Seller {cid}: {len(loader.dataset)} samples")

    test_set_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    return buyer_loader, client_loaders, dataset, test_set_loader
