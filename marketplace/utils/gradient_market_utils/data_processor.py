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
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
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

def split_dataset_by_label(
        dataset: Any,
        buyer_count: int,
        num_sellers: int,
        distribution_type: str = "UNI",  # "UNI" or "POW"
        label_split_type: str = "NonIID",  # "IID" or "NonIID"
        seller_label_distribution: Optional[Dict[int, List[int]]] = None,
        buyer_label_distribution: Optional[List[int]] = None,
        seller_qualities: Optional[Dict[str, float]] = None,
        malicious_sellers: Optional[List[int]] = None,
        seed: Optional[int] = None
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Splits the dataset indices between one buyer (DA) and several sellers (DPs) based on martFL paper's setup.

    Parameters:
        dataset: A PyTorch dataset object with a `targets` attribute or indexable as (image, label).
        buyer_count: Number of samples to allocate to the buyer (DA's root dataset).
        num_sellers: Number of seller clients (DPs).
        distribution_type: How to distribute data quantity among sellers
            - "UNI": All sellers get equal amount of data
            - "POW": Power-law distribution (some sellers get more data than others)
        label_split_type: How to distribute class labels among sellers
            - "IID": Each seller has samples from all classes (uniform)
            - "NonIID": Each seller has samples from a subset of classes
        seller_label_distribution: Maps seller IDs to list of labels they should receive.
        buyer_label_distribution: List of labels the buyer should predominantly receive.
        seller_qualities: Dict defining seller types and their proportions
            e.g., {"high_quality": 0.3, "biased": 0.3, "malicious": 0.4}
        malicious_sellers: List of seller IDs that should be treated as malicious
        seed: Random seed for reproducibility

    Returns:
        buyer_indices: List of dataset indices assigned to the buyer.
        seller_splits: Dictionary mapping seller IDs to lists of dataset indices.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_indices = list(range(len(dataset)))

    # Get dataset targets
    try:
        targets = dataset.targets
        if hasattr(targets, 'numpy'):  # Handle torch tensors
            targets = targets.numpy().tolist()
    except AttributeError:
        targets = [dataset[i][1] for i in range(len(dataset))]
        if hasattr(targets[0], 'item'):  # Handle torch tensors
            targets = [t.item() for t in targets]

    unique_labels = sorted(list(set(targets)))
    num_classes = len(unique_labels)

    # --- Buyer Split (DA's root dataset) ---
    if buyer_label_distribution is not None:
        # For biased buyer dataset distribution (Type-I or Type-II Bias in paper)
        buyer_candidates = [i for i, t in enumerate(targets) if t in buyer_label_distribution]
    else:
        # For unbiased buyer dataset
        buyer_candidates = all_indices.copy()

    if len(buyer_candidates) < buyer_count:
        print(f"Warning: Not enough buyer candidates ({len(buyer_candidates)}) for requested count ({buyer_count})")
        buyer_count = min(buyer_count, len(buyer_candidates))

    buyer_indices = random.sample(buyer_candidates, buyer_count)
    remaining_indices = list(set(all_indices) - set(buyer_indices))

    # --- Create seller splits based on seller quality types ---
    if seller_qualities is None:
        # Default from paper: 30% high-quality, 30% biased, 40% malicious
        seller_qualities = {"high_quality": 0.5, "biased": 0.5}

    # Calculate number of sellers in each quality category
    seller_counts = {}
    remaining_sellers = num_sellers
    for quality, proportion in seller_qualities.items():
        if quality == list(seller_qualities.keys())[-1]:
            # Last category gets all remaining sellers to avoid rounding issues
            seller_counts[quality] = remaining_sellers
        else:
            count = int(proportion * num_sellers)
            seller_counts[quality] = count
            remaining_sellers -= count

    # Distribute seller IDs to quality types
    seller_qualities_map = {}
    seller_id = 0
    for quality, count in seller_counts.items():
        for _ in range(count):
            seller_qualities_map[seller_id] = quality
            seller_id += 1

    # If malicious_sellers is provided, override the quality map
    if malicious_sellers is not None:
        for s_id in malicious_sellers:
            if s_id < num_sellers:
                seller_qualities_map[s_id] = "malicious"

    # --- Seller Splits based on distribution_type ---
    # Calculate how many samples each seller should get
    if distribution_type == "UNI":
        # Uniform distribution - equal number of samples per seller
        samples_per_seller = {i: len(remaining_indices) // num_sellers for i in range(num_sellers)}
        # Distribute remaining samples
        for i in range(len(remaining_indices) % num_sellers):
            samples_per_seller[i] += 1
    elif distribution_type == "POW":
        # Power-law distribution as used in the paper
        alpha = 1.5  # Power-law exponent (adjust as needed)
        weights = np.array([1 / (i + 1) ** alpha for i in range(num_sellers)])
        weights = weights / np.sum(weights)

        # Calculate samples per seller based on power-law weights
        total_samples = len(remaining_indices)
        samples_raw = weights * total_samples
        samples_per_seller = {i: int(samples_raw[i]) for i in range(num_sellers)}

        # Distribute remaining samples
        remaining = total_samples - sum(samples_per_seller.values())
        for i in np.argsort(samples_raw - np.floor(samples_raw))[-int(remaining):]:
            samples_per_seller[i] += 1
    else:
        raise ValueError(f"Unknown distribution_type: {distribution_type}")

    # --- Prepare label distribution for each seller ---
    if label_split_type == "IID":
        # IID: each seller gets data from all classes
        indices_by_label = {label: [] for label in unique_labels}
        for idx in remaining_indices:
            indices_by_label[targets[idx]].append(idx)

        seller_splits = {i: [] for i in range(num_sellers)}
        for label, indices in indices_by_label.items():
            random.shuffle(indices)
            for i, seller_id in enumerate(range(num_sellers)):
                # Calculate proportion of this label's samples for this seller
                proportion = samples_per_seller[seller_id] / len(remaining_indices)
                # Assign approximately that proportion of this label's samples
                start = int(i * len(indices) / num_sellers)
                end = int((i + 1) * len(indices) / num_sellers)
                seller_splits[seller_id].extend(indices[start:end])

        # Adjust to match the exact sample counts
        for seller_id in range(num_sellers):
            if len(seller_splits[seller_id]) > samples_per_seller[seller_id]:
                # Remove excess samples
                excess = len(seller_splits[seller_id]) - samples_per_seller[seller_id]
                seller_splits[seller_id] = random.sample(seller_splits[seller_id],
                                                         len(seller_splits[seller_id]) - excess)
            elif len(seller_splits[seller_id]) < samples_per_seller[seller_id]:
                # Add more samples
                shortage = samples_per_seller[seller_id] - len(seller_splits[seller_id])
                available = list(set(remaining_indices) - set().union(*seller_splits.values()))
                if available:
                    seller_splits[seller_id].extend(random.sample(available, min(shortage, len(available))))

    else:  # NonIID
        # Create seller_label_distribution if not provided
        if seller_label_distribution is None:
            seller_label_distribution = {}

            # High-quality sellers get all labels (evenly distributed)
            for seller_id, quality in seller_qualities_map.items():
                if quality == "high_quality":
                    seller_label_distribution[seller_id] = unique_labels.copy()
                elif quality == "biased":
                    # Biased sellers get half of the labels randomly
                    half_labels = random.sample(unique_labels, num_classes // 2)
                    seller_label_distribution[seller_id] = half_labels
                else:  # malicious
                    # For the paper's setup, malicious sellers also get data
                    # Can be customized based on attack type
                    seller_label_distribution[seller_id] = random.sample(unique_labels, num_classes // 2 + 1)

        # Group indices by label
        indices_by_label = defaultdict(list)
        for idx in remaining_indices:
            indices_by_label[targets[idx]].append(idx)

        # Initialize seller splits
        seller_splits = {i: [] for i in range(num_sellers)}

        # First pass: assign indices according to label preferences
        for seller_id, preferred_labels in seller_label_distribution.items():
            target_count = samples_per_seller[seller_id]
            # Calculate how many samples per preferred label
            samples_per_label = target_count // len(preferred_labels) if preferred_labels else 0

            for label in preferred_labels:
                if label in indices_by_label and indices_by_label[label]:
                    # Take up to samples_per_label samples for this label
                    take_count = min(samples_per_label, len(indices_by_label[label]))
                    chosen_indices = random.sample(indices_by_label[label], take_count)
                    seller_splits[seller_id].extend(chosen_indices)

                    # Remove chosen indices from available pool
                    indices_by_label[label] = list(set(indices_by_label[label]) - set(chosen_indices))

        # Second pass: fill up to target count with remaining indices
        available_indices = [idx for label_indices in indices_by_label.values() for idx in label_indices]
        for seller_id, target_count in samples_per_seller.items():
            if len(seller_splits[seller_id]) < target_count and available_indices:
                shortfall = target_count - len(seller_splits[seller_id])
                additional_indices = random.sample(available_indices, min(shortfall, len(available_indices)))
                seller_splits[seller_id].extend(additional_indices)
                available_indices = list(set(available_indices) - set(additional_indices))

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


# def get_data_set(dataset_name, buyer_count, num_sellers, iid=True,
#                  seller_label_distribution=None, buyer_label_distribution=None):
#     """
#     Load the dataset and split it between one buyer and several sellers.
#     The buyer can have biased (non-IID) data if buyer_label_distribution is provided.
#
#     Parameters:
#       dataset_name (str): Name of the dataset ("FMNIST", "CIFAR", etc.)
#       buyer_count (int): Number of samples to allocate to the buyer.
#       num_sellers (int): Number of seller clients.
#       iid (bool): If True, assume IID splits; if False, use non-IID splitting.
#       seller_label_distribution (dict, optional): Mapping for seller non-IID splits.
#       buyer_label_distribution (list, optional): List of labels for buyer's biased data.
#
#     Returns:
#       tuple: (buyer_loader, client_loaders, full_dataset, test_set_loader)
#     """
#     if dataset_name == "FMNIST":
#         transform = transforms.ToTensor()  # Images: [0,1] shape (1, H, W)
#         dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#     elif dataset_name == "CIFAR":
#         transform = transforms.ToTensor()
#         dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     else:
#         raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
#
#     # Split the dataset into buyer and sellers.
#     buyer_indices, seller_splits = split_dataset_buyer_seller(
#         dataset, buyer_count, num_sellers, iid,
#         seller_label_distribution=seller_label_distribution,
#         buyer_label_distribution=buyer_label_distribution
#     )
#
#     # Create DataLoader for the buyer.
#     buyer_loader = DataLoader(Subset(dataset, buyer_indices), batch_size=64, shuffle=True)
#     # Create DataLoaders for sellers.
#     client_loaders = create_client_dataloaders(dataset, seller_splits, batch_size=64, shuffle=True)
#
#     print("Buyer split:")
#     print(f"  Buyer: {len(buyer_indices)} samples")
#     print("Seller splits:")
#     for cid, loader in client_loaders.items():
#         print(f"  Seller {cid}: {len(loader.dataset)} samples")
#
#     test_set_loader = DataLoader(test_set, batch_size=64, shuffle=False)
#     return buyer_loader, client_loaders, dataset, test_set_loader

# def get_data_set(
#         dataset_name,
#         buyer_count=None,
#         buyer_percentage=0.01,
#         num_sellers=10,
#         distribution_type="UNI",  # "UNI" or "POW"
#         label_split_type="IID",  # "IID" or "NonIID"
#         seller_label_distribution=None,
#         buyer_label_distribution=None,
#         seller_qualities=None,
#         malicious_sellers=None,
#         batch_size=64,
#         normalize_data=True,
#         visualize=False,
#         n_adversaries=0
# ):
#     """
#     Load the dataset and split it between one buyer (DA) and several sellers (DPs)
#     according to the martFL paper's setup.
#
#     Parameters:
#       dataset_name (str): Name of the dataset ("FMNIST", "CIFAR", "MNIST", "AGNEWS", "TREC")
#       buyer_count (int, optional): Number of samples to allocate to the buyer. If None, uses buyer_percentage.
#       buyer_percentage (float): Percentage of total data for buyer (default: 0.01 = 1%)
#       num_sellers (int): Number of seller clients (DPs).
#       distribution_type (str): "UNI" for uniform distribution among sellers, "POW" for power-law distribution
#       label_split_type (str): "IID" for IID splits, "NonIID" for non-IID splits
#       seller_label_distribution (dict, optional): Mapping for seller non-IID splits.
#       buyer_label_distribution (list, optional): List of labels for buyer's biased data.
#       seller_qualities (dict, optional): Quality distribution among sellers (e.g., {"high_quality": 0.3, "biased": 0.3, "malicious": 0.4})
#       malicious_sellers (list, optional): List of seller IDs to treat as malicious
#       batch_size (int): Batch size for DataLoaders
#       seed (int): Random seed for reproducibility
#       normalize_data (bool): Whether to normalize data
#       visualize (bool): Whether to visualize the data distribution
#
#     Returns:
#       tuple: (buyer_loader, seller_loaders, full_dataset, test_loader, class_names)
#     """
#     # Define transforms based on the dataset
#     if normalize_data:
#         if dataset_name == "FMNIST":
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.2860,), (0.3530,))
#             ])
#         elif dataset_name == "CIFAR":
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#             ])
#         elif dataset_name == "MNIST":
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         else:
#             transform = transforms.ToTensor()  # Default for other datasets
#     else:
#         transform = transforms.ToTensor()
#
#     # Load dataset
#     if dataset_name == "FMNIST":
#         dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#         class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#     elif dataset_name == "CIFAR":
#         dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#         class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                        'dog', 'frog', 'horse', 'ship', 'truck']
#     elif dataset_name == "MNIST":
#         dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#         test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#         class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     elif dataset_name == "AGNEWS":
#         try:
#             from torchtext.datasets import AG_NEWS
#             from torchtext.data.utils import get_tokenizer
#             from torchtext.vocab import build_vocab_from_iterator
#             # This is a placeholder for AGNEWS, which would need additional text processing
#             print("AGNEWS dataset requires additional setup and is not fully implemented in this function")
#             # You would need to implement text tokenization, vocabulary building, etc.
#             class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
#         except ImportError:
#             raise ImportError("torchtext is required for AGNEWS dataset")
#     elif dataset_name == "TREC":
#         # This would require a custom dataset loader since it's not in torchvision
#         raise NotImplementedError("TREC dataset is not implemented in this function. Please implement a custom loader.")
#     else:
#         raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
#
#     # Calculate buyer count if not provided
#     if buyer_count is None:
#         buyer_count = int(len(dataset) * buyer_percentage)
#         print(f"Setting buyer count to {buyer_count} samples ({buyer_percentage * 100:.2f}% of data)")
#
#     # Set default buyer label distribution if not provided and using NonIID
#     if buyer_label_distribution is None and label_split_type == "NonIID":
#         num_classes = len(class_names)
#         # In martFL paper, buyer often has biased dataset with half the labels
#         buyer_label_distribution = list(range(num_classes // 2))
#         print(f"Setting buyer to have biased data with labels: {buyer_label_distribution}")
#
#     # Set default seller qualities if not provided
#     if seller_qualities is None:
#         # Default from martFL paper: 30% high-quality, 30% biased, 40% malicious
#         seller_qualities = {"high_quality": 0.3, "biased": 0.3, "malicious": 0.4}
#         print(f"Using default seller qualities: {seller_qualities}")
#
#     # Split the dataset into buyer and sellers
#     buyer_indices, seller_splits = split_dataset_buyer_seller(
#         dataset=dataset,
#         buyer_count=buyer_count,
#         num_sellers=num_sellers,
#         distribution_type=distribution_type,
#         label_split_type=label_split_type,
#         seller_label_distribution=seller_label_distribution,
#         buyer_label_distribution=buyer_label_distribution,
#         seller_qualities=seller_qualities,
#         malicious_sellers=malicious_sellers,
#     )
#
#     # Create DataLoader for the buyer
#     buyer_loader = DataLoader(
#         Subset(dataset, buyer_indices),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=torch.cuda.is_available()
#     )
#
#     # Create DataLoaders for sellers
#     seller_loaders = {}
#     for seller_id, indices in seller_splits.items():
#         seller_loaders[seller_id] = DataLoader(
#             Subset(dataset, indices),
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=torch.cuda.is_available()
#         )
#
#     # Create test loader
#     test_loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=torch.cuda.is_available()
#     )
#
#     # Print summary statistics
#     print("\nData Distribution Summary:")
#     print(f"Buyer (DA): {len(buyer_indices)} samples")
#
#     # Calculate distribution of classes in buyer data
#     buyer_labels = [dataset.targets[i] if hasattr(dataset, 'targets') else dataset[i][1] for i in buyer_indices]
#     if hasattr(buyer_labels[0], 'item'):
#         buyer_labels = [label.item() for label in buyer_labels]
#     buyer_class_counts = {class_names[i]: buyer_labels.count(i) for i in range(len(class_names))}
#     print(f"Buyer class distribution: {buyer_class_counts}")
#
#     # Calculate quality types for each seller
#     seller_quality_types = {}
#     high_quality_count = int(num_sellers * seller_qualities.get("high_quality", 0))
#     biased_count = int(num_sellers * seller_qualities.get("biased", 0))
#
#     for seller_id in range(num_sellers):
#         if seller_id < high_quality_count:
#             quality = "high_quality"
#         elif seller_id < high_quality_count + biased_count:
#             quality = "biased"
#         else:
#             quality = "malicious"
#
#         # Override if in malicious_sellers list
#         if malicious_sellers is not None and seller_id in malicious_sellers:
#             quality = "malicious"
#
#         seller_quality_types[seller_id] = quality
#
#     # Print seller information
#     print("\nSeller distribution:")
#     for seller_id, loader in seller_loaders.items():
#         quality = seller_quality_types[seller_id]
#         print(f"Seller {seller_id} ({quality}): {len(loader.dataset)} samples")
#
#         # For the first few sellers, print class distribution
#         if seller_id < 3:  # Limit to avoid too much output
#             seller_indices = seller_splits[seller_id]
#             seller_labels = [dataset.targets[i] if hasattr(dataset, 'targets') else dataset[i][1] for i in
#                              seller_indices]
#             if hasattr(seller_labels[0], 'item'):
#                 seller_labels = [label.item() for label in seller_labels]
#
#             seller_class_counts = {class_names[i]: seller_labels.count(i) for i in range(len(class_names))}
#             print(f"  Class distribution: {seller_class_counts}")
#
#     # Visualize the data distribution if requested
#     if visualize:
#         visualize_data_distribution(dataset, buyer_indices, seller_splits,
#                                     seller_quality_types, class_names)
#
#     return buyer_loader, seller_loaders, dataset, test_loader, class_names


def get_data_set(
        dataset_name,
        buyer_percentage=0.01,
        num_sellers=10,
        batch_size=64,
        normalize_data=True,
        split_method="Dirichlet",
        n_adversaries=0,
        save_path='./result'
):
    # Define transforms based on the dataset.
    if normalize_data:
        if dataset_name == "FMNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        elif dataset_name == "CIFAR":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform = transforms.ToTensor()
    else:
        transform = transforms.ToTensor()

    # Load training and test datasets.
    if dataset_name == "FMNIST":
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == "CIFAR":
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")

    # Determine the number of buyer samples.
    total_samples = len(dataset)
    buyer_count = int(total_samples * buyer_percentage)
    print(f"Allocating {buyer_count} samples ({buyer_percentage * 100:.2f}%) for the buyer.")

    if split_method == "label":
        buyer_indices, seller_splits = split_dataset_by_label(dataset=dataset,
                                                              buyer_count=buyer_count,
                                                              num_sellers=num_sellers, )
    else:
        buyer_indices, seller_splits = split_dataset_buyer_seller_improved(
            dataset=dataset,
            buyer_count=buyer_count,
            num_sellers=num_sellers,
            split_method=split_method,
            dirichlet_alpha=0.7,
            n_adversaries=n_adversaries
        )
    data_distribution_info = print_and_save_data_statistics(dataset, buyer_indices, seller_splits, save_results=True,
                                                            output_dir=save_path)
    # Create DataLoaders.
    buyer_loader = DataLoader(Subset(dataset, buyer_indices), batch_size=batch_size, shuffle=True)
    seller_loaders = {i: DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
                      for i, indices in seller_splits.items()}
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return buyer_loader, seller_loaders, dataset, test_loader, class_names


def dirichlet_partition(indices_by_class: dict, n_clients: int, alpha: float) -> dict:
    """
    Partition indices among n_clients using a Dirichlet distribution.

    Parameters:
        indices_by_class (dict): Mapping from class label to list of indices.
        n_clients (int): Number of clients to partition data among.
        alpha (float): Dirichlet concentration parameter.

    Returns:
        client_indices (dict): Mapping from client id (0 to n_clients-1) to list of assigned indices.
    """
    client_indices = {i: [] for i in range(n_clients)}
    for c, indices in indices_by_class.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        # Sample proportions from Dirichlet distribution.
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        # Determine the split points
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    return client_indices


def balanced_partition(indices_by_class: dict, n_clients: int) -> dict:
    """
    Partition indices evenly among n_clients in a balanced (IID) manner.

    Parameters:
        indices_by_class (dict): Mapping from class label to list of indices.
        n_clients (int): Number of clients.

    Returns:
        client_indices (dict): Mapping from client id (0 to n_clients-1) to list of indices.
    """
    client_indices = {i: [] for i in range(n_clients)}
    for c, indices in indices_by_class.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        split_size = len(indices) // n_clients
        # For simplicity, each client gets an equal number; remainder is discarded.
        for i in range(n_clients):
            client_indices[i].extend(indices[i * split_size: (i + 1) * split_size].tolist())
    return client_indices


def split_dataset_buyer_seller_improved(dataset,
                                        buyer_count: int,
                                        num_sellers: int,
                                        split_method: str = "Dirichlet",
                                        dirichlet_alpha: float = 0.5,
                                        n_adversaries: int = 0) -> (np.ndarray, dict):
    """
    Split the dataset indices into a buyer set and seller splits.

    Two modes are supported:
      - "Dirichlet": Partition the remaining data among all sellers using a Dirichlet distribution.
      - "AdversaryFirst": First assign balanced (IID) data to the first n_adversaries sellers,
                           then partition the remaining data among the remaining sellers using Dirichlet.

    Parameters:
      dataset: A dataset object with attribute 'targets' (list/array of labels).
      buyer_count (int): Number of samples reserved for the buyer.
      num_sellers (int): Total number of seller clients.
      split_method (str): "Dirichlet" or "AdversaryFirst".
      dirichlet_alpha (float): Dirichlet parameter for non-IID partitioning.
      n_adversaries (int): Number of sellers to assign balanced data (for "AdversaryFirst").

    Returns:
      buyer_indices (np.ndarray): Array of indices for the buyer.
      seller_splits (dict): Mapping from seller id to list of indices.
    """
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    buyer_indices = all_indices[:buyer_count]
    seller_indices = all_indices[buyer_count:]

    # Obtain the targets for the seller data.
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        # Assume dataset[i] returns (data, label)
        targets = np.array([dataset[i][1] for i in range(total_samples)])
    seller_targets = targets[seller_indices]

    # Build dictionary: class -> list of indices among seller_indices
    indices_by_class = {}
    unique_classes = np.unique(seller_targets)
    for c in unique_classes:
        indices_by_class[c] = seller_indices[seller_targets == c].tolist()

    seller_splits = {}
    if split_method.lower() == "dirichlet":
        # Partition all seller data using Dirichlet.
        seller_splits = dirichlet_partition(indices_by_class, num_sellers, dirichlet_alpha)
    elif split_method.lower() == "adversaryfirst":
        # For the first n_adversaries, assign balanced (IID) data.
        if n_adversaries <= 0 or n_adversaries > num_sellers:
            raise ValueError("n_adversaries must be between 1 and num_sellers")
        adversary_splits = balanced_partition(indices_by_class, n_adversaries)
        # Remove the indices assigned to adversaries.
        adversary_assigned = []
        for i in range(n_adversaries):
            adversary_assigned.extend(adversary_splits[i])
        remaining_indices = np.setdiff1d(seller_indices, np.array(adversary_assigned))
        # Recompute indices_by_class for the remaining indices.
        remaining_targets = targets[remaining_indices]
        indices_by_class_remaining = {}
        for c in unique_classes:
            indices_by_class_remaining[c] = remaining_indices[remaining_targets == c].tolist()
        benign_splits = dirichlet_partition(indices_by_class_remaining, num_sellers - n_adversaries, dirichlet_alpha)
        seller_splits = {}
        # First n_adversaries get balanced splits.
        for i in range(n_adversaries):
            seller_splits[i] = adversary_splits[i]
        # The remaining sellers get non-IID splits.
        for j in range(n_adversaries, num_sellers):
            seller_splits[j] = benign_splits[j - n_adversaries]
    else:
        raise ValueError("Unknown split_method. Use 'Dirichlet' or 'AdversaryFirst'.")

    # Ensure each seller has at least one sample.
    for seller_id, indices in seller_splits.items():
        if len(indices) == 0:
            seller_splits[seller_id] = [int(np.random.choice(seller_indices))]

    return buyer_indices, seller_splits


import numpy as np
import matplotlib.pyplot as plt


def print_and_save_data_statistics(dataset, buyer_indices, seller_splits, save_results=True, output_dir='./results'):
    """
    Print and visualize the class distribution statistics for the buyer and each seller.
    Optionally, save the statistics to a JSON file and the figures as PNG images.

    Parameters:
      dataset: Dataset object with attribute 'targets' or that returns (data, label).
      buyer_indices: Array-like of indices for the buyer.
      seller_splits: Dictionary mapping seller id to list of indices.
      save_results (bool): Whether to save the results.
      output_dir (str): Directory where the results will be saved.
    """
    # Get the targets from the dataset.
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        # Assume dataset[i] returns (data, label)
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    # Compute unique classes present in the dataset.
    unique_classes = np.unique(targets)

    # Compute buyer statistics.
    buyer_targets = targets[buyer_indices]
    buyer_counts = {str(c): int(np.sum(buyer_targets == c)) for c in unique_classes}
    buyer_stats = {
        "total_samples": int(len(buyer_indices)),
        "class_distribution": buyer_counts
    }

    print("Buyer Data Statistics:")
    print(f"  Total Samples: {buyer_stats['total_samples']}")
    for c in unique_classes:
        print(f"  Class {c}: {buyer_counts[str(c)]}")
    print("\n" + "=" * 40 + "\n")

    # Compute seller statistics.
    seller_stats = {}
    for seller_id, indices in seller_splits.items():
        seller_targets = targets[indices]
        counts = {str(c): int(np.sum(seller_targets == c)) for c in unique_classes}
        seller_stats[seller_id] = {
            "total_samples": int(len(indices)),
            "class_distribution": counts
        }
        print(f"Seller {seller_id} Data Statistics:")
        print(f"  Total Samples: {seller_stats[seller_id]['total_samples']}")
        for c in unique_classes:
            print(f"  Class {c}: {counts[str(c)]}")
        print("-" * 30)

    # Save statistics to a dictionary.
    results = {
        "buyer_stats": buyer_stats,
        "seller_stats": seller_stats
    }

    # Save the results if requested.
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        stats_file = os.path.join(output_dir, 'data_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Statistics saved to {stats_file}")

    # Visualize Buyer Distribution.
    plt.figure(figsize=(8, 4))
    plt.bar([str(c) for c in unique_classes], [buyer_counts[str(c)] for c in unique_classes])
    plt.title("Buyer Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    if save_results:
        buyer_fig_file = os.path.join(output_dir, 'buyer_distribution.png')
        plt.savefig(buyer_fig_file)
        print(f"Buyer distribution figure saved to {buyer_fig_file}")
    plt.close()

    # Visualize Seller Distributions.
    num_sellers = len(seller_splits)
    n_cols = 3
    n_rows = int(np.ceil(num_sellers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if num_sellers == 1:
        axes = [axes]  # make it iterable
    else:
        axes = axes.flatten()

    for i, (seller_id, indices) in enumerate(seller_splits.items()):
        seller_targets = targets[indices]
        counts = {str(c): int(np.sum(seller_targets == c)) for c in unique_classes}
        axes[i].bar([str(c) for c in unique_classes], [counts[str(c)] for c in unique_classes])
        axes[i].set_title(f"Seller {seller_id}")
        axes[i].set_xlabel("Class")
        axes[i].set_ylabel("Count")

    # Hide any unused subplots.
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_results:
        sellers_fig_file = os.path.join(output_dir, 'seller_distribution.png')
        plt.savefig(sellers_fig_file)
        print(f"Seller distribution figure saved to {sellers_fig_file}")
    plt.close()

    return results


def visualize_data_distribution(dataset, buyer_indices, seller_splits,
                                seller_quality_types, class_names):
    """Visualize the distribution of data among buyer and sellers."""
    import matplotlib.pyplot as plt

    # Get targets
    try:
        targets = dataset.targets
        if hasattr(targets, 'numpy'):
            targets = targets.numpy().tolist()
    except AttributeError:
        targets = [dataset[i][1] for i in range(len(dataset))]
        if hasattr(targets[0], 'item'):
            targets = [t.item() for t in targets]

    num_classes = len(class_names)

    # Count class distribution for buyer
    buyer_dist = [0] * num_classes
    for idx in buyer_indices:
        target = targets[idx]
        if hasattr(target, 'item'):
            target = target.item()
        buyer_dist[target] += 1

    # Count class distribution for each seller
    seller_distributions = {}

    for seller_id, indices in seller_splits.items():
        seller_distributions[seller_id] = [0] * num_classes
        for idx in indices:
            target = targets[idx]
            if hasattr(target, 'item'):
                target = target.item()
            seller_distributions[seller_id][target] += 1

    # Plot
    plt.figure(figsize=(15, 10))

    # Plot buyer distribution
    plt.subplot(2, 1, 1)
    plt.bar(range(num_classes), buyer_dist, tick_label=class_names)
    plt.title("Buyer (DA) Data Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Plot seller distributions
    plt.subplot(2, 1, 2)
    colors = {"high_quality": 'green', "biased": 'orange', "malicious": 'red'}

    # Group sellers by quality type
    high_quality_sellers = []
    biased_sellers = []
    malicious_sellers = []

    for seller_id, quality in seller_quality_types.items():
        if quality == "high_quality":
            high_quality_sellers.append(seller_id)
        elif quality == "biased":
            biased_sellers.append(seller_id)
        else:
            malicious_sellers.append(seller_id)

    # Plot average distribution for each seller type
    for quality, seller_ids in [
        ("high_quality", high_quality_sellers),
        ("biased", biased_sellers),
        ("malicious", malicious_sellers)
    ]:
        if not seller_ids:
            continue

        # Calculate average distribution
        avg_dist = [0] * num_classes
        for seller_id in seller_ids:
            for i in range(num_classes):
                avg_dist[i] += seller_distributions[seller_id][i]

        # Normalize
        if seller_ids:
            avg_dist = [count / len(seller_ids) for count in avg_dist]

        plt.bar(
            range(num_classes),
            avg_dist,
            alpha=0.5,
            label=f"{quality.capitalize()} Sellers (avg)",
            color=colors[quality]
        )

    plt.title("Average Distribution by Seller Type")
    plt.xlabel("Class")
    plt.ylabel("Average Count")
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot individual seller distributions by type
    plt.figure(figsize=(15, 15))

    # Plot a few individual sellers of each type
    for i, (quality, seller_ids) in enumerate([
        ("high_quality", high_quality_sellers[:3]),  # Show up to 3 of each type
        ("biased", biased_sellers[:3]),
        ("malicious", malicious_sellers[:3])
    ]):
        if not seller_ids:
            continue

        plt.subplot(3, 1, i + 1)

        for seller_id in seller_ids:
            plt.plot(
                range(num_classes),
                seller_distributions[seller_id],
                marker='o',
                linestyle='-',
                alpha=0.7,
                label=f"Seller {seller_id}"
            )

        plt.title(f"{quality.capitalize()} Sellers - Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(range(num_classes), class_names, rotation=45)
        plt.legend()

    plt.tight_layout()
    plt.show()
