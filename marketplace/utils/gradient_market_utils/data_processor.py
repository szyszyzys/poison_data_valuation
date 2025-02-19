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


def split_dataset(dataset, num_clients, iid=True, alpha=0.5):
    """
    Splits the dataset indices among num_clients.

    Parameters:
      - dataset: a PyTorch dataset (must have dataset.targets or dataset.labels).
      - num_clients: number of clients to split the data among.
      - iid: if True, perform an IID (random) split; if False, use a Dirichlet-based non-IID split.
      - alpha: Dirichlet concentration parameter (lower alpha means more heterogeneity).

    Returns:
      A dictionary mapping client id (int) to a numpy array of indices.
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    if iid:
        # Shuffle and split equally.
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
    else:
        # Non-IID split using Dirichlet distribution.
        # Assume dataset has attribute 'targets' or 'labels'
        try:
            labels = np.array(dataset.targets)
        except AttributeError:
            labels = np.array(dataset.labels)

        num_classes = np.unique(labels).shape[0]
        splits = {i: [] for i in range(num_clients)}

        # For each class, split its indices among clients using Dirichlet sampling.
        for c in range(num_classes):
            c_idx = indices[labels[indices] == c]
            np.random.shuffle(c_idx)
            # Draw a distribution over clients for this class.
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Compute the number of samples for each client for class c.
            proportions = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
            c_splits = np.split(c_idx, proportions)
            for i, sub_idx in enumerate(c_splits):
                splits[i].extend(sub_idx.tolist())
        # Convert lists to numpy arrays and sort indices.
        splits = [np.array(sorted(splits[i])) for i in range(num_clients)]

    return {i: splits[i] for i in range(num_clients)}


def create_client_dataloaders(dataset, splits, batch_size=32, shuffle=True):
    """
    Given a dataset and a dictionary of splits (client_id: indices),
    return a dictionary mapping client_id to a DataLoader for that client's data.
    """
    client_loaders = {}
    for client_id, indices in splits.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        client_loaders[client_id] = loader
    return client_loaders


def get_data_set(dataset_name, num_clients=10, iid=True):
    # Load dataset (train split)
    match dataset_name:
        case "FMINIST":
            dataset = load_fmnist_dataset(train=True, download=True)
        case "CIFAR":
            dataset = load_cifar10_dataset(train=True, download=True)
        case _:
            raise NotImplementedError(f"No current dataset {dataset_name}")
    splits = split_dataset(dataset, num_clients, iid=iid)
    client_loaders = create_client_dataloaders(dataset, splits, batch_size=64)
    print("IID Splits:")
    for cid, loader in client_loaders.items():
        print(f"  Client {cid}: {len(loader.dataset)} samples")
    return client_loaders, dataset


if __name__ == "__main__":
    # Example usage with FashionMNIST:
    num_clients = 5

    # Load dataset (train split)
    dataset = load_fmnist_dataset(train=True, download=True)

    # IID split:
    splits_iid = split_dataset(dataset, num_clients, iid=True)
    client_loaders_iid = create_client_dataloaders(dataset, splits_iid, batch_size=64)
    print("IID Splits:")
    for cid, loader in client_loaders_iid.items():
        print(f"  Client {cid}: {len(loader.dataset)} samples")

    # Non-IID split (Dirichlet with alpha=0.5)
    splits_noniid = split_dataset(dataset, num_clients, iid=False, alpha=0.5)
    client_loaders_noniid = create_client_dataloaders(dataset, splits_noniid, batch_size=64)
    print("\nNon-IID Splits:")
    for cid, loader in client_loaders_noniid.items():
        print(f"  Client {cid}: {len(loader.dataset)} samples")


