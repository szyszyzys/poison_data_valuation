#!/usr/bin/env python3
"""
fl_training.py

This file defines:
  - A simple neural network model (SimpleMLP) for demonstration.
  - A local training routine that trains a local copy of the global model on local data.
  - Functions to compute the gradient update as the difference between the trained and initial model parameters.
  - A helper to flatten the gradient into a single vector (useful for federated aggregation).

Usage (within a seller’s get_gradient method):
    flat_update, data_size = local_training_and_get_gradient(model, train_dataset, batch_size=16, device, local_epochs=1, lr=0.01)
    # flat_update is a numpy array representing the update from local training.
"""

import copy
import logging
import os
import time
from typing import List, Tuple, Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset, Dataset

# from model.text_model import TEXTCNN
from model.vision_model import LeNet, TextCNN, SimpleCNN


def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> Tuple[nn.Module, Union[float, None]]:
    model.train()
    batch_losses_all = []

    if not train_loader or len(train_loader) == 0:
        logging.warning("train_loader is empty or None. Skipping training.")
        return model, None

    logging.debug(f"Starting local training for {epochs} epochs on device {device}...")
    for epoch in range(epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                # Simplified data unpacking assumes a standard (data, labels, ...) format
                data, labels = batch_data[0], batch_data[1]

                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)

                if not torch.isfinite(loss):
                    logging.warning(
                        f"Non-finite loss ({loss.item()}) encountered in batch {batch_idx}. Skipping update."
                    )
                    continue

                loss.backward()
                optimizer.step()
                batch_losses_all.append(loss.item())

            except Exception as e:
                logging.warning(
                    f"Error during training step for batch {batch_idx} in epoch {epoch + 1}: {e}",
                    exc_info=False
                )
                continue

    if not batch_losses_all:
        logging.warning("No batches were successfully processed during training.")
        overall_avg_loss = None
    else:
        overall_avg_loss = np.mean(batch_losses_all)
        logging.info(
            f"Finished local training. Total successful batches: {len(batch_losses_all)}. "
            f"Overall Avg Loss: {overall_avg_loss:.4f}"
        )

    return model, overall_avg_loss

def compute_gradient_update(initial_model: nn.Module,
                            trained_model: nn.Module) -> List[torch.Tensor]:
    """
    Compute the gradient update as the difference between the trained model's parameters
    and the initial model's parameters. Returns a list of tensors.
    """
    grad_update = []
    for init_param, trained_param in zip(initial_model.parameters(), trained_model.parameters()):
        # The update is defined as (trained - initial)
        grad_update.append(trained_param.detach().cpu() - init_param.detach().cpu())
    return grad_update


def flatten_gradients(grad_list: List[torch.Tensor]) -> np.ndarray:
    """
    Flatten a list of gradient tensors into a single 1D numpy array.
    """
    flat_grad = torch.cat([g.view(-1) for g in grad_list])
    return flat_grad.numpy()


def test_local_model(model: nn.Module,
                     test_loader: DataLoader,
                     criterion: nn.Module,
                     device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if not test_loader or len(test_loader) == 0:
        logging.warning("test_loader is empty or None. Returning NaN metrics.")
        return {"loss": float('nan'), "accuracy": float('nan')}

    with torch.no_grad():
        for batch_idx, batch_items in enumerate(test_loader):
            try:
                # Simplified data unpacking
                batch_data, batch_labels = batch_items[0], batch_items[1]

                batch_data = batch_data.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                if torch.isfinite(loss):
                    total_loss += loss.item() * batch_data.size(0)
                    _, predicted = torch.max(outputs, dim=1)
                    total_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_data.size(0)
                else:
                    logging.warning(f"Non-finite loss in test batch {batch_idx}. Skipping.")

            except Exception as e:
                logging.warning(f"Error processing test batch {batch_idx}: {e}. Skipping.", exc_info=False)
                continue

    if total_samples == 0:
        logging.warning("No samples were successfully processed during testing.")
        return {"loss": float('nan'), "accuracy": float('nan')}

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


def local_training_and_get_gradient(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 0.01,
        opt_str: str = "SGD",
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        evaluate_on_full_train_set: bool = False
) -> Tuple[Optional[List[torch.Tensor]], Optional[np.ndarray], Optional[nn.Module], Dict, Optional[float]]:
    """
    Performs local training on a copy of the input model and returns the WEIGHT DELTA (not gradient),
    the trained local model, evaluation results, and average training loss.
    The input model (passed as `model`) is not modified.
    """
    try:
        local_model = copy.deepcopy(model)
        local_model.to(device)
        # Store the initial state dictionary before training
        initial_state_dict = copy.deepcopy(model.state_dict())
    except Exception as e:
        logging.error(f"Failed to deepcopy model for local training: {e}", exc_info=True)
        return None, None, None, {"loss": float('nan'), "accuracy": float('nan')}, None

    criterion = nn.CrossEntropyLoss()
    if opt_str.upper() == "ADAM":
        optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Default to SGD
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Train the local model
    trained_model, avg_train_loss = train_local_model(
        local_model, train_loader, criterion, optimizer, device, epochs=local_epochs
    )

    # Get the state dictionary of the trained model
    trained_state_dict = trained_model.state_dict()

    # Compute the weight delta using state dictionaries
    weight_delta_tensors: List[torch.Tensor] = []
    with torch.no_grad():
        for key in initial_state_dict:
            delta = trained_state_dict[key].cpu() - initial_state_dict[key].cpu()
            weight_delta_tensors.append(delta)

    flat_delta_np = flatten_gradients(weight_delta_tensors)

    eval_res = {"loss": float('nan'), "accuracy": float('nan')}
    if evaluate_on_full_train_set:
        eval_res = test_local_model(trained_model, train_loader, criterion, device)
    elif avg_train_loss is not None:
        eval_res["loss"] = avg_train_loss

    return weight_delta_tensors, flat_delta_np, trained_model, eval_res, avg_train_loss

def apply_gradient_update(
        initial_model: nn.Module,  # The model state to start from
        grad_update: List[torch.Tensor]  # The delta: (trained_params - initial_params)
) -> nn.Module:
    """
    Create a new model by adding the computed gradient update to a copy of the
    initial model's parameters.
    The gradient update is assumed to be computed as (trained_model_param - initial_model_param).

    Args:
        initial_model (nn.Module): The model before local training.
        grad_update (List[torch.Tensor]): List of gradient update tensors (deltas).

    Returns:
        nn.Module: A new model instance with updated parameters.
    """
    # Create a deep copy of the initial model to avoid modifying it in-place.
    # For maximal efficiency IF signature could change, one would reconstruct the model.
    try:
        updated_model = copy.deepcopy(initial_model)
    except Exception as e:
        logging.error(f"Failed to deepcopy initial_model in apply_gradient_update: {e}", exc_info=True)
        raise  # Re-raise, as we can't proceed

    # Ensure the model is on a device (if it's CPU, this does nothing; if CUDA, ensures params are there)
    # This step might be redundant if initial_model is already on the target device.
    # However, grad_update tensors are typically on CPU.
    # We'll move deltas to the parameter's device.

    num_model_params = len(list(updated_model.parameters()))
    if len(grad_update) != num_model_params:
        logging.error(
            f"Parameter mismatch in apply_gradient_update: "
            f"Model has {num_model_params} param groups, grad_update has {len(grad_update)}."
        )
        # Fallback: return the un-updated copy or raise error
        # For safety, let's return the un-updated copy with a warning.
        # A better approach might be to raise an error if strictness is required.
        return updated_model  # or raise ValueError(...)

    with torch.no_grad():  # Ensure operations are not tracked during parameter update
        for param, delta_tensor_cpu in zip(updated_model.parameters(), grad_update):
            if param.shape != delta_tensor_cpu.shape:
                logging.error(
                    f"Shape mismatch for parameter update: param shape {param.shape}, "
                    f"delta shape {delta_tensor_cpu.shape}. Skipping this parameter."
                )
                continue
            # Move delta to the same device as the parameter and add
            param.add_(delta_tensor_cpu.to(param.device))

    return updated_model


def get_model_params(model):
    """
    Extracts all parameters from a PyTorch model into a dictionary.

    Parameters:
        model: A PyTorch model

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to their values
    """
    return {name: param.clone().detach() for name, param in model.state_dict().items()}


# ---------------------------
# Model Saving/Loading Utilities
# ---------------------------

def save_model(model: nn.Module, path: str):
    """
    Save the model's state_dict to the specified file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device):
    """
    Load a model's state_dict from the specified file path.
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {path}")
    return model


def load_param(path: str, device: torch.device):
    """
    Load a model's state_dict from the specified file path.
    """
    state_dict = torch.load(path, map_location=device)
    print(f"Model loaded from {path}")
    return state_dict


MODEL_REGISTRY = {
    "simple_cnn": {
        "class": SimpleCNN,
        "supported_datasets": ["cifar", "celeba", "camelyon16"],
    },
    "lenet": {
        "class": LeNet,
        "supported_datasets": ["mnist", "fmnist", "cifar"],  # LeNet can now support CIFAR
    },
}


# --- 0c. Flexible Model Dispatcher Function ---

def get_image_model(
        model_name: str,
        dataset: Dataset,
        device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Gets an initialized model instance based on its name and the dataset.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"Model '{model_name}' is not in the registry. Available models: {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info["class"]

    dataset_name = dataset.__class__.__name__.lower().replace('custom', '').replace('fashion', '')
    if isinstance(dataset, Subset):
        dataset_name = dataset.dataset.__class__.__name__.lower().replace('custom', '').replace('fashion', '')

    if dataset_name not in model_info["supported_datasets"]:
        raise ValueError(
            f"Model '{model_name}' does not support dataset '{dataset_name}'. Supported: {model_info['supported_datasets']}")

    try:
        num_classes = len(dataset.classes)
    except AttributeError:
        # Handle Subset case
        if isinstance(dataset, Subset):
            num_classes = len(dataset.dataset.classes)
        else:
            raise AttributeError(f"Dataset '{dataset_name}' must have a '.classes' attribute.")

    try:
        sample_image, _ = dataset[0]
        in_channels = sample_image.shape[0]
    except (IndexError, TypeError):
        in_channels = 3 if dataset_name in ["cifar", "celeba", "camelyon16"] else 1
        print(f"Warning: Could not determine input channels from dataset sample. Defaulting to {in_channels}.")

    print(
        f"Instantiating model '{model_name}' for '{dataset_name}' with {in_channels} channels and {num_classes} classes.")
    model = model_class(in_channels=in_channels, num_classes=num_classes)

    if device:
        model.to(torch.device(device) if isinstance(device, str) else device)
        print(f"Model moved to device: {device}")

    return model


# --- Improved Snippet (get_text_model) ---
def get_text_model(
        dataset_name: str,
        num_classes: int,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,  # Added device
        **model_kwargs: Any
) -> nn.Module:
    print(f"Getting model for dataset: {dataset_name}")
    model: nn.Module
    match dataset_name.lower():
        case "ag_news" | "trec":
            # ... (original TextCNN instantiation logic) ...
            if vocab_size is None: raise ValueError("`vocab_size` is required for TextCNN model.")
            if padding_idx is None: raise ValueError("`padding_idx` is required for TextCNN model.")
            embed_dim = model_kwargs.get("embed_dim", 100)
            num_filters = model_kwargs.get("num_filters", 100)
            filter_sizes = model_kwargs.get("filter_sizes", [3, 4, 5])
            dropout = model_kwargs.get("dropout", 0.5)
            if not isinstance(filter_sizes, list): raise TypeError(
                f"Expected 'filter_sizes' to be a list, got {type(filter_sizes)}")
            model = TextCNN(
                vocab_size=vocab_size, embed_dim=embed_dim, num_filters=num_filters,
                filter_sizes=filter_sizes, num_class=num_classes, dropout=dropout,
                padding_idx=padding_idx
            )
        case _:
            raise NotImplementedError(f"Cannot find a model for dataset {dataset_name}")

    if device:  # Apply device if specified
        model.to(torch.device(device) if isinstance(device, str) else device)
    return model


def get_model_name(dataset_name):
    match dataset_name.lower():
        case "cifar":
            model = 'LeNet'
        case "fmnist":
            model = 'LeNet'
        case "trec" | "ag_news":
            model = 'TEXTCNN'
        case _:
            raise NotImplementedError(f"Cannot find the model for dataset {dataset_name}")
    return model


def get_domain(dataset_name: str) -> str:
    """
    Determines the data domain ('image' or 'text') based on the dataset name.

    Args:
        dataset_name (str): The name of the dataset (case-insensitive).

    Returns:
        str: The domain type ('image' or 'text').

    Raises:
        NotImplementedError: If the dataset name is not recognized.
    """
    domain: str  # Variable to hold the result

    # Convert to lower once for case-insensitive matching
    dataset_name_lower = dataset_name.lower()

    match dataset_name_lower:
        case "cifar":
            domain = 'image'
        case "fmnist":
            domain = 'image'
        case "celeba":
            domain = 'image'
        case "camelyon16":
            domain = 'image'
        # Use | (OR pattern) to match multiple string literals
        case "trec" | "ag_news":
            domain = 'text'
        # Add other datasets as needed using |
        # case "mnist" | "svhn":
        #     domain = 'image'
        # case "imdb" | "sst2":
        #     domain = 'text'
        case _:
            # Log the error for better debugging if needed
            logging.error(f"Unrecognized dataset name: {dataset_name}")
            # Raise error as before
            raise NotImplementedError(
                f"Cannot determine domain for dataset '{dataset_name}'")  # Added quotes for clarity

    return domain

def save_samples(data_loader, filename, n_samples=16, nrow=4, title="Samples"):
    """
    Save a grid of images from the provided DataLoader for visualization_226.

    Args:
        data_loader: DataLoader object from which to fetch a batch of images.
        filename: Path to save the visualization_226 image.
        n_samples: Number of images to display (default: 16).
        nrow: Number of images per row in the grid (default: 4).
        title: Title for the plot.
    """
    # Get one batch from the loader
    images, labels = next(iter(data_loader))

    # Ensure we don't exceed available images
    images = images[:n_samples]

    # Create a grid of images
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)

    # Convert grid to numpy for plotting
    np_grid = grid.cpu().numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))  # from (C, H, W) to (H, W, C)

    # Plot and save the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
