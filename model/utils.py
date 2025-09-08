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
from typing import List, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from model.text_model import TEXTCNN
from model.models import LeNet, TextCNN, SimpleCNN


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
        weight_decay: float = 0.0005
) -> Tuple[Optional[List[torch.Tensor]], Optional[float]]:
    """
    Performs local training on a copy of the input model and returns the WEIGHT DELTA.
    The input model (passed as `model`) is not modified.
    """
    # Store the initial state of the original model
    initial_state_dict = model.state_dict()

    # --- FIX: Create a deepcopy of the model to be trained ---
    # This ensures the original model passed into this function is not modified.
    model_for_training = copy.deepcopy(model)
    model_for_training.to(device)

    criterion = nn.CrossEntropyLoss()
    if opt_str.upper() == "ADAM":
        optimizer = optim.Adam(model_for_training.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Default to SGD
        optimizer = optim.SGD(model_for_training.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Train the copied model
    trained_model, avg_train_loss = train_local_model(
        model_for_training, train_loader, criterion, optimizer, device, epochs=local_epochs
    )
    trained_state_dict = trained_model.state_dict()

    # Compute the weight delta using state dictionaries
    weight_delta_tensors: List[torch.Tensor] = []
    with torch.no_grad():
        for key in initial_state_dict:
            # Use the original model's state dict for the subtraction
            delta = trained_state_dict[key].cpu() - initial_state_dict[key].cpu()
            weight_delta_tensors.append(delta)

    # Return only the necessary values
    return weight_delta_tensors, avg_train_loss


# --- 1. Update the MODEL_REGISTRY to include text models ---
MODEL_REGISTRY = {
    # Image Models
    "simple_cnn": {
        "class": SimpleCNN,
        "domain": "image",
        "supported_datasets": ["cifar", "celeba", "camelyon16"],
    },
    "lenet": {
        "class": LeNet,
        "domain": "image",
        "supported_datasets": ["mnist", "fmnist", "cifar", "celeba", "camelyon16"],
    },
    # Text Models
    "text_cnn": {
        "class": TextCNN,
        "domain": "text",
        "supported_datasets": ["ag_news", "trec"],
        # Define required parameters for text models
        "required_params": ["vocab_size", "num_classes", "padding_idx"]
    }
}


# --- 2. Create a single, unified get_model factory ---
def get_model(
        model_name: str,
        dataset_name: str,  # Pass name instead of full dataset object
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any  # Pass all other params like num_classes, vocab_size here
) -> nn.Module:
    """
    Gets an initialized model instance from the central registry.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise NotImplementedError(f"Model '{model_name}' not in registry.")

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info["class"]

    if dataset_name.lower() not in model_info["supported_datasets"]:
        raise ValueError(f"Model '{model_name}' does not support '{dataset_name}'.")

    # Filter kwargs to only pass what the model constructor needs
    # This avoids "unexpected keyword argument" errors
    import inspect
    sig = inspect.signature(model_class.__init__)
    model_params = {p.name for p in sig.parameters.values()}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_params}

    # Instantiate the model
    model = model_class(**filtered_kwargs)

    if device:
        model.to(torch.device(device) if isinstance(device, str) else device)

    return model


# --- 0c. Flexible Model Dispatcher Function ---

# In the file where get_image_model is defined

def get_image_model(
        model_name: str,
        num_classes: int,
        in_channels: int,
        device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Gets an initialized model instance from the registry using provided parameters.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"Model '{model_name}' is not in the registry. Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]["class"]

    print(
        f"Instantiating model '{model_name}' with {in_channels} channels and {num_classes} classes.")
    model = model_class(in_channels=in_channels, num_classes=num_classes)

    if device:
        model.to(torch.device(device) if isinstance(device, str) else device)
        print(f"Model moved to device: {device}")

    return model


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
