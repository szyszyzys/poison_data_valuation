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
from typing import List, Tuple, Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset

# from model.text_model import TEXTCNN
from model.vision_model import CNN_CIFAR, LeNet, TextCNN


def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> Tuple[nn.Module, float or None]: # Modified return type
    """
    Train the model on the given train_loader for a specified number of epochs.
    Calculates and returns the average loss over all batches trained.

    Args:
        model: The model to train (will be modified in place).
        train_loader: DataLoader for the training data.
        criterion: Loss function module.
        optimizer: Optimizer instance.
        device: The torch device ('cuda' or 'cpu').
        epochs: Number of local epochs to train.

    Returns:
        Tuple (trained_model, average_loss):
            trained_model: The model after training (same object as input model).
            average_loss: The average loss across all batches and epochs.
                          Returns None if train_loader was empty or no batches completed.
    """
    model.train() # Set model to training mode
    batch_losses_all = [] # Collect losses from ALL batches across ALL epochs


    logging.debug(f"Starting local training for {epochs} epochs...")
    for epoch in range(epochs):
        num_batches_processed_epoch = 0
        epoch_start_time = time.time() # Optional: time epochs

        for batch_idx, batch_data in enumerate(train_loader):
            # --- Standard Training Batch ---
            # Handle different data formats (vision vs text with lengths)
            if len(batch_data) == 3 and isinstance(batch_data[2], torch.Tensor): # Text format check
                data, labels, _ = batch_data # Ignore lengths for basic training loss calc
            elif len(batch_data) == 2: # Vision format
                data, labels = batch_data
            else:
                 logging.warning(f"Unexpected batch data format len {len(batch_data)} in epoch {epoch}, batch {batch_idx}. Skipping.")
                 continue # Skip batch

            try:
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)

                # Check for NaN loss
                if torch.isnan(loss):
                    logging.warning(f"NaN loss encountered in epoch {epoch}, batch {batch_idx}. Skipping batch update.")
                    continue # Skip optimizer step if loss is NaN

                loss.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=...)
                optimizer.step()

                batch_losses_all.append(loss.item()) # Append loss of this batch
                num_batches_processed_epoch += 1

            except Exception as batch_e:
                logging.error(f"Error during batch {batch_idx} in epoch {epoch}: {batch_e}", exc_info=True)
                # Optionally break or continue depending on desired robustness
                continue # Skip to next batch on error

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss_display = np.mean(batch_losses_all[-num_batches_processed_epoch:]) if num_batches_processed_epoch > 0 else float('nan')
        logging.debug(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s. Avg Loss (epoch): {avg_epoch_loss_display:.4f}")


    # Calculate overall average loss from all recorded batch losses
    if not batch_losses_all:
        logging.warning("No batches were successfully processed during training.")
        overall_avg_loss = None
    else:
        overall_avg_loss = np.mean(batch_losses_all)
        logging.debug(f"Finished local training. Overall Avg Loss: {overall_avg_loss:.4f}")

    # Model was trained in-place, so we return the same object
    # Optionally set to eval mode if needed, but train mode is often fine before grad calc
    # model.eval()
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
    """
    Evaluate the model on the given test_loader.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function (e.g., nn.CrossEntropyLoss).
        device (torch.device): Device on which to perform evaluation.

    Returns:
        dict: A dictionary containing 'loss' and 'accuracy' for the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_data.size(0)

            # Compute the number of correct predictions
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


# def local_training_and_get_gradient(model: nn.Module,
#                                     train_dataset: TensorDataset,
#                                     batch_size: int,
#                                     device: torch.device,
#                                     local_epochs: int = 1,
#                                     lr: float = 0.01, opt="SGD", momentum=0.9, weight_decay=0.0005):
#     """
#     Perform local training on a copy of the given model using the provided dataset.
#     Returns:
#       - flat_update: a flattened numpy array representing the gradient update (trained - initial)
#       - data_size: the number of samples in the train_dataset
#
#     This function is intended to be used by a seller in a federated learning setup.
#     """
#     # Create a local copy of the model for training
#     local_model = copy.deepcopy(model)
#     local_model.to(device)
#
#     # Create a DataLoader for the local dataset
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#     # Use a standard loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     if opt == "SGD":
#         optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
#     elif opt == "ADAM":
#         optimizer = optim.Adam(local_model.parameters(), lr=lr)
#     else:
#         raise NotImplementedError(f"No such optimizer: {opt}")
#     # Save a copy of the initial model parameters for computing the update
#     initial_model = copy.deepcopy(local_model)
#
#     # Train the local model for a few epochs
#     local_model = train_local_model(local_model, train_loader, criterion, optimizer, device, epochs=local_epochs)
#
#     # Compute the gradient update as (trained_model - initial_model)
#     grad_update = compute_gradient_update(initial_model, local_model)
#
#     # Flatten the list of gradients into a single vector
#     flat_update = flatten_gradients(grad_update)
#
#     # evaluate the model
#     eval_res_o = test_local_model(initial_model, train_loader, criterion, device)
#     eval_res = test_local_model(local_model, train_loader, criterion, device)
#     print(f"evaluation_result before local train: {eval_res_o}")
#     print(f"evaluation_result after local train: {eval_res}")
#
#     return grad_update, flat_update, local_model, eval_res

def local_training_and_get_gradient(model: nn.Module,
                                    train_dataset: TensorDataset,
                                    batch_size: int,
                                    device: torch.device,
                                    local_epochs: int = 1,
                                    lr: float = 0.01,
                                    opt: str = "SGD", # Changed opt typo to opt
                                    momentum: float = 0.9,
                                    weight_decay: float = 0.0005
                                    ) -> Tuple[Any, Any, nn.Module, Dict, float or None]: # Added float|None for avg_loss
    """
    MODIFIED: Perform local training and return gradient, model, eval results, AND avg loss.

    Requires train_local_model to return (trained_model, average_loss).

    Returns:
        Tuple (gradient, flattened_gradient, updated_model, eval_results, avg_train_loss)
    """
    # Create a local copy of the model for training
    local_model = copy.deepcopy(model)
    local_model.to(device)
    initial_model = copy.deepcopy(local_model) # Keep initial state for gradient calc

    # Create a DataLoader for the local dataset
    # Handle potential empty dataset
    if len(train_dataset) == 0:
        logging.warning("Received empty dataset for local training. Returning zero gradient.")
        # Return zero gradient of the same structure as model params
        zero_grad = OrderedDict([(n, np.zeros_like(p.cpu().detach().numpy())) for n, p in model.named_parameters()])
        zero_flat = flatten_gradients(zero_grad)
        return zero_grad, zero_flat, local_model, {"acc": float('nan'), "loss": float('nan')}, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Use a standard loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if opt.upper() == "SGD": # Use .upper() for case-insensitivity
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt.upper() == "ADAM":
        optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Defaulting or raising error is better than NotImplementedError in except block later
        logging.warning(f"Unsupported optimizer: {opt}. Defaulting to SGD.")
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


    # --- Call MODIFIED train_local_model ---
    # It now returns the average training loss as the second value
    try:
        local_model, avg_train_loss = train_local_model(
            local_model, train_loader, criterion, optimizer, device, epochs=local_epochs
        )
    except Exception as e:
         logging.error(f"Error during train_local_model call: {e}", exc_info=True)
         # Return zero gradient and None loss on error
         zero_grad = OrderedDict([(n, np.zeros_like(p.cpu().detach().numpy())) for n, p in model.named_parameters()])
         zero_flat = flatten_gradients(zero_grad)
         return zero_grad, zero_flat, local_model, {"acc": float('nan'), "loss": float('nan')}, None
    # --------------------------------------

    # Compute the gradient update as (trained_model - initial_model)
    grad_update = compute_gradient_update(initial_model, local_model)

    # Flatten the list of gradients into a single vector
    flat_update = flatten_gradients(grad_update)

    # evaluate the model (optional, maybe use avg_train_loss instead?)
    eval_res_o = test_local_model(initial_model, train_loader, criterion, device)
    eval_res = test_local_model(local_model, train_loader, criterion, device)
    print(f"evaluation_result before local train: {eval_res_o}")
    print(f"evaluation_result after local train: {eval_res}")

    # Return original values PLUS the average training loss
    return grad_update, flat_update, local_model, eval_res, avg_train_loss

def apply_gradient_update(initial_model: nn.Module, grad_update: List[torch.Tensor]) -> nn.Module:
    """
    Create a new model by adding the computed gradient update to the initial model's parameters.
    The gradient update is assumed to be computed as (trained_model_param - initial_model_param),
    so adding it to the initial model should produce the trained model.

    Parameters:
        initial_model (nn.Module): The model before local training.
        grad_update (List[torch.Tensor]): List of gradient update tensors.

    Returns:
        nn.Module: A new model with updated parameters.
    """
    # Create a deep copy of the initial model to avoid in-place modification.
    updated_model = copy.deepcopy(initial_model)

    # Iterate over model parameters and add the corresponding gradient update.
    for param, delta in zip(updated_model.parameters(), grad_update):
        # Make sure delta is on the same device as param
        param.data.add_(delta.to(param.device))

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


def get_text_model(
    dataset_name: str,
    num_classes: int,
    vocab_size: Optional[int] = None,
    padding_idx: Optional[int] = None,
    **model_kwargs: Any # Use kwargs for model-specific hyperparameters
) -> nn.Module:
    """
    Gets an appropriate model instance based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset (e.g., "CIFAR", "FMNIST", "AG_NEWS", "TREC").
        num_classes (int): The number of output classes required for the model.
        vocab_size (Optional[int]): The vocabulary size. Required for text models.
        padding_idx (Optional[int]): The padding index in the vocabulary. Required for text models.
        model_structure_name (str): Optional name for specific model variants (currently unused).
        **model_kwargs (Any): Additional keyword arguments passed directly to the model constructor.
                              Used for hyperparameters like embed_dim, num_filters, etc.

    Returns:
        nn.Module: An instance of the appropriate neural network model.

    Raises:
        NotImplementedError: If no model is defined for the given dataset_name.
        ValueError: If required arguments (like vocab_size for text) are missing.
    """
    print(f"Getting model for dataset: {dataset_name}")

    model: nn.Module # Type hint for the returned model

    match dataset_name.lower():
        case "ag_news" | "trec":
            print(f"Initializing TextCNN for {num_classes} classes.")
            # --- Text Model Configuration ---
            if vocab_size is None:
                raise ValueError("`vocab_size` is required for TextCNN model.")
            if padding_idx is None:
                raise ValueError("`padding_idx` is required for TextCNN model.")

            # Extract hyperparameters from kwargs or use defaults
            embed_dim = model_kwargs.get("embed_dim", 100)
            num_filters = model_kwargs.get("num_filters", 100)
            filter_sizes = model_kwargs.get("filter_sizes", [3, 4, 5])
            dropout = model_kwargs.get("dropout", 0.5)

            # Validate types if necessary (e.g., filter_sizes should be list)
            if not isinstance(filter_sizes, list):
                raise TypeError(f"Expected 'filter_sizes' to be a list, got {type(filter_sizes)}")

            model = TextCNN(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                num_class=num_classes,
                dropout=dropout,
                padding_idx=padding_idx
            )
        case _:
            raise NotImplementedError(f"Cannot find a model for dataset {dataset_name}")

    return model



def get_image_model(dataset_name, model_structure_name=""):
    match dataset_name.lower():
        case "cifar":
            model = CNN_CIFAR()
        case "fmnist":
            model = LeNet()
        case _:
            raise NotImplementedError(f"Cannot find the model for dataset {dataset_name}")
    return model

def get_model_name(dataset_name):
    match dataset_name.lower():
        case "cifar":
            model = 'CNN'
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
    domain: str # Variable to hold the result

    # Convert to lower once for case-insensitive matching
    dataset_name_lower = dataset_name.lower()

    match dataset_name_lower:
        case "cifar":
            domain = 'image'
        case "fmnist":
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
            raise NotImplementedError(f"Cannot determine domain for dataset '{dataset_name}'") # Added quotes for clarity

    return domain


import torch
import numpy as np
from collections import OrderedDict


def apply_gradient(model, aggregated_gradient, learning_rate: float = 1.0, device: torch.device = None):
    """
    Update the model parameters by descending along the aggregated gradient.

    This function supports two cases:
      1. If `model` is an instance of nn.Module, the model's state_dict will be updated.
      2. If `model` is an OrderedDict (i.e. a state_dict), then the dict will be updated directly.

    Args:
        model: The model (nn.Module) or state_dict (OrderedDict) to update.
        aggregated_gradient: The aggregated gradient, either as a list of tensors
                             or as a flattened numpy array.
        learning_rate (float): The step size for the update.
        device (torch.device, optional): The device on which to perform the update.
            If None, the device is inferred from the model or state_dict.

    Returns:
        The updated model (if model was an nn.Module) or an updated state_dict (if model was an OrderedDict).
    """
    # Determine if model is a module or a state_dict.
    if hasattr(model, 'parameters'):
        # model is an nn.Module
        if device is None:
            device = next(model.parameters()).device
        current_params = model.state_dict()
        is_module = True
    elif isinstance(model, dict):
        # model is a state_dict (OrderedDict)
        if device is None:
            device = list(model.values())[0].device
        current_params = model
        is_module = False
    else:
        raise ValueError("model must be an nn.Module or a state_dict (OrderedDict).")

    # If aggregated_gradient is a list of tensors, flatten and convert to a numpy array.
    if isinstance(aggregated_gradient, list):
        aggregated_gradient = np.concatenate(
            [grad.cpu().numpy().ravel() for grad in aggregated_gradient]
        )

    # If the aggregated gradient is empty, return the model/state_dict unchanged.
    if aggregated_gradient.size == 0:
        return model

    # Convert the numpy array back to a torch tensor.
    aggregated_torch = torch.from_numpy(aggregated_gradient).float().to(device)

    # Update the parameters by slicing the aggregated gradient accordingly.
    updated_params = OrderedDict()
    idx = 0
    for name, tensor in current_params.items():
        numel = tensor.numel()
        grad_slice = aggregated_torch[idx: idx + numel].reshape(tensor.shape)
        idx += numel
        # Apply the SGD update: new_param = param - learning_rate * grad_slice
        updated_params[name] = tensor - learning_rate * grad_slice

    # If we started with a module, load the updated state dict back into the model.
    if is_module:
        model.load_state_dict(updated_params)
        return model
    else:
        # Otherwise, return the updated state_dict.
        return updated_params


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
