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
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# from model.text_model import TEXTCNN
from model.vision_model import CNN_CIFAR, LeNet


def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> nn.Module:
    """
    Train the model on the given train_loader for a specified number of epochs.
    """
    model.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    return model


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


def local_training_and_get_gradient(model: nn.Module,
                                    train_dataset: TensorDataset,
                                    batch_size: int,
                                    device: torch.device,
                                    local_epochs: int = 1,
                                    lr: float = 0.01, opt="SGD"):
    """
    Perform local training on a copy of the given model using the provided dataset.
    Returns:
      - flat_update: a flattened numpy array representing the gradient update (trained - initial)
      - data_size: the number of samples in the train_dataset

    This function is intended to be used by a seller in a federated learning setup.
    """
    # Create a local copy of the model for training
    local_model = copy.deepcopy(model)
    local_model.to(device)

    # Create a DataLoader for the local dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Use a standard loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if opt == "SGD":
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
    elif opt == "ADAM":
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"No such optimizer: {opt}")
    # Save a copy of the initial model parameters for computing the update
    initial_model = copy.deepcopy(local_model)

    # Train the local model for a few epochs
    local_model = train_local_model(local_model, train_loader, criterion, optimizer, device, epochs=local_epochs)

    # Compute the gradient update as (trained_model - initial_model)
    grad_update = compute_gradient_update(initial_model, local_model)

    # Flatten the list of gradients into a single vector
    flat_update = flatten_gradients(grad_update)

    # evaluate the model
    eval_res_o = test_local_model(local_model, train_loader, criterion, device)
    eval_res = test_local_model(local_model, train_loader, criterion, device)
    print(f"evaluation_result before local train: {eval_res_o}")
    print(f"evaluation_result after local train: {eval_res}")
    return grad_update, flat_update, local_model, eval_res


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


def get_model(dataset_name):
    match dataset_name:
        case "CIFAR":
            model = CNN_CIFAR()
        case "FMINIST":
            model = LeNet()
        # case ["TREC", "AG_NEWS"]:
        #     model = TEXTCNN
        case _:
            raise NotImplementedError(f"Cannot find the model for dataset {dataset_name}")
    return model


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
