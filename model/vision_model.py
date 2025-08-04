#!/usr/bin/env python3
"""
cnn_federated.py

This file defines two CNN models:
  - CNN_FMNIST for Fashion MNIST (28x28, single-channel)
  - CNN_CIFAR for CIFAR-10 (32x32, 3-channel)

It also provides functions to:
  - Load the FMNIST and CIFAR-10 datasets with appropriate transforms.
  - Train a local model and compute the gradient update (difference between
    trained model and initial model).
  - Flatten the gradient into a single vector (useful for aggregation).
  - Save and load model states (to support federated rounds).

Usage:
    from cnn_federated import (
         CNN_FMNIST, CNN_CIFAR,
         get_fmnist_dataloaders, get_cifar_dataloaders,
         local_training_and_get_gradient,
         save_model, load_model
    )

    # For Fashion MNIST:
    train_loader, test_loader = get_fmnist_dataloaders(batch_size=64)
    model = CNN_FMNIST(num_classes=10)
    model.to(device)
    flat_update, data_size = local_training_and_get_gradient(
                                  model, train_loader, device=device,
                                  local_epochs=1, lr=0.01)
    save_model(model, "saved_models/fmnist_round1.pt")

    # For CIFAR-10:
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)
    model = CNN_CIFAR(num_classes=10)
    model.to(device)
    flat_update, data_size = local_training_and_get_gradient(
                                  model, train_loader, device=device,
                                  local_epochs=1, lr=0.01)
    save_model(model, "saved_models/cifar_round1.pt")
"""

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class TextCNN(nn.Module):
    """
    A Text Convolutional Neural Network (TextCNN) model for text classification.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        embed_dim (int): The dimension of the word embeddings.
        num_filters (int): The number of filters (output channels) for each convolution kernel size.
        filter_sizes (List[int]): A list of kernel heights (e.g., [3, 4, 5] corresponding to 3-grams, 4-grams, 5-grams).
        num_class (int): The number of output classes for classification.
        dropout (float): The dropout probability applied to the concatenated pooled features.
        padding_idx (int): The index of the padding token in the vocabulary, used by the embedding layer.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_filters: int,
                 filter_sizes: List[int],
                 num_class: int,
                 dropout: float = 0.5,
                 padding_idx: int = 1,  # Default assumes <pad> is often index 1, but should be passed correctly
                 concise=True,
                 ):
        super(TextCNN, self).__init__()

        # --- Validation ---
        if not vocab_size > 0:
            raise ValueError("vocab_size must be greater than 0")
        if not embed_dim > 0:
            raise ValueError("embed_dim must be greater than 0")
        if not num_class > 0:
            raise ValueError("num_class must be greater than 0")
        if not filter_sizes:
            raise ValueError("filter_sizes list cannot be empty")
        if not all(fs > 0 for fs in filter_sizes):
            raise ValueError("All filter_sizes must be positive integers")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Create convolution layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,  # Input channels (1 for text)
                out_channels=num_filters,  # Number of filters
                kernel_size=(fs, embed_dim)  # Kernel: (height, width) = (filter_size, embed_dim)
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Fully connected layer: input size is total number of filters across all kernel sizes
        total_output_filters = num_filters * len(filter_sizes)
        self.fc = nn.Linear(total_output_filters, num_class)
        if not concise:
            print(f"Initialized TextCNN:")
            print(f"  Vocab Size: {vocab_size}")
            print(f"  Embed Dim: {embed_dim}")
            print(f"  Num Filters per Size: {num_filters}")
            print(f"  Filter Sizes: {filter_sizes}")
            print(f"  Total Output Filters: {total_output_filters}")
            print(f"  Num Classes: {num_class}")
            print(f"  Dropout: {dropout}")
            print(f"  Padding Index: {padding_idx}")

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = []
        for i, conv in enumerate(self.convs):
            try:
                conv_out = F.relu(conv(embedded))
                conved.append(conv_out)
            except Exception as e:
                print(f"!!!!! ERROR applying conv {i} !!!!!")
                print(f"Conv layer details: {conv}")  # Print the layer config
                print(f"Input shape was: {embedded.shape}")
                raise e  # Re-raise the exception
        # 1. Embedding
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_len, embed_dim)

        # 2. Prepare for Conv2d
        # Add a channel dimension: (batch_size, 1, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)

        # 3. Convolution + Activation + Pooling for each filter size
        # Apply each conv layer to the embedded input
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[i] shape: (batch_size, num_filters, seq_len - filter_sizes[i] + 1, 1)

        # Apply max-pooling over the sequence length dimension
        # Pool kernel size should cover the entire height of the convolved output
        pooled = [F.max_pool2d(conv_out, (conv_out.shape[2], 1)).squeeze(3).squeeze(2)
                  for conv_out in conved]
        # pooled[i] shape: (batch_size, num_filters)

        # 4. Concatenate Pooled Features & Apply Dropout
        # Concatenate along the feature dimension (num_filters)
        cat = torch.cat(pooled, dim=1)
        # cat shape: (batch_size, num_filters * len(filter_sizes))

        dropped = self.dropout(cat)

        # 5. Fully Connected Layer (Output)
        logits = self.fc(dropped)
        # logits shape: (batch_size, num_class)

        return logits


def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> nn.Module:
    """
    Train the model on the given DataLoader for a specified number of epochs.
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
                            trained_model: nn.Module) -> list:
    """
    Compute the gradient update as the difference between the trained model's parameters
    and the initial model's parameters. Returns a list of tensors.
    """
    grad_update = []
    for init_param, trained_param in zip(initial_model.parameters(), trained_model.parameters()):
        grad_update.append(trained_param.detach().cpu() - init_param.detach().cpu())
    return grad_update


def flatten_gradients(grad_list: list) -> np.ndarray:
    """
    Flatten a list of gradient tensors into a single 1D numpy array.
    """
    flat_grad = torch.cat([g.view(-1) for g in grad_list])
    return flat_grad.numpy()


def local_training_and_get_gradient(model: nn.Module,
                                    train_loader: DataLoader,
                                    device: torch.device,
                                    local_epochs: int = 1, batch_size=64,
                                    lr: float = 0.01) -> tuple:
    """
    Perform local training on a copy of the model using the provided DataLoader.
    Returns:
      - flat_update: flattened numpy array representing the update (trained - initial)
      - data_size: number of samples used in training.
    """
    local_model = copy.deepcopy(model)
    local_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=lr)

    # Save initial parameters
    initial_model = copy.deepcopy(local_model)

    # Train locally
    local_model = train_local_model(local_model, train_loader, criterion, optimizer, device, epochs=local_epochs)

    # Compute gradient update (i.e., parameter difference)
    grad_update = compute_gradient_update(initial_model, local_model)

    # Flatten the gradient list into a single vector
    flat_update = flatten_gradients(grad_update)

    data_size = len(train_loader.dataset)
    return flat_update, data_size


class SimpleCNN(nn.Module):
    """A generic CNN that can be adapted for various image datasets."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # To make this adaptable, we use an AdaptiveAvgPool2d layer
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet(nn.Module):
    """A classic LeNet architecture, adapted for different input sizes and channels."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
