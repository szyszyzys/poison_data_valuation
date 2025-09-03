#!/usr/bin/env python3
"""
Refactored models and training utilities for federated learning.

This script provides:
  - Extendable image models (LeNet, SimpleCNN) using adaptive pooling.
  - A TextCNN model for NLP tasks.
  - A model factory `get_model` to easily instantiate models by name.
  - A flexible local training function that accepts different optimizers and loss functions.
"""
import copy
from typing import List, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# ======================================================================================
#  1. MODEL DEFINITIONS
# ======================================================================================

class LeNet(nn.Module):
    """A classic LeNet architecture, adapted for different input sizes and channels."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Adaptive pooling makes the model robust to input size variations
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """A generic CNN adaptable for various image datasets."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Adaptive pooling ensures a fixed size output for the linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TextCNN(nn.Module):
    """A TextCNN model for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int, num_filters: int,
                 filter_sizes: List[int], num_class: int, dropout: float = 0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: (batch_size, seq_len)
        embedded = self.embedding(text)
        # embedded: (batch_size, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)
        # embedded: (batch_size, 1, seq_len, embed_dim)

        # Convolve, activate, and pool
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[i]: (batch_size, num_filters, seq_len - fs + 1, 1)
        pooled = [F.max_pool2d(conv, (conv.shape[2], conv.shape[3])).squeeze(3).squeeze(2) for conv in conved]
        # pooled[i]: (batch_size, num_filters)

        # Concatenate, dropout, and classify
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: (batch_size, num_filters * len(filter_sizes))
        return self.fc(cat)


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Model factory to return an instance of a specified model.

    Args:
        name (str): The name of the model ('lenet', 'simple_cnn', 'text_cnn').
        **kwargs: Arguments to pass to the model's constructor.

    Returns:
        An initialized nn.Module.
    """
    if name.lower() == 'lenet':
        return LeNet(**kwargs)
    elif name.lower() == 'simple_cnn':
        return SimpleCNN(**kwargs)
    elif name.lower() == 'text_cnn':
        return TextCNN(**kwargs)
    else:
        raise ValueError(f"Model '{name}' not recognized.")


# ======================================================================================
#  2. FEDERATED LEARNING UTILITY FUNCTIONS
# ======================================================================================

def local_training_and_get_gradient(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        local_epochs: int,
        lr: float,
        optimizer_cls: Type[optim.Optimizer] = optim.SGD,
        criterion: nn.Module = nn.CrossEntropyLoss()
) -> tuple[np.ndarray, int]:
    """
    Performs local training and computes the model update.

    Args:
        model: The global model to start training from.
        train_loader: DataLoader for the client's local data.
        device: The torch device (e.g., 'cuda' or 'cpu').
        local_epochs: The number of epochs to train locally.
        lr: Learning rate for the optimizer.
        optimizer_cls: The optimizer class to use (e.g., torch.optim.Adam).
        criterion: The loss function.

    Returns:
        A tuple containing:
        - flat_update (np.ndarray): A 1D numpy array of the model weights difference.
        - data_size (int): The number of samples used for training.
    """
    initial_model_state = copy.deepcopy(model.state_dict())
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()

    optimizer = optimizer_cls(local_model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Compute the difference between trained and initial model weights
    grad_update = []
    trained_state_dict = local_model.state_dict()
    for key in initial_model_state:
        diff = trained_state_dict[key].cpu() - initial_model_state[key].cpu()
        grad_update.append(diff.view(-1))

    flat_update = torch.cat(grad_update).numpy()
    data_size = len(train_loader.dataset)

    return flat_update, data_size