import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_loss_and_pred(model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Calculates loss and predictions for a given model and data loader."""
    model.eval()
    losses = []
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            losses.extend(loss.cpu().numpy())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return np.array(losses), np.array(predictions), np.array(true_labels)


def extract_mia_features(
        global_model_state_dict: Dict[str, Any],
        individual_grad: torch.Tensor,
        sample_index: int,
        true_label: int,
        dataset: Dataset,  # The full dataset to get specific sample
        seller_round_history: pd.DataFrame,  # For weights/payments
        round_number: int,
        device: str,
        ModelClass: nn.Module,
        learning_rate: float
) -> Optional[Dict[str, Any]]:
    """
    Extracts features for Membership Inference Attack for a single sample and seller.
    Assumes the global model is trained on CrossEntropyLoss.
    """
    model = ModelClass().to(device)
    model.load_state_dict(global_model_state_dict)
    model.eval()

    # Get the specific sample
    try:
        sample_input, _ = dataset[sample_index]
        sample_input = sample_input.unsqueeze(0).to(device)  # Add batch dimension
    except IndexError:
        logging.error(f"Sample index {sample_index} not found in dataset.")
        return None

    # Calculate pre-update loss and confidence
    with torch.no_grad():
        outputs_pre = model(sample_input)
        loss_pre = nn.CrossEntropyLoss()(outputs_pre, torch.tensor([true_label]).to(device)).item()
        confidence_pre = torch.softmax(outputs_pre, dim=1)[0, true_label].item()

    # Simulate model update with individual gradient
    # This requires applying the gradient to the global model parameters
    # This is a simplification; a more accurate way would be to get the *model update* from the seller
    # However, for gradient-based attacks, we use the gradient directly.
    # We are simulating the *effect* if this gradient was applied directly.
    temp_model = ModelClass().to(device)
    temp_model.load_state_dict(global_model_state_dict)
    optimizer = torch.optim.SGD(temp_model.parameters(), lr=learning_rate)

    # Apply individual_grad to temp_model (requires careful handling)
    # This part is tricky. A global model update from a *single* seller gradient is not typical in FL.
    # More realistic: The buyer sees the *aggregated* gradient, and wants to infer membership
    # based on the *difference* in global model state, or by trying to approximate a seller's model.
    # For a direct "buyer sees individual grad" scenario:
    with torch.no_grad():
        for i, (name, param) in enumerate(temp_model.named_parameters()):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.data.copy_(individual_grad[i])  # Assuming individual_grad is a list/tuple of tensors

    optimizer.step()

    # Calculate post-update loss and confidence
    temp_model.eval()
    with torch.no_grad():
        outputs_post = temp_model(sample_input)
        loss_post = nn.CrossEntropyLoss()(outputs_post, torch.tensor([true_label]).to(device)).item()
        confidence_post = torch.softmax(outputs_post, dim=1)[0, true_label].item()

    # Marketplace Features
    # Filter seller_round_history for the current round
    round_data = seller_round_history[seller_round_history['round'] == round_number]
    assigned_weight = round_data['assigned_weight'].iloc[0] if not round_data.empty else 0.0
    payment_received = round_data['payment_received'].iloc[0] if not round_data.empty else 0.0

    features = {
        "loss_pre": loss_pre,
        "loss_post": loss_post,
        "loss_diff": loss_pre - loss_post,
        "confidence_pre": confidence_pre,
        "confidence_post": confidence_post,
        "confidence_diff": confidence_post - confidence_pre,
        "assigned_weight": assigned_weight,
        "payment_received": payment_received,
        "gradient_norm": torch.norm(torch.cat([g.flatten() for g in individual_grad])).item()  # Norm of flattened grad
        # Add more features like statistics of gradient per layer if needed
    }
    return features


def extract_pia_features(
        individual_grad: torch.Tensor,
        seller_round_history: pd.DataFrame,
        round_number: int,
        ModelClass: nn.Module,  # Needed to interpret gradient structure
        device: str
) -> Optional[Dict[str, Any]]:
    """
    Extracts features for Property Inference Attack for a single seller.
    Focuses on gradient statistics and marketplace signals.
    """
    if individual_grad is None:
        return None

    features = {}

    # Gradient Statistics (example features)
    flattened_grad = torch.cat([g.flatten() for g in individual_grad])
    features["grad_norm"] = torch.norm(flattened_grad).item()
    features["grad_mean"] = torch.mean(flattened_grad).item()
    features["grad_std"] = torch.std(flattened_grad).item()
    features["grad_min"] = torch.min(flattened_grad).item()
    features["grad_max"] = torch.max(flattened_grad).item()

    # Marketplace Features
    round_data = seller_round_history[seller_round_history['round'] == round_number]
    features["assigned_weight"] = round_data['assigned_weight'].iloc[0] if not round_data.empty else 0.0
    features["payment_received"] = round_data['payment_received'].iloc[0] if not round_data.empty else 0.0

    # You could add more sophisticated features here, e.g.:
    # - Cosine similarity of this grad to a 'reference grad' for a known property
    # - Sub-layer specific gradient norms
    # - Changes in buyer model performance on a 'probe set' after this gradient (more complex)

    return features
