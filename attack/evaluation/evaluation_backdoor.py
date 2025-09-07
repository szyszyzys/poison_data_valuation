import logging
from collections.abc import Sequence
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


# --- Helper functions (These are already excellent, no changes needed) ---

def _move_to_device(x, device):
    """Recursively move tensors inside lists / dicts to device."""
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, dict): return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_move_to_device(v, device) for v in x)
    return x


def _forward(model, batch):
    """Handles model forward pass for tensor or dict inputs."""
    return model(**batch) if isinstance(batch, dict) else model(batch)


def _split_batch(batch: Sequence) -> Tuple[Any, torch.Tensor]:
    """Auto-detects batch format and returns (inputs, labels)."""
    a, b = batch[0], batch[1]
    if torch.is_tensor(a) and a.dtype in (torch.int32, torch.int64) and a.dim() <= 1:
        return b, a  # (labels, inputs) format
    return a, b  # (inputs, labels) format


# --- The Refined Evaluation Function ---

def evaluate_attack_performance(
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: str,
        poison_generator: Any,
        target_label: int,
        plot: bool = True,
        save_path: str = "attack_performance.png",
) -> Dict[str, Any]:
    """
    Evaluates model robustness against a poison attack using a modular PoisonGenerator.

    This function is agnostic to the attack type (image, text, label-flip)
    and handles various data formats.

    Args:
        model: The model to evaluate.
        test_loader: DataLoader for the test set.
        device: The device to run evaluation on.
        poison_generator: An object with an `apply(data, label)` method (our standard interface).
        target_label: The integer target label for the attack.
        plot: If True, generates and saves a plot of the results.
        save_path: Path to save the visualization plot.

    Returns:
        A dictionary containing performance metrics.
    """
    model.eval()
    clean_preds, clean_labels = [], []
    poison_preds, poison_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = _split_batch(batch)
            inputs, labels = _move_to_device(inputs, device), labels.to(device)

            # 1. Evaluate on Clean Data
            clean_outputs = _forward(model, inputs)
            clean_preds.append(clean_outputs.argmax(dim=1).cpu().numpy())
            clean_labels.append(labels.cpu().numpy())

            # 2. Evaluate on Poisoned Data
            # Apply the poison sample-by-sample and re-collate the batch.
            # This cleanly uses the standard PoisonGenerator interface.
            poisoned_input_list, poisoned_label_list = [], []
            # Handle dictionary-style inputs (e.g., from HuggingFace)
            if isinstance(inputs, dict):
                # Unzip the dict of batches into a list of sample dicts
                input_sample_dicts = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
                for i in range(len(labels)):
                    p_data, p_label = poison_generator.apply(input_sample_dicts[i], labels[i].item())
                    poisoned_input_list.append(p_data)
                    poisoned_label_list.append(p_label)
                # Re-zip the list of sample dicts back into a batch dict
                poisoned_inputs = {k: torch.stack([dic[k] for dic in poisoned_input_list]) for k in
                                   poisoned_input_list[0]}
            else:  # Handle tensor inputs
                for i in range(len(labels)):
                    p_data, p_label = poison_generator.apply(inputs[i], labels[i].item())
                    poisoned_input_list.append(p_data)
                    poisoned_label_list.append(p_label)
                poisoned_inputs = torch.stack(poisoned_input_list)

            poison_outputs = _forward(model, poisoned_inputs)
            poison_preds.append(poison_outputs.argmax(dim=1).cpu().numpy())
            poison_labels.append(labels.cpu().numpy())  # ASR is w.r.t original labels

    # 3. Compute Metrics
    clean_preds = np.concatenate(clean_preds)
    clean_labels = np.concatenate(clean_labels)
    poison_preds = np.concatenate(poison_preds)

    clean_accuracy = np.mean(clean_preds == clean_labels)
    attack_success_rate = np.mean(poison_preds == target_label)

    metrics = {
        "clean_accuracy": float(clean_accuracy),
        "attack_success_rate": float(attack_success_rate),
    }

    # 4. Generate Visualization
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        labels_bar = ["Clean Accuracy", "Attack Success Rate"]
        values_bar = [metrics["clean_accuracy"] * 100, metrics["attack_success_rate"] * 100]
        ax[0].bar(labels_bar, values_bar, color=["steelblue", "crimson"])
        ax[0].set_ylabel("Percentage (%)")
        ax[0].set_ylim(0, 100)
        ax[0].set_title("Model Performance")
        for i, v in enumerate(values_bar):
            ax[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight='bold')

        conf_mat = confusion_matrix(clean_labels, poison_preds)
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax[1],
                    xticklabels=range(conf_mat.shape[1]), yticklabels=range(conf_mat.shape[0]))
        ax[1].set_xlabel("Predicted Label (on Poisoned Data)")
        ax[1].set_ylabel("True Original Label")
        ax[1].set_title("Poisoned Confusion Matrix")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Evaluation plot saved to {save_path}")

    return metrics
