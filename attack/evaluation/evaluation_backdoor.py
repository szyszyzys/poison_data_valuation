import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def evaluate_attack_performance_backdoor_poison(model, clean_loader, triggered_loader, device, target_label=None,
                                                plot=True):
    """
    Evaluate the final model performance in a poisoning/backdoor scenario.

    Parameters:
      model         : PyTorch model.
      clean_loader  : DataLoader for the clean test set.
      triggered_loader: DataLoader for the triggered (backdoor) test set.
      device        : torch.device.
      target_label  : (Optional) integer target label for backdoor attack.
                      If provided, attack success is defined as the fraction of triggered
                      samples predicted as this target label.
      plot          : If True, create visualizations.

    Returns:
      metrics       : Dictionary containing:
                        - 'clean_accuracy': Accuracy on clean data.
                        - 'triggered_accuracy': Accuracy on triggered data.
                        - 'attack_success_rate': (if target_label provided) Fraction of triggered samples
                                                 classified as target_label.
                        - 'confusion_matrix_triggered': Confusion matrix for triggered test set.
    """

    model.eval()

    # Evaluate on clean test set
    all_clean_preds = []
    all_clean_labels = []
    with torch.no_grad():
        for X, y in clean_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            all_clean_preds.append(preds.cpu().numpy())
            all_clean_labels.append(y.cpu().numpy())
    all_clean_preds = np.concatenate(all_clean_preds)
    all_clean_labels = np.concatenate(all_clean_labels)
    clean_accuracy = np.mean(all_clean_preds == all_clean_labels)

    # Evaluate on triggered test set
    all_triggered_preds = []
    all_triggered_labels = []
    with torch.no_grad():
        for X, y in triggered_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            all_triggered_preds.append(preds.cpu().numpy())
            all_triggered_labels.append(y.cpu().numpy())
    all_triggered_preds = np.concatenate(all_triggered_preds)
    all_triggered_labels = np.concatenate(all_triggered_labels)
    triggered_accuracy = np.mean(all_triggered_preds == all_triggered_labels)

    # Compute attack success rate if target_label is provided.
    attack_success_rate = None
    if target_label is not None:
        attack_success_rate = np.mean(all_triggered_preds == target_label)

    # Compute confusion matrix for triggered data.
    conf_matrix = confusion_matrix(all_triggered_labels, all_triggered_preds)

    # Prepare metrics dictionary.
    metrics = {
        "clean_accuracy": clean_accuracy,
        "triggered_accuracy": triggered_accuracy,
        "attack_success_rate": attack_success_rate,
        "confusion_matrix_triggered": conf_matrix
    }

    # Visualization: Bar chart for accuracies and attack success rate; heatmap for confusion matrix.
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Bar Chart: Clean accuracy, Triggered accuracy, (if available) Attack success rate.
        labels = ["Clean Accuracy", "Triggered Accuracy"]
        values = [clean_accuracy * 100, triggered_accuracy * 100]
        if attack_success_rate is not None:
            labels.append("Attack Success Rate")
            values.append(attack_success_rate * 100)

        ax[0].bar(labels, values, color=["blue", "orange", "red"][:len(labels)])
        ax[0].set_ylabel("Percentage (%)")
        ax[0].set_title("Model Performance Comparison")
        for i, v in enumerate(values):
            ax[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)

        # Confusion Matrix: Heatmap for triggered test set.
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax[1])
        ax[1].set_title("Confusion Matrix (Triggered Data)")
        ax[1].set_xlabel("Predicted Label")
        ax[1].set_ylabel("True Label")

        plt.tight_layout()
        plt.show()

    return metrics


# Example usage (to be run separately, not in the module if imported):
if __name__ == "__main__":
    # Dummy example: Replace with your actual model and dataloaders.
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Create a dummy model (e.g., a simple CNN for demonstration)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dummy clean and triggered datasets.
    # Here we simulate 100 samples of 28x28 images.
    X_clean = torch.randn(100, 1, 28, 28)
    y_clean = torch.randint(0, 10, (100,))
    clean_dataset = TensorDataset(X_clean, y_clean)
    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False)

    # For triggered data, simulate the same data but assume that the attacker’s
    # target label is 0. For simplicity, we use the same images but set the labels to 0.
    X_triggered = torch.randn(100, 1, 28, 28)
    y_triggered = torch.zeros(100, dtype=torch.long)
    triggered_dataset = TensorDataset(X_triggered, y_triggered)
    triggered_loader = DataLoader(triggered_dataset, batch_size=32, shuffle=False)

    # Evaluate with target label 0 for backdoor attack.
    metrics = evaluate_attack_performance_backdoor_poison(model, clean_loader, triggered_loader, device, target_label=0,
                                                          plot=True)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix_triggered":
            print(f"  {k}: {v}")
