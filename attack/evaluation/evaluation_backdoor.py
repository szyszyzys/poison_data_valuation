import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def evaluate_attack_performance_backdoor_poison(model, test_loader, device, backdoor_generator, target_label=None,
                                                plot=True, save_path="attack_performance.png"):
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
      plot          : If True, create and save visualizations.
      save_path     : File path to save the plot. If not provided, defaults to "attack_performance.png".

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
    all_triggered_preds = []
    all_triggered_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            all_clean_preds.append(preds.cpu().numpy())
            all_clean_labels.append(y.cpu().numpy())

            backdoored_data = backdoor_generator.apply_trigger_tensor(X)
            outputs = model(backdoored_data)
            preds = outputs.argmax(dim=1)
            all_triggered_preds.append(preds.cpu().numpy())
            all_triggered_labels.append(y.cpu().numpy())

    all_clean_preds = np.concatenate(all_clean_preds)
    all_clean_labels = np.concatenate(all_clean_labels)
    clean_accuracy = np.mean(all_clean_preds == all_clean_labels)
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
        # Save the figure instead of showing it.
        plt.savefig(save_path)
        plt.close(fig)

    return metrics

# def evaluate_attack_performance_backdoor_poison(model, clean_loader, triggered_loader, device, target_label=None,
#                                                 plot=True, save_path="attack_performance.png"):
#     """
#     Evaluate the final model performance in a poisoning/backdoor scenario.
#
#     Parameters:
#       model         : PyTorch model.
#       clean_loader  : DataLoader for the clean test set.
#       triggered_loader: DataLoader for the triggered (backdoor) test set.
#       device        : torch.device.
#       target_label  : (Optional) integer target label for backdoor attack.
#                       If provided, attack success is defined as the fraction of triggered
#                       samples predicted as this target label.
#       plot          : If True, create and save visualizations.
#       save_path     : File path to save the plot. If not provided, defaults to "attack_performance.png".
#
#     Returns:
#       metrics       : Dictionary containing:
#                         - 'clean_accuracy': Accuracy on clean data.
#                         - 'triggered_accuracy': Accuracy on triggered data.
#                         - 'attack_success_rate': (if target_label provided) Fraction of triggered samples
#                                                  classified as target_label.
#                         - 'confusion_matrix_triggered': Confusion matrix for triggered test set.
#     """
#
#     model.eval()
#
#     # Evaluate on clean test set
#     all_clean_preds = []
#     all_clean_labels = []
#     with torch.no_grad():
#         for X, y in clean_loader:
#             X = X.to(device)
#             y = y.to(device)
#             outputs = model(X)
#             preds = outputs.argmax(dim=1)
#             all_clean_preds.append(preds.cpu().numpy())
#             all_clean_labels.append(y.cpu().numpy())
#     all_clean_preds = np.concatenate(all_clean_preds)
#     all_clean_labels = np.concatenate(all_clean_labels)
#     clean_accuracy = np.mean(all_clean_preds == all_clean_labels)
#
#     # Evaluate on triggered test set
#     all_triggered_preds = []
#     all_triggered_labels = []
#     with torch.no_grad():
#         for X, y in triggered_loader:
#             X = X.to(device)
#             y = y.to(device)
#             outputs = model(X)
#             preds = outputs.argmax(dim=1)
#             all_triggered_preds.append(preds.cpu().numpy())
#             all_triggered_labels.append(y.cpu().numpy())
#     all_triggered_preds = np.concatenate(all_triggered_preds)
#     all_triggered_labels = np.concatenate(all_triggered_labels)
#     triggered_accuracy = np.mean(all_triggered_preds == all_triggered_labels)
#
#     # Compute attack success rate if target_label is provided.
#     attack_success_rate = None
#     if target_label is not None:
#         attack_success_rate = np.mean(all_triggered_preds == target_label)
#
#     # Compute confusion matrix for triggered data.
#     conf_matrix = confusion_matrix(all_triggered_labels, all_triggered_preds)
#
#     # Prepare metrics dictionary.
#     metrics = {
#         "clean_accuracy": clean_accuracy,
#         "triggered_accuracy": triggered_accuracy,
#         "attack_success_rate": attack_success_rate,
#         "confusion_matrix_triggered": conf_matrix
#     }
#
#     # Visualization: Bar chart for accuracies and attack success rate; heatmap for confusion matrix.
#     if plot:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#
#         # Bar Chart: Clean accuracy, Triggered accuracy, (if available) Attack success rate.
#         labels = ["Clean Accuracy", "Triggered Accuracy"]
#         values = [clean_accuracy * 100, triggered_accuracy * 100]
#         if attack_success_rate is not None:
#             labels.append("Attack Success Rate")
#             values.append(attack_success_rate * 100)
#
#         ax[0].bar(labels, values, color=["blue", "orange", "red"][:len(labels)])
#         ax[0].set_ylabel("Percentage (%)")
#         ax[0].set_title("Model Performance Comparison")
#         for i, v in enumerate(values):
#             ax[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)
#
#         # Confusion Matrix: Heatmap for triggered test set.
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax[1])
#         ax[1].set_title("Confusion Matrix (Triggered Data)")
#         ax[1].set_xlabel("Predicted Label")
#         ax[1].set_ylabel("True Label")
#
#         plt.tight_layout()
#         # Save the figure instead of showing it.
#         plt.savefig(save_path)
#         plt.close(fig)
#
#     return metrics
