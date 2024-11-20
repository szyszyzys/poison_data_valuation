import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from attack.general_attack.my_utils import evaluate_reconstruction


# def construct_fim(X_selected, weights):
#     """
#     Constructs the Fisher Information Matrix (FIM) from selected samples and their weights.
#
#     I(w) = sum_j w_j x_j x_j^T
#     """
#     fim = torch.zeros(X_selected.shape[1], X_selected.shape[1], device=X_selected.device)
#     for j in range(X_selected.shape[0]):
#         x_j = X_selected[j].unsqueeze(1)  # (n_features, 1)
#         fim += weights[j] * (x_j @ x_j.T)  # (n_features, n_features)
#     return fim

def construct_fim(X_selected, weights):
    """
    Constructs the Fisher Information Matrix (FIM) from selected samples and their weights.

    I(w) = sum_j w_j x_j x_j^T
    """
    if X_selected.shape[0] == 0:
        # If no selected samples, return a small identity matrix to prevent singularity
        return torch.eye(X_selected.shape[1], device=X_selected.device) * 1e-6
    fim = torch.zeros(X_selected.shape[1], X_selected.shape[1], device=X_selected.device)
    for j in range(X_selected.shape[0]):
        x_j = X_selected[j].unsqueeze(1)  # (n_features, 1)
        fim += weights[j] * (x_j @ x_j.T)  # (n_features, n_features)
    return fim


def optimize_test_samples_with_fim(
        X,
        selected_indices_list,
        unselected_indices_list,
        n_tests=3,
        n_iterations=5000,
        lr=1e-2,
        device='cpu',
        verbose=True,
        early_stop_threshold=1e-6,
        patience=100
):
    """
    Infers test samples using an optimization-based method incorporating the Fisher Information Matrix (FIM).

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features)
    - selected_indices_list (list of np.ndarray): Each element contains selected indices for a test sample
    - unselected_indices_list (list of np.ndarray): Each element contains unselected indices for a test sample
    - n_tests (int): Number of test samples to infer
    - n_iterations (int): Number of optimization steps
    - lr (float): Learning rate for optimizer
    - device (str): 'cpu' or 'cuda'
    - verbose (bool): If True, prints progress
    - early_stop_threshold (float): Threshold for early stopping based on loss improvement
    - patience (int): Number of iterations to wait for improvement before stopping

    Returns:
    - x_tests_opt (np.ndarray): Optimized test samples of shape (n_tests, n_features)
    """
    # Convert data to torch tensors and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # (n_samples, n_features)
    n_features = X.shape[1]

    # Initialize test samples as parameters to optimize
    # Initialize closer to the mean of selected samples with some noise
    initial_x = torch.zeros(n_tests, n_features, device=device)
    for i in range(n_tests):
        mean_selected = X_tensor[selected_indices_list].mean(dim=0, keepdim=True)
        initial_x[i] = mean_selected + 0.1 * torch.randn(n_features, device=device)

    x_tests_opt = nn.Parameter(initial_x)

    # Initialize optimizer
    optimizer = optim.Adam([x_tests_opt], lr=lr)

    # Define weights for loss components
    w_trace = 1.0  # Weight for trace objectives
    w_alignment = 1.0  # Weight for alignment objectives
    w_reg = 0.01  # Weight for regularization

    # Early stopping variables
    prev_loss = float('inf')
    counter = 0

    # Retrieve selected and unselected indices for the current test sample
    selected_indices = selected_indices_list
    unselected_indices = unselected_indices_list

    # Handle cases where there are no selected or unselected indices
    if selected_indices.shape[0] > 0:
        X_selected = X_tensor[selected_indices]  # (k, n_features)
        weights_selected = torch.ones(X_selected.shape[0], device=device)  # (k,)
    else:
        # If no selected samples, use a small identity matrix
        X_selected = torch.empty(0, n_features, device=device)
        weights_selected = torch.tensor([], device=device)

    if len(unselected_indices) > 0:
        X_unselected = X_tensor[unselected_indices]  # (n_samples - k, n_features)
        weights_unselected = torch.ones(X_unselected.shape[0], device=device) * 0.8  # Slightly lower weight
    else:
        # If no unselected samples, use a small identity matrix
        X_unselected = torch.empty(0, n_features, device=device)
        weights_unselected = torch.tensor([], device=device)

    # Construct FIM for selected and unselected samples
    fim_selected = construct_fim(X_selected, weights_selected)  # (n_features, n_features)
    fim_unselected = construct_fim(X_unselected, weights_unselected)  # (n_features, n_features)

    # Optimization loop
    for it in tqdm(range(n_iterations), desc="Optimizing Test Samples"):
        optimizer.zero_grad()
        loss = 0.0

        for i in range(n_tests):
            x_test = x_tests_opt[i]  # (n_features,)

            # Compute trace-based objectives
            trace_selected = torch.trace(fim_selected)
            trace_unselected = torch.trace(fim_unselected)

            # Compute alignment-based objectives
            alignment_selected = torch.matmul(x_test, torch.matmul(fim_selected, x_test))
            alignment_unselected = torch.matmul(x_test, torch.matmul(fim_unselected, x_test))

            # Aggregate loss components
            # Objective:
            # 1. Maximize trace_selected (information from selected)
            # 2. Minimize trace_unselected (information from unselected)
            # 3. Maximize alignment_selected (alignment with selected FIM)
            # 4. Minimize alignment_unselected (alignment with unselected FIM)
            # 5. Regularization to prevent trivial solutions

            loss += (-w_trace * trace_selected + w_trace * trace_unselected)
            loss += (-w_alignment * alignment_selected + w_alignment * alignment_unselected)
            loss += w_reg * torch.norm(x_test, p=2)

        # Average loss over test samples
        loss = loss / n_tests

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Early stopping based on loss improvement
        loss_value = loss.item()
        if loss_value < prev_loss - early_stop_threshold:
            prev_loss = loss_value
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping at iteration {it + 1} with loss {loss_value:.6f}")
                break

        # Logging
        if verbose and ((it + 1) % 500 == 0 or it == 0):
            print(f"Iteration {it + 1}/{n_iterations}, Loss: {loss_value:.4f}")

    # Detach and move to CPU for final output
    x_tests_opt_np = x_tests_opt.detach().cpu().numpy()

    return x_tests_opt_np


# def optimize_test_samples_with_fim(X, selected_indices_list, unselected_indices_list, n_tests=3,
#                                    n_iterations=5000, lr=1e-2, device='cpu'):
#     """
#     Infers test samples using an optimization-based method incorporating the Fisher Information Matrix (FIM).
#
#     Parameters:
#     - X: np.ndarray of shape (n_samples, n_features)
#     - selected_indices_list: list of np.ndarray, each containing selected indices for a test sample
#     - n_tests: number of test samples to infer
#     - k: number of selected data points per test sample
#     - n_iterations: number of optimization steps
#     - lr: learning rate for optimizer
#     - device: 'cpu' or 'cuda'
#
#     Returns:
#     - x_tests_opt: torch.Tensor of shape (n_tests, n_features), optimized test samples
#     """
#     X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # (n_samples, n_features)
#     n_features = X.shape[1]
#
#     mean_selected = X_tensor[selected_indices_list].mean(dim=0, keepdim=True)
#     x_tests_opt = nn.Parameter(mean_selected.repeat(n_tests, 1) + torch.randn(n_tests, n_features, device=device) * 0.1)
#
#     optimizer = optim.Adam([x_tests_opt], lr=lr)
#
#     # Define weights for selected and unselected samples
#     weight_selected = 1.0
#     weight_unselected = 0.8  # Lower weight for unselected samples
#
#     w_trace = 1.0
#     w_alignment = 1.0
#     w_reg = 0.01
#     for it in range(n_iterations):
#         optimizer.zero_grad()
#         loss = 0.0
#         for i in range(n_tests):
#             x_test = x_tests_opt[i]  # (n_features,)
#
#             X_selected = X_tensor[selected_indices_list]  # (k, n_features)
#             X_unselected = X_tensor[unselected_indices_list]  # (n_samples - k, n_features)
#             # Assign higher weights to selected samples
#             weights_selected = torch.ones(X_selected.shape[0], device=device) * weight_selected
#             weights_unselected = torch.ones(X_unselected.shape[0], device=device) * weight_unselected
#
#             # Construct FIM for selected and unselected
#             fim_selected = construct_fim(X_selected.to(device), weights_selected.to(device))  # (n_features, n_features)
#             fim_unselected = construct_fim(X_unselected.to(device),
#                                            weights_unselected.to(device))  # (n_features, n_features)
#
#             # Total FIM
#             fim_total = fim_selected + fim_unselected  # (n_features, n_features)
#
#             # Objective:
#             # 1. Maximize the information from selected samples: maximize trace(fim_selected)
#             # 2. Minimize the information from unselected samples: minimize trace(fim_unselected)
#             # 3. Encourage the test sample to align with the FIM_selected and misalign with FIM_unselected
#
#             # Trace-based objectives
#             trace_selected = torch.trace(fim_selected)
#             trace_unselected = torch.trace(fim_unselected)
#
#             # Alignment-based objectives
#             alignment_selected = torch.matmul(x_test, torch.matmul(fim_selected, x_test))
#             alignment_unselected = torch.matmul(x_test, torch.matmul(fim_unselected, x_test))
#
#             # Update loss computation
#             loss += (-w_trace * trace_selected + w_trace * trace_unselected)
#             loss += (-w_alignment * alignment_selected + w_alignment * alignment_unselected)
#             loss += w_reg * torch.norm(x_test, p=2)
#
#             # Optionally, add regularization to keep x_test within reasonable bounds
#             reg_strength = 0.01
#             loss += reg_strength * torch.norm(x_test, p=2)
#
#         # Normalize loss by number of test samples
#         loss = loss / n_tests
#         loss.backward()
#         optimizer.step()
#
#         if (it + 1) % 500 == 0 or it == 0:
#             print(f"Iteration {it + 1}/{n_iterations}, Loss: {loss.item():.4f}")
#
#     return x_tests_opt.detach().cpu().numpy()


def fim_reverse_emb_opt_v2(x_s, selected_indices, unselected_indices, x_query, device, save_dir="./data", verbose=True):
    # Configuration
    n_tests = x_query.shape[0]
    n_iterations = 2000
    lr = 1e-2

    # Step 4: Reverse engineer test samples using optimization with FIM
    x_tests_est = optimize_test_samples_with_fim(
        x_s, selected_indices, unselected_indices, n_tests=n_tests,
        n_iterations=n_iterations, lr=lr, device=device
    )

    # Step 5: Evaluation
    best_cosine_similarities, best_euclidean_distances, matching_indices = evaluate_reconstruction(x_query, x_tests_est)
    for i in range(n_tests):
        print(f"\nTest Sample {i + 1}:")
        print(f" Cosine Similarity: {best_cosine_similarities[i]:.4f}")
        print(f" Euclidean Distance: {best_euclidean_distances[i]:.4f}")
        print(f" Matching index: {matching_indices[i]:.4f}")

    results = {
        "best_cosine_similarities": best_cosine_similarities,
        "best_euclidean_distances": best_euclidean_distances,
        "matching_indices": matching_indices
    }

    return results
