import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from attack.general_attack.my_utils import evaluate_reconstruction


def generate_synthetic_data(n_samples=200, n_features=20, random_state=42):
    """
    Generates synthetic data for demonstration.
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    return X


def create_test_samples(X, n_tests=3, n_components=5, random_state=42):
    """
    Creates multiple test samples as linear combinations of selected data points.
    """
    np.random.seed(random_state)
    test_samples = []
    true_selected_indices = []
    true_coefficients = []
    for _ in range(n_tests):
        selected_indices = np.random.choice(X.shape[0], n_components, replace=False)
        coefficients = np.random.randn(n_components)
        x_test = X[selected_indices].T @ coefficients
        test_samples.append(x_test)
        true_selected_indices.append(selected_indices)
        true_coefficients.append(coefficients)
    return np.stack(test_samples), true_selected_indices, true_coefficients


def selection_mechanism(X, x_test, k=10):
    """
    Simulates the selection mechanism based on alignment with x_test.
    Selects top-k data points that are most aligned with x_test.
    """
    scores = X @ x_test
    selected_indices = np.argsort(scores)[-k:]
    return selected_indices


def simulate_selection(X, x_tests, k=10):
    """
    Simulates selection for multiple test samples.
    Returns a list of selected indices for each test sample.
    """
    selected_indices_list = []
    for x_test in x_tests:
        selected_indices = selection_mechanism(X, x_test, k)
        selected_indices_list.append(selected_indices)
    return selected_indices_list


def construct_fim(X_selected, weights):
    """
    Constructs the Fisher Information Matrix (FIM) from selected samples and their weights.

    I(w) = sum_j w_j x_j x_j^T
    """
    fim = torch.zeros(X_selected.shape[1], X_selected.shape[1], device=X_selected.device)
    for j in range(X_selected.shape[0]):
        x_j = X_selected[j].unsqueeze(1)  # (n_features, 1)
        fim += weights[j] * (x_j @ x_j.T)  # (n_features, n_features)
    return fim


def optimize_test_samples_with_fim(X, selected_indices_list, unselected_indices_list, n_tests=3,
                                   n_iterations=5000, lr=1e-2, device='cpu'):
    """
    Infers test samples using an optimization-based method incorporating the Fisher Information Matrix (FIM).

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features)
    - selected_indices_list: list of np.ndarray, each containing selected indices for a test sample
    - n_tests: number of test samples to infer
    - k: number of selected data points per test sample
    - n_iterations: number of optimization steps
    - lr: learning rate for optimizer
    - device: 'cpu' or 'cuda'

    Returns:
    - x_tests_opt: torch.Tensor of shape (n_tests, n_features), optimized test samples
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # (n_samples, n_features)
    n_features = X.shape[1]
    n_selected_samples = len(selected_indices_list)
    n_unselected_samples = len(selected_indices_list)

    # Initialize test samples as parameters to optimize
    x_tests_opt = nn.Parameter(torch.randn(n_tests, n_features, device=device) * 0.1)

    optimizer = optim.Adam([x_tests_opt], lr=lr)

    # Define weights for selected and unselected samples
    weight_selected = 1.0
    weight_unselected = 0.1  # Lower weight for unselected samples

    for it in range(n_iterations):
        optimizer.zero_grad()
        loss = 0.0
        for i in range(n_tests):
            x_test = x_tests_opt[i]  # (n_features,)
            selected = selected_indices_list[i]

            X_selected = X_tensor[selected]  # (k, n_features)
            X_unselected = X_tensor[unselected_indices_list]  # (n_samples - k, n_features)

            # Assign higher weights to selected samples
            weights_selected = torch.ones(n_selected_samples, device=device) * weight_selected
            weights_unselected = torch.ones(n_unselected_samples, device=device) * weight_unselected

            # Construct FIM for selected and unselected
            fim_selected = construct_fim(X_selected, weights_selected)  # (n_features, n_features)
            fim_unselected = construct_fim(X_unselected, weights_unselected)  # (n_features, n_features)

            # Total FIM
            fim_total = fim_selected + fim_unselected  # (n_features, n_features)

            # Objective:
            # 1. Maximize the information from selected samples: maximize trace(fim_selected)
            # 2. Minimize the information from unselected samples: minimize trace(fim_unselected)
            # 3. Encourage the test sample to align with the FIM_selected and misalign with FIM_unselected

            # Trace-based objectives
            trace_selected = torch.trace(fim_selected)
            trace_unselected = torch.trace(fim_unselected)
            loss += -trace_selected + trace_unselected  # Want to maximize trace_selected and minimize trace_unselected

            # Alignment-based objectives
            alignment_selected = torch.matmul(x_test, torch.matmul(fim_selected, x_test))
            alignment_unselected = torch.matmul(x_test, torch.matmul(fim_unselected, x_test))
            loss += -alignment_selected + alignment_unselected  # Encourage alignment with selected, discourage with unselected

            # Optionally, add regularization to keep x_test within reasonable bounds
            reg_strength = 0.01
            loss += reg_strength * torch.norm(x_test, p=2)

        # Normalize loss by number of test samples
        loss = loss / n_tests
        loss.backward()
        optimizer.step()

        if (it + 1) % 500 == 0 or it == 0:
            print(f"Iteration {it + 1}/{n_iterations}, Loss: {loss.item():.4f}")

    return x_tests_opt.detach().cpu().numpy()


def fim_reverse_emb_opt_v2(x_s, selected_indices, unselected_indices, x_query, device, save_dir="./data", verbose=True):
    # Configuration
    n_tests = x_query.shape[0]
    n_iterations = 5000
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
