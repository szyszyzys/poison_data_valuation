import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
np.random.seed(42)


def compute_extended_fim(X_selected, X_unselected, epsilon=1e-3, lambda_reg=1e-5):
    """
    Constructs the extended Fisher Information Matrix (FIM) using selected and unselected data.

    Parameters:
    - X_selected: Selected data matrix of shape (n_selected, n_features)
    - X_unselected: Unselected data matrix of shape (n_unselected, n_features)
    - epsilon: Weight assigned to each unselected data point
    - lambda_reg: Regularization parameter

    Returns:
    - fim: Extended FIM matrix of shape (n_features, n_features)
    """
    # Compute FIM for selected data with selection weights (uniform in this example)
    W_selected = np.ones(X_selected.shape[0])  # Can be modified to have different weights
    I_selected = X_selected.T @ (W_selected[:, np.newaxis] * X_selected)

    # Compute FIM contribution from unselected data with small weights
    W_unselected = epsilon * np.ones(X_unselected.shape[0])
    I_unselected = X_unselected.T @ (W_unselected[:, np.newaxis] * X_unselected)

    # Regularize the FIM
    fim = I_selected + I_unselected + lambda_reg * np.eye(X_selected.shape[1])
    return fim


def eigen_decompose(fim_inv):
    """
    Performs eigenvalue decomposition on the inverse FIM.

    Parameters:
    - fim_inv: Inverse of the FIM matrix of shape (n_features, n_features)

    Returns:
    - eigenvalues: Sorted eigenvalues in descending order
    - eigenvectors: Corresponding eigenvectors sorted by eigenvalues
    """
    eigenvalues, eigenvectors = np.linalg.eigh(fim_inv)
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def infer_x_test_extended(X_selected, X_unselected, x_test, fim_inv, lambda_reg=1e-5):
    """
    Infers the test data as a linear combination of selected and unselected datapoints.

    Parameters:
    - X_selected: Selected data matrix of shape (n_selected, n_features)
    - X_unselected: Unselected data matrix of shape (n_unselected, n_features)
    - x_test: True test data vector of shape (n_features,)
    - fim_inv: Inverse of the extended FIM matrix
    - lambda_reg: Regularization parameter

    Returns:
    - x_test_hat: Reconstructed test data vector of shape (n_features,)
    - alpha_selected: Coefficients for selected datapoints
    - alpha_unselected: Coefficients for unselected datapoints
    """
    n_selected = X_selected.shape[0]
    n_unselected = X_unselected.shape[0]
    d = X_selected.shape[1]

    # Combine selected and unselected data
    X_combined = np.vstack((X_selected, X_unselected))  # Shape: (n_selected + n_unselected, d)

    # Ridge regression to find coefficients with regularization
    # Minimize ||x_test - X_combined.T * alpha||^2 + lambda_reg * ||alpha||^2
    ridge = Ridge(alpha=lambda_reg, fit_intercept=False)
    ridge.fit(X_combined.T, x_test)
    alpha = ridge.coef_  # Shape: (n_selected + n_unselected,)

    # Split coefficients
    alpha_selected = alpha[:n_selected]
    alpha_unselected = alpha[n_selected:]

    # Reconstruct x_test
    x_test_hat = X_combined.T @ alpha  # Shape: (d,)

    return x_test_hat, alpha_selected, alpha_unselected


def evaluate_inference(x_test, x_test_hat):
    """
    Evaluates the inference using Mean Squared Error and Cosine Similarity.

    Parameters:
    - x_test: True test data vector of shape (n_features,)
    - x_test_hat: Reconstructed test data vector of shape (n_features,)

    Returns:
    - mse: Mean Squared Error
    - cosine_sim: Cosine Similarity
    """
    mse = mean_squared_error(x_test, x_test_hat)
    cosine_sim = cosine_similarity([x_test], [x_test_hat])[0, 0]
    return mse, cosine_sim


def infer_statistical_properties(X_selected, X_unselected, x_test):
    """
    Infers statistical properties of the test data based on selected and unselected data.

    Parameters:
    - X_selected: Selected data matrix of shape (n_selected, n_features)
    - X_unselected: Unselected data matrix of shape (n_unselected, n_features)
    - x_test: True test data vector of shape (n_features,)

    Returns:
    - stats: Dictionary containing mean, variance, and mean difference
    """
    mean_selected = np.mean(X_selected, axis=0)
    variance_selected = np.var(X_selected, axis=0)

    mean_unselected = np.mean(X_unselected, axis=0)
    variance_unselected = np.var(X_unselected, axis=0)

    mean_diff_selected = x_test - mean_selected
    mean_diff_unselected = x_test - mean_unselected

    stats = {
        'mean_selected': mean_selected,
        'variance_selected': variance_selected,
        'mean_unselected': mean_unselected,
        'variance_unselected': variance_unselected,
        'mean_diff_selected': mean_diff_selected,
        'mean_diff_unselected': mean_diff_unselected
    }
    return stats


def additional_inference(X_selected, X_unselected, x_test, eigenvectors, top_k=2, n_clusters=3):
    """
    Performs additional inferences such as cluster membership and attribute importance.

    Parameters:
    - X_selected: Selected data matrix of shape (n_selected, n_features)
    - X_unselected: Unselected data matrix of shape (n_unselected, n_features)
    - x_test: True test data vector of shape (n_features,)
    - eigenvectors: Matrix of eigenvectors from eigen decomposition
    - top_k: Number of top principal components to consider
    - n_clusters: Number of clusters for KMeans

    Returns:
    - cluster_label: Cluster label assigned to x_test
    - attribute_importance: Importance scores for each attribute based on top_k eigenvectors
    """
    # Cluster Membership Inference
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_selected)
    cluster_label = kmeans.predict([x_test])[0]

    # Attribute Importance Inference using top_k eigenvectors
    projection = eigenvectors[:, :top_k].T @ x_test
    attribute_importance = np.abs(projection)
    # Normalize importance scores
    attribute_importance /= np.sum(attribute_importance)

    return cluster_label, attribute_importance


def plot_eigenvalues(eigenvalues, top_k=10):
    """
    Plots the top_k eigenvalues.

    Parameters:
    - eigenvalues: Array of eigenvalues
    - top_k: Number of top eigenvalues to plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, top_k + 1), eigenvalues[:top_k], marker='o')
    plt.title('Top Eigenvalues of Inverse Extended FIM')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()


def plot_reconstruction(x_test, x_test_hat):
    """
    Plots the true test data vs. reconstructed test data.

    Parameters:
    - x_test: True test data vector
    - x_test_hat: Reconstructed test data vector
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_test))
    plt.plot(indices, x_test, label='True Test Data', marker='o')
    plt.plot(indices, x_test_hat, label='Reconstructed Test Data', marker='x')
    plt.title('True vs. Reconstructed Test Data')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_experiment(
        x_s,
        x_query,
        selected_indices,
        unselected_indices,
        epsilon=1e-3,
        lambda_reg=1e-5,
        top_k=2,
        n_clusters=3,
        verbose=True
):
    """
    Runs the extended Test Data Inference Attack experiment.

    Parameters:
    - n_selected: Number of selected data points
    - n_unselected: Number of unselected data points
    - n_features: Number of features
    - epsilon: Weight for unselected data points in FIM
    - lambda_reg: Regularization parameter for FIM and regression
    - top_k: Number of top principal components for attribute importance
    - n_clusters: Number of clusters for KMeans
    - verbose: If True, prints detailed information

    Returns:
    - results: Dictionary containing all inference results
    """
    results = {}
    n_selected = 100,
    n_unselected = 200,
    n_features = 10,

    x_selected = x_s[selected_indices]
    x_unselected = x_s[unselected_indices]
    # Step 1: Construct Extended FIM
    fim = compute_extended_fim(x_selected, x_unselected, epsilon=epsilon, lambda_reg=lambda_reg)
    results['fim'] = fim

    # Step 2: Compute Inverse of FIM
    try:
        fim_inv = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding more regularization
        fim += 1e-3 * np.eye(fim.shape[0])
        fim_inv = np.linalg.inv(fim)
    results['fim_inv'] = fim_inv

    # Step 3: Eigenvalue Decomposition
    eigenvalues, eigenvectors = eigen_decompose(fim_inv)
    results['eigenvalues'] = eigenvalues
    results['eigenvectors'] = eigenvectors
    if verbose:
        print(f"Top {top_k} Eigenvalues: {eigenvalues[:top_k]}")

    # Optional: Plot eigenvalues
    plot_eigenvalues(eigenvalues, top_k=top_k)

    # Step 4: Infer x_test
    x_test_hat, alpha_selected, alpha_unselected = infer_x_test_extended(
        x_selected, x_unselected, x_query, fim_inv, lambda_reg=lambda_reg
    )
    results['x_test_hat'] = x_test_hat
    results['alpha_selected'] = alpha_selected
    results['alpha_unselected'] = alpha_unselected
    if verbose:
        print(f"Reconstructed x_test: {x_test_hat}")

    # Step 5: Evaluate Inference
    mse, cosine_sim = evaluate_inference(x_query, x_test_hat)
    results['mse'] = mse
    results['cosine_similarity'] = cosine_sim
    if verbose:
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")

    # Optional: Plot reconstruction
    plot_reconstruction(x_query, x_test_hat)

    # Step 6: Infer Statistical Properties
    stats = infer_statistical_properties(x_selected, x_unselected, x_query)
    results['stats'] = stats
    if verbose:
        print("\n--- Statistical Properties ---")
        for key, value in stats.items():
            print(f"{key}: {value}")

    # Step 7: Additional Inference
    cluster_label, attribute_importance = additional_inference(
        x_selected, x_unselected, x_query, eigenvectors, top_k=top_k, n_clusters=n_clusters
    )
    results['cluster_label'] = cluster_label
    results['attribute_importance'] = attribute_importance
    if verbose:
        print(f"\nCluster Label Assigned to x_test: {cluster_label}")
        print(f"Attribute Importance Scores: {attribute_importance}")

    return results


def fim_reverse_math():
    """
    Main function to execute the extended Test Data Inference Attack.
    """
    # Define tunable parameters
    params = {
        'n_selected': 100,
        'n_unselected': 200,
        'n_features': 10,
        'epsilon': 1e-3,  # Weight for unselected data points
        'lambda_reg': 1e-5,  # Regularization parameter
        'top_k': 2,  # Number of top principal components
        'n_clusters': 3,  # Number of clusters for KMeans
        'verbose': True  # Enable verbose output
    }

    # Run experiment
    results = run_experiment(**params)

    # Summary of Results
    print("\n--- Inference Summary ---")
    print(f"Mean Squared Error (MSE): {results['mse']:.6f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.6f}")
    print(f"Cluster Label Assigned to x_test: {results['cluster_label']}")
    print(f"Attribute Importance Scores: {results['attribute_importance']}")

    # Additional Insights
    # For example, examining the reconstruction coefficients
    print("\n--- Reconstruction Coefficients (Selected) ---")
    print(results['alpha_selected'])
    print("\n--- Reconstruction Coefficients (Unselected) ---")
    print(results['alpha_unselected'])
