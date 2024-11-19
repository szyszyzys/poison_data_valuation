import matplotlib.pyplot as plt
import numpy as np


# 1. Generate Synthetic Data
def generate_data(n_samples=100, n_features=10, selected_ratio=0.3):
    """
    Generate synthetic data with selected and unselected samples.
    """
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, n_features))  # Feature matrix
    selected_indices = np.random.choice(n_samples, int(selected_ratio * n_samples), replace=False)
    unselected_indices = list(set(range(n_samples)) - set(selected_indices))
    return X, selected_indices, unselected_indices


# 2. Compute Fisher Information Matrix (FIM)
def compute_fim(X, indices, weights=None):
    """
    Compute the Fisher Information Matrix for selected indices.
    """
    if weights is None:
        weights = np.ones(len(indices))
    X_subset = X[indices]
    return (weights[:, None, None] * (X_subset[:, :, None] @ X_subset[:, None, :])).sum(axis=0)


# 3. Optimize Query Embedding
def optimize_query_embedding(I_selected, I_unselected, dim, lambda_reg=0.1, lr=0.01, iterations=100):
    """
    Optimize the query embedding to align with selected data and penalize unselected data.
    """
    q = np.random.randn(dim)  # Initialize embedding randomly
    for _ in range(iterations):
        grad = 2 * I_selected @ q - 2 * lambda_reg * I_unselected @ q  # Gradient
        q += lr * grad  # Update
        q /= np.linalg.norm(q)  # Normalize for stability
    return q


# 4. Score Samples
def score_samples(X, q):
    """
    Compute scores for all samples based on alignment with the embedding q.
    """
    return np.array([q.T @ (x[:, None] @ x[None, :]) @ q for x in X])


# 5. Visualization
def visualize_ranking(scores, selected_indices, unselected_indices):
    """
    Visualize the ranking of selected and unselected samples.
    """
    ranked_indices = np.argsort(-scores)  # Higher scores first
    rankings = [1 if idx in selected_indices else 0 for idx in ranked_indices]

    plt.figure(figsize=(8, 6))
    plt.plot(rankings, marker='o', linestyle='-', color='blue')
    plt.title("Ranking of Selected (1) vs Unselected (0) Samples")
    plt.xlabel("Ranked Position")
    plt.ylabel("Selected (1) / Unselected (0)")
    plt.show()


# Full Workflow
if __name__ == "__main__":
    # Generate synthetic data
    X, selected_indices, unselected_indices = generate_data(n_samples=100, n_features=10, selected_ratio=0.3)

    # Compute FIMs
    I_selected = compute_fim(X, selected_indices)
    I_unselected = compute_fim(X, unselected_indices)

    # Optimize query embedding
    q = optimize_query_embedding(I_selected, I_unselected, dim=X.shape[1], lambda_reg=0.1)

    # Score all samples
    scores = score_samples(X, q)

    # Visualize rankings
    visualize_ranking(scores, selected_indices, unselected_indices)

    # Print rankings
    ranked_indices = np.argsort(-scores)
    selected_in_ranking = [idx for idx in ranked_indices if idx in selected_indices]
    unselected_in_ranking = [idx for idx in ranked_indices if idx in unselected_indices]

    print("Top 10 Selected Samples in Ranking:", selected_in_ranking[:10])
    print("Top 10 Unselected Samples in Ranking:", unselected_in_ranking[:10])
