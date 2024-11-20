import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def compute_fim(X, indices, weights=None):
    """
    Compute the Fisher Information Matrix (FIM) for given indices.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - indices (np.ndarray): Indices of samples to include in FIM computation.
    - weights (np.ndarray or None): Weights for each selected sample. If None, equal weights are used.

    Returns:
    - fim (np.ndarray): Computed FIM of shape (n_features, n_features).
    """
    X_subset = X[indices]
    if weights is None:
        weights = np.ones(len(indices))
    # Reshape weights for broadcasting
    weights = weights[:, np.newaxis, np.newaxis]
    # Compute outer products and weight them
    outer_products = X_subset[:, :, np.newaxis] * X_subset[:, np.newaxis, :]
    weighted_outer = weights * outer_products
    # Sum over all samples to get the FIM
    fim = weighted_outer.sum(axis=0)
    return fim


def optimize_query_embedding(I_selected, I_unselected, dim, lambda_reg=0.1, lr=0.01, iterations=100, verbose=False):
    """
    Optimize the query embedding to align with selected data and penalize unselected data.

    Parameters:
    - I_selected (np.ndarray): FIM for selected samples.
    - I_unselected (np.ndarray): FIM for unselected samples.
    - dim (int): Dimension of the embedding space.
    - lambda_reg (float): Regularization parameter.
    - lr (float): Learning rate for gradient ascent.
    - iterations (int): Number of optimization iterations.
    - verbose (bool): If True, prints progress.

    Returns:
    - q (np.ndarray): Optimized query embedding of shape (dim,).
    """
    # Initialize embedding randomly
    q = np.random.randn(dim)
    q /= np.linalg.norm(q)  # Normalize for stability

    for i in range(iterations):
        # Compute gradient: 2 * I_selected @ q - 2 * lambda_reg * I_unselected @ q
        grad = 2 * I_selected.dot(q) - 2 * lambda_reg * I_unselected.dot(q)

        # Update embedding using gradient ascent
        q += lr * grad

        # Normalize to prevent numerical issues
        norm = np.linalg.norm(q)
        if norm == 0:
            norm = 1e-8
        q /= norm

        if verbose and (i + 1) % 10 == 0:
            loss = q.dot(I_selected).dot(q) - lambda_reg * q.dot(I_unselected).dot(q)
            print(f"Iteration {i + 1}/{iterations}, Loss: {loss:.4f}")

    return q


def score_samples(X, q):
    """
    Compute scores for all samples based on alignment with the embedding q.

    The score is defined as the squared cosine similarity between q and each sample.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - q (np.ndarray): Query embedding of shape (n_features,).

    Returns:
    - scores (np.ndarray): Scores for each sample of shape (n_samples,).
    """
    # Compute dot product between each sample and q
    dot_products = X.dot(q)
    # Since q is normalized, dot_products are the cosine similarities
    # Square them to emphasize higher similarities
    scores = dot_products ** 2
    return scores


def visualize_ranking(scores, selected_indices, unselected_indices, top_n=50):
    """
    Visualize the ranking of selected and unselected samples based on their scores.

    Parameters:
    - scores (np.ndarray): Scores for each sample.
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - top_n (int): Number of top-ranked samples to visualize.
    """
    # Sort indices by descending scores
    ranked_indices = np.argsort(-scores)
    top_indices = ranked_indices[:top_n]

    # Create labels: 1 for selected, 0 for unselected
    labels = np.isin(top_indices, selected_indices).astype(int)

    plt.figure(figsize=(12, 6))
    plt.scatter(range(top_n), labels, c=labels, cmap='bwr', edgecolor='k')
    plt.title(f"Top {top_n} Ranked Samples: Selected (1) vs Unselected (0)")
    plt.xlabel("Rank Position")
    plt.ylabel("Selected (1) / Unselected (0)")
    plt.yticks([0, 1])
    plt.grid(True, axis='y')
    plt.show()


def display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=10):
    """
    Display the top_k selected and unselected samples from the ranked list.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - ranked_indices (np.ndarray): Indices sorted by rank.
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - top_k (int): Number of top samples to display for each category.
    """
    top_selected = [idx for idx in ranked_indices if idx in selected_indices][:top_k]
    top_unselected = [idx for idx in ranked_indices if idx in unselected_indices][:top_k]

    print(f"Top {top_k} Selected Samples in Ranking:", top_selected)
    print(f"Top {top_k} Unselected Samples in Ranking:", top_unselected)


def fim_reverse_opt(X, selected_indices, unselected_indices, dim, lambda_reg=0.1, lr=0.01, iterations=100, top_n=50,
                    top_k=10, verbose=False):
    """
    Perform the full embedding reconstruction and visualization process.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - dim (int): Dimension of the embedding space.
    - lambda_reg (float): Regularization parameter.
    - lr (float): Learning rate for optimization.
    - iterations (int): Number of optimization iterations.
    - top_n (int): Number of top-ranked samples to visualize.
    - top_k (int): Number of top samples to display for each category.
    - verbose (bool): If True, prints detailed logs.

    Returns:
    - q (np.ndarray): Reconstructed query embedding.
    - scores (np.ndarray): Scores for all samples.
    """
    # Compute FIMs
    I_selected = compute_fim(X, selected_indices)
    I_unselected = compute_fim(X, unselected_indices)

    if verbose:
        print("FIMs computed.")

    # Optimize query embedding
    q = optimize_query_embedding(
        I_selected,
        I_unselected,
        dim=dim,
        lambda_reg=lambda_reg,
        lr=lr,
        iterations=iterations,
        verbose=verbose
    )

    if verbose:
        print("Optimization completed.")

    # Score all samples
    scores = score_samples(X, q)

    if verbose:
        print("Scoring completed.")

    # Visualize rankings
    visualize_ranking(scores, selected_indices, unselected_indices, top_n=top_n)

    # Sort indices by descending scores
    ranked_indices = np.argsort(-scores)

    # Display top_k selected and unselected samples
    display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=top_k)

    return q, scores


def fim_reverse_emb_opt_normal(x_s, selected_indices, unselected_indices, verbose=True):
    """
    Main function to execute the refined embedding reconstruction and visualization.
    """
    # Parameters
    dim = x_s.shape[-1]
    lambda_reg = 0.1
    margin = 1.0
    iterations = 100
    top_n = 50
    top_k = 10

    if verbose:
        print(f"Generated data with {n_samples} samples and {n_features} features.")
        print(f"Selected samples: {len(selected_indices)}, Unselected samples: {len(unselected_indices)}")

    # 2. Reconstruct Embedding and Visualize
    q, scores = fim_reverse_opt(
        x_s,
        selected_indices,
        unselected_indices,
        dim=dim,
        lambda_reg=lambda_reg,
        lr=lr,
        iterations=iterations,
        top_n=top_n,
        top_k=top_k,
        verbose=verbose
    )

    # Additional Evaluation Metrics (Optional)
    # For example, visualize the distribution of scores
    plt.figure(figsize=(10, 6))
    plt.hist(scores[selected_indices], bins=30, alpha=0.5, label='Selected', color='green')
    plt.hist(scores[unselected_indices], bins=30, alpha=0.5, label='Unselected', color='red')
    plt.title("Score Distribution: Selected vs Unselected Samples")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Save the reconstructed embedding and scores if needed
    # np.save("reconstructed_embedding.npy", q)
    # np.save("scores.npy", scores)
