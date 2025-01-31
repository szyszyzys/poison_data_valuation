from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors


def reconstruction_loss(X_buy_hat, S_obs, X_sell, k):
    """
    Compute loss with distribution alignment and proper masking.
    """
    # --- Normalization: Match Seller Distribution ---
    mu_sell = np.mean(X_sell, axis=0)
    cov_sell = np.cov(X_sell, rowvar=False)
    inv_cov_sell = np.linalg.pinv(cov_sell)

    # Whitening transformation
    X_buy_hat_whitened = (X_buy_hat - mu_sell) @ sqrtm(inv_cov_sell)

    # --- Differentiable Selection Loss ---
    n = X_sell.shape[0]
    scores = np.linalg.norm(X_buy_hat_whitened @ X_sell.T, axis=0)  # Shape: (n,)

    # Find nearest neighbors of S_obs in X_sell
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
    _, selected_indices = nbrs.kneighbors(S_obs)  # Shape: (k, 1)
    selected_indices = selected_indices.flatten()  # Shape: (k,)

    # Initialize mask correctly as 1D array
    selected_mask = np.zeros(n, dtype=int)
    selected_mask[selected_indices] = 1  # Now works

    # Softmax-based differentiable loss
    temperature = 0.1
    soft_scores = scores / temperature
    soft_probs = np.exp(soft_scores) / np.sum(np.exp(soft_scores))

    # Cross-entropy loss between soft_probs and selected_mask
    loss = -np.sum(selected_mask * np.log(soft_probs + 1e-8))

    # Gradient calculation
    grad_soft = (soft_probs - selected_mask) / temperature
    grad = grad_soft @ X_sell @ sqrtm(inv_cov_sell).T

    return loss, grad


def reconstruct_X_buy(S_obs, X_sell, k, X_buy_hat_init, alpha=0.01, max_iter=100, tol=1e-6):
    """
    Robust reconstruction with shape checks and normalization.
    """
    # Ensure X_buy_hat_init is 2D (even for single-sample reconstruction)
    if X_buy_hat_init.ndim == 1:
        X_buy_hat = X_buy_hat_init[np.newaxis, :]
    else:
        X_buy_hat = X_buy_hat_init.copy()

    prev_loss = np.inf
    for t in range(max_iter):
        loss, grad = reconstruction_loss(X_buy_hat, S_obs, X_sell, k)

        if np.abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

        # Update with adaptive learning rate
        X_buy_hat -= alpha * grad

        # Project to unit sphere (optional)
        X_buy_hat = X_buy_hat / np.linalg.norm(X_buy_hat, axis=1, keepdims=True)

    return X_buy_hat.squeeze()  # Return 1D if input was 1D


# def reconstruction_loss(X_buy_hat, S_obs, X_sell, k):
#     """
#     Compute reconstruction loss with robustness to numerical precision issues.
#
#     Args:
#         X_buy_hat (np.array): Reconstructed buyer data (shape: (m, d)).
#         S_obs (np.array): Observed selected seller data (shape: (k, d)).
#         X_sell (np.array): Seller data (shape: (n, d)).
#         k (int): Number of selected points.
#
#     Returns:
#         loss (float): Reconstruction loss.
#         grad_X_buy_hat (np.array): Gradient of the loss w.r.t. X_buy_hat (shape: (m, d)).
#     """
#     # --- Find indices of S_obs in X_sell using approximate matching ---
#     nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
#     _, indices = nbrs.kneighbors(S_obs)
#     selected_indices = indices.flatten()  # Shape: (k,)
#
#     # --- Compute scores and ranks ---
#     weights = np.ones(len(X_sell)) / len(X_sell)
#     inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)  # Pseudo-inverse for stability
#     scores = np.linalg.norm(inv_cov @ X_sell.T, axis=0)  # Shape: (n,)
#     ranks = np.argsort(-scores)  # Descending order
#
#     # --- Loss Calculation ---
#     # Penalize if selected points are not in top-k ranks
#     loss = 0
#     grad_X_buy_hat = np.zeros_like(X_buy_hat)
#
#     # For selected points: Penalize if rank > k
#     for idx in selected_indices:
#         if ranks[idx] >= k:
#             loss += (ranks[idx] - k + 1) ** 2
#             grad_X_buy_hat += 2 * (ranks[idx] - k + 1) * (inv_cov @ X_sell[idx])
#
#     # For unselected points: Penalize if rank < k
#     unselected_mask = np.ones(len(X_sell), dtype=bool)
#     unselected_mask[selected_indices] = False
#     for idx in np.where(unselected_mask)[0]:
#         if ranks[idx] < k:
#             loss += (k - ranks[idx]) ** 2
#             grad_X_buy_hat += 2 * (k - ranks[idx]) * (inv_cov @ X_sell[idx])
#
#     return loss, grad_X_buy_hat
#
#
# def reconstruct_X_buy(S_obs, X_sell, k, X_buy_hat_init, alpha=0.01, max_iter=100, tol=1e-6):
#     """
#     Reconstruct the buyer's data X_buy.
#
#     Args:
#         S_obs (np.array): Observed selected seller data (shape: (k, d)).
#         X_sell (np.array): Seller data (shape: (n, d)).
#         k (int): Number of selected points.
#         X_buy_hat_init (np.array): Initial guess for X_buy (shape: (m, d)).
#         alpha (float): Learning rate.
#         max_iter (int): Maximum number of iterations.
#         tol (float): Convergence tolerance.
#
#     Returns:
#         X_buy_hat (np.array): Reconstructed buyer data.
#     """
#     X_buy_hat = X_buy_hat_init.copy()
#     for t in range(max_iter):
#         loss, grad_X_buy_hat = reconstruction_loss(X_buy_hat, S_obs, X_sell, k)
#         if t > 0 and np.linalg.norm(grad_X_buy_hat) < tol:
#             break
#         X_buy_hat -= alpha * grad_X_buy_hat
#     return X_buy_hat


import numpy as np
from scipy.stats import multivariate_normal


def likelihood(q, S_obs, X_sell, alpha=1.0):
    """
    Compute the likelihood of the observed selection S_obs given the query q.

    Args:
        q (np.array): Query (shape: (d,)).
        S_obs (np.array): Observed selected data points (shape: (k, d)).
        X_sell (np.array): Seller's data (shape: (n, d)).
        alpha (float): Temperature parameter.

    Returns:
        log_likelihood (float): Log-likelihood of S_obs given q.
    """
    # Compute gradient influence (simplified for illustration)
    residuals = X_sell @ q  # Simplified residuals (replace with actual model predictions if available)
    scores = np.linalg.norm(X_sell * residuals[:, np.newaxis], axis=1)

    # Compute likelihood
    log_likelihood = alpha * np.sum(scores[np.isin(X_sell, S_obs).all(axis=1)])
    return log_likelihood


def prior(q, mean, cov):
    """
    Compute the log-prior probability of q.

    Args:
        q (np.array): Query (shape: (d,)).
        mean (np.array): Mean of the prior distribution (shape: (d,)).
        cov (np.array): Covariance of the prior distribution (shape: (d, d)).

    Returns:
        log_prior (float): Log-prior probability of q.
    """
    return multivariate_normal.logpdf(q, mean=mean, cov=cov)


def metropolis_hastings(S_obs, X_sell, prior_mean, prior_cov, num_samples=1000, step_size=0.1, alpha=1.0):
    """
    Perform Metropolis-Hastings sampling to infer the posterior distribution of q.

    Args:
        S_obs (np.array): Observed selected data points (shape: (k, d)).
        X_sell (np.array): Seller's data (shape: (n, d)).
        prior_mean (np.array): Mean of the prior distribution (shape: (d,)).
        prior_cov (np.array): Covariance of the prior distribution (shape: (d, d)).
        num_samples (int): Number of MCMC samples.
        step_size (float): Step size for proposing new candidates.
        alpha (float): Temperature parameter for the likelihood.

    Returns:
        samples (np.array): Samples from the posterior distribution (shape: (num_samples, d)).
    """
    d = X_sell.shape[1]
    samples = np.zeros((num_samples, d))
    q_current = np.random.randn(d)  # Initialize q randomly

    for t in range(num_samples):
        # Propose a new candidate
        q_proposed = q_current + step_size * np.random.randn(d)

        # Compute acceptance probability
        log_likelihood_current = likelihood(q_current, S_obs, X_sell, alpha)
        log_likelihood_proposed = likelihood(q_proposed, S_obs, X_sell, alpha)
        log_prior_current = prior(q_current, prior_mean, prior_cov)
        log_prior_proposed = prior(q_proposed, prior_mean, prior_cov)

        log_acceptance = (log_likelihood_proposed + log_prior_proposed) - (log_likelihood_current + log_prior_current)
        acceptance_prob = min(1, np.exp(log_acceptance))

        # Accept or reject the candidate
        if np.random.rand() < acceptance_prob:
            q_current = q_proposed

        # Store the sample
        samples[t] = q_current

    return samples
