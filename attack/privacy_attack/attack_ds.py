from scipy.linalg import inv


def reconstruction_loss(X_buy_hat, S_obs, X_sell, k):
    """
    Compute the reconstruction loss.

    Args:
        X_buy_hat (np.array): Reconstructed buyer data (shape: (m, d)).
        S_obs (np.array): Observed selected seller data (shape: (k, d)).
        X_sell (np.array): Seller data (shape: (n, d)).
        k (int): Number of selected points.

    Returns:
        loss (float): Reconstruction loss.
        grad_X_buy_hat (np.array): Gradient of the loss w.r.t. X_buy_hat (shape: (m, d)).
    """
    n, d = X_sell.shape
    m = X_buy_hat.shape[0]

    # Compute inverse covariance matrix
    weights = np.ones(n) / n  # Uniform weights (can be adjusted)
    inv_cov = inv(X_sell.T @ np.diag(weights) @ X_sell)

    # Compute scores for seller data points
    scores = np.linalg.norm(inv_cov @ X_sell.T, axis=0)

    # Compute ranks
    ranks = np.argsort(-scores)  # Descending order

    # Compute loss
    loss = 0
    grad_X_buy_hat = np.zeros_like(X_buy_hat)
    for i, x in enumerate(S_obs):
        idx = np.where((X_sell == x).all(axis=1))[0][0]  # Find index of x in X_sell
        loss += (ranks[idx] - 1) ** 2
        grad_X_buy_hat += 2 * (ranks[idx] - 1) * (inv_cov @ X_sell[idx])
    for x in X_sell[np.isin(X_sell, S_obs, invert=True)]:
        idx = np.where((X_sell == x).all(axis=1))[0][0]
        loss += max(0, k - ranks[idx]) ** 2
        grad_X_buy_hat += 2 * max(0, k - ranks[idx]) * (inv_cov @ X_sell[idx])

    return loss, grad_X_buy_hat


def reconstruct_X_buy(S_obs, X_sell, k, X_buy_hat_init, alpha=0.01, max_iter=100, tol=1e-6):
    """
    Reconstruct the buyer's data X_buy.

    Args:
        S_obs (np.array): Observed selected seller data (shape: (k, d)).
        X_sell (np.array): Seller data (shape: (n, d)).
        k (int): Number of selected points.
        X_buy_hat_init (np.array): Initial guess for X_buy (shape: (m, d)).
        alpha (float): Learning rate.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        X_buy_hat (np.array): Reconstructed buyer data.
    """
    X_buy_hat = X_buy_hat_init.copy()
    for t in range(max_iter):
        loss, grad_X_buy_hat = reconstruction_loss(X_buy_hat, S_obs, X_sell, k)
        if t > 0 and np.linalg.norm(grad_X_buy_hat) < tol:
            break
        X_buy_hat -= alpha * grad_X_buy_hat
    return X_buy_hat


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
