import kmeans1d
import numpy as np
from sklearn.datasets import make_blobs


def kmeans(x, k):
    clusters, centroids = kmeans1d.cluster(x, k)
    return clusters, centroids


def optimal_k_gap(X, k_max=10, B=10, random_state=42):
    """
    Compute the optimal number of clusters for data X using the Gap Statistic.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data to be clustered.
    k_max : int, default=10
        Maximum number of clusters to evaluate (i.e., k = 1, 2, ..., k_max).
    B : int, default=10
        Number of reference datasets (bootstraps) to generate.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    optimal_k : int
        The estimated optimal number of clusters. (May be 1 if that is optimal.)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    ks = np.arange(1, k_max + 1)

    def compute_dispersion(data, k):
        """
        Compute within-cluster dispersion for data given k clusters.
        """
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        dispersion = 0.0
        for j in range(k):
            # Get data points in cluster j
            cluster_data = data[kmeans.labels_ == j]
            if cluster_data.shape[0] > 0:
                center = np.mean(cluster_data, axis=0)
                dispersion += np.sum((cluster_data - center) ** 2)
        return dispersion

    # Compute dispersion for the actual data for each k
    Wks = np.array([compute_dispersion(X, k) for k in ks])

    # Generate B reference datasets and compute their dispersions for each k
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    Wkbs = np.zeros((len(ks), B))
    for b in range(B):
        # Generate a reference dataset uniformly at random within the bounding box of X.
        X_ref = np.random.uniform(low=mins, high=maxs, size=(n_samples, n_features))
        for i, k in enumerate(ks):
            Wkbs[i, b] = compute_dispersion(X_ref, k)

    # Compute the gap statistic for each k
    logWks = np.log(Wks)
    logWkbs = np.log(Wkbs)
    gap = np.mean(logWkbs, axis=1) - logWks
    sdk = np.std(logWkbs, axis=1) * np.sqrt(1 + 1.0 / B)

    # Select the smallest k such that:
    #   gap(k) >= gap(k+1) - sdk(k+1)
    optimal_k = ks[-1]  # default to the largest k in case none meet the condition
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sdk[i + 1]:
            optimal_k = ks[i]
            break

    return optimal_k

# def gap(x):
#     optimalK = OptimalK()
#     n_clusters = optimalK(x, cluster_array=np.arange(1, 5))
#     return n_clusters


# def kmeans_1d(x, k):
#     """
#     Perform k-means clustering on 1D data using scikit-learn's KMeans.
#
#     Args:
#         x (array-like): 1D array or a 2D array with one feature.
#         k (int): Number of clusters.
#
#     Returns:
#         clusters (list): Cluster labels for each data point.
#         centroids (list): Cluster centroids.
#     """
#     x = np.asarray(x)
#     # Ensure x is 2D (n_samples, 1)
#     if x.ndim == 1:
#         x = x.reshape(-1, 1)
#
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(x)
#     clusters = kmeans.labels_.tolist()
#     centroids = kmeans.cluster_centers_.flatten().tolist()
#     return clusters, centroids


# def optimal_k(X, k_range=range(2, 10)):
#     best_k = None
#     best_score = -1
#     # Try different numbers of clusters in the provided range.
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
#         # Check how many clusters are actually found.
#         unique_labels = np.unique(kmeans.labels_)
#         if len(unique_labels) < 2:
#             # If there's only one cluster, silhouette_score can't be computed.
#             score = -1
#         else:
#             score = silhouette_score(X, kmeans.labels_)
#         # Update best_k if a better score is found.
#         if score > best_score:
#             best_score = score
#             best_k = k
#     return best_k


# def optimal_k(x, k_range=range(1, 5)):
#     """
#     Determine the optimal number of clusters based on silhouette analysis.
#
#     Args:
#         x (array-like): 1D array or a 2D array with one feature.
#         k_range (iterable): Range of cluster counts to try.
#
#     Returns:
#         best_k (int): The number of clusters that maximizes the silhouette score.
#     """
#     x = np.asarray(x)
#     if x.ndim == 1:
#         x = x.reshape(-1, 1)
#
#     # If the variance is nearly zero, we have essentially one cluster.
#     if np.var(x) < 1e-6:
#         return 1
#
#     best_k = None
#     best_score = -1
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42).fit(x)
#         # Silhouette score is only defined for k >= 2.
#         score = silhouette_score(x, kmeans.labels_)
#         if score > best_score:
#             best_score = score
#             best_k = k
#     return best_k

if __name__ == '__main__':
    # Generate synthetic 1D data with 3 centers
    x, y = make_blobs(n_samples=int(1e3), n_features=1, centers=3, random_state=25)
    print('Data shape:', x.shape)

    # Determine the optimal number of clusters using silhouette analysis.
    n_clusters = optimal_k(x, k_range=range(2, 5))
    print('Optimal clusters:', n_clusters)

    # Perform clustering using the determined number of clusters.
    clusters, centroids = kmeans_1d(x, n_clusters)
    print("Cluster labels (first 10):", clusters[:10])
    print("Centroids:", centroids)
