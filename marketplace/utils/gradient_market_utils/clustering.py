import kmeans1d
import numpy as np
from gap_statistic import OptimalK
from sklearn.datasets import make_blobs


def kmeans(x, k):
    clusters, centroids = kmeans1d.cluster(x, k)
    return clusters, centroids


def gap(x, cluster_array=np.arange(1, 5)):
    """
    Determine the optimal number of clusters using the gap statistic.

    Parameters:
      x : numpy array, shape (n_samples, n_features)
         The data to be clustered.
      cluster_array : array-like, optional
         The candidate numbers of clusters to try (default is np.arange(1, 11)).

    Returns:
      n_clusters : int
         The optimal number of clusters, even if it is 1.
    """
    if cluster_array is None:
        cluster_array = np.arange(1, 11)  # Try cluster counts from 1 to 10 by default.

    optimalK = OptimalK(n_jobs=1)
    n_clusters = optimalK(x, cluster_array=cluster_array)
    return n_clusters


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
