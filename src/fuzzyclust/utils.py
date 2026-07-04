"""Utility functions for fuzzy clustering."""

import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, silhouette_samples, homogeneity_completeness_v_measure)
from scipy.spatial.distance import cdist

def plot_results(X, y, title="Fuzzy Clustering Results"):
  """X is the original data, y is the true labels (discrete)."""
  plt.figure(figsize=(8, 6))
  scatter = plt.scatter(
      X[:, 0], X[:, 1],
      c=y,
      cmap="viridis",
      s=20,
      alpha=0.7
  )

  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.title(title)
  plt.axis("equal")
  plt.grid(True, alpha=0.3)

  # Legend
  handles, labels = scatter.legend_elements()
  plt.legend(handles, [f"Cluster {i}" for i in range(3)])

  plt.show()



# _________________________________________________________________

# Validity-index helpers

# Fuzzy clustering metrics
def partition_coefficient(u):
    """PC = (1/N) sum_ij mu_ij^2.  Range [1/c, 1]; higher = crisper."""
    return float(np.sum(u ** 2) / u.shape[0])

def modified_partition_coefficient(u):
    """MPC = 1 - C/(C-1) * (1 - PC).  Range [0, 1]; lower = crisper."""
    PC = partition_coefficient(u)
    C = u.shape[1]
    return float(1 - C/(C-1) * (1 - PC))

def partition_entropy(u):
    """PE = -(1/N) sum_ij u_ij * ln(u_ij). Range [0, ln(c)]; lower = crisper."""
    N = u.shape[0]
    u = np.where(u == 0, 1e-10, u) # prevent log(0)
    entropy_sum = np.sum(u * np.log2(u))
    partition_entropy = - (1.0 / N) * entropy_sum
    return partition_entropy

def xie_beni(X, u, centers, distance_metric="sqeuclidean"):
    """XBI = [sum_ij mu^m ||x_i - v_j||^2]   /   [N * min_{i!=j} ||v_i - v_j||^2].
    Lower = better (compactness / separation)"""
    if centers.shape[0] < 2:
        return np.nan
    um = u ** 2
    numerator = np.sum(um * distance(X, u, centers, distance_metric=distance_metric))
    # for the denominator we just use sqeuclidean since the distance is between centers, not between points and centers
    cc = cdist(centers, centers, "sqeuclidean") 
    np.fill_diagonal(cc, np.inf)
    min_sep = cc.min()
    return float(numerator / (X.shape[0] * min_sep)) if min_sep > 0 else np.nan

def fukuyama_sugeno(X, u, centers, distance_metric="sqeuclidean"):
    """FSI = sum_ij mu^m ||x_i - v_j||^2  -  sum_ij mu^m ||v_j - xbar||^2.
    Lower = better (compactness minus separation)."""
    um = u ** 2
    compact = np.sum(um * distance(X, u, centers, distance_metric=distance_metric))
    xbar = X.mean(axis=0) # global centroid
    sep_j = np.sum((centers - xbar) ** 2, axis=1)          
    separation = np.sum(um * sep_j[None, :])
    return float(compact - separation)

def fuzzy_silhouette(X, u, centers):
    """
    NOT IDEAL FOR GK CLUSTERING, since it uses Euclidean distance by default.
    FS = sum_i (u_ij - u_ij')^alpha * s_i  /  sum_i (u_ij - u_ij')^alpha

    where s_i is the crisp silhouette of sample i, and u_ij, u_ij' are the
    largest and second-largest membership values of row i of U.

    Higher = better
    """
    X = np.asarray(X)
    u = np.asarray(u, dtype=float)
    labels = np.argmax(u, axis=1)

    # Safe check
    k = len(np.unique(labels))
    if k < 2 or k > len(labels) - 1:
        return np.nan
    try:
        s = silhouette_samples(X, labels, metric="euclidean")  
    except Exception:
        return np.nan
    
    # Two largest memberships per row
    u_sorted = np.sort(u, axis=1)
    u_ij = u_sorted[:, -1]        
    u_ij_prime = u_sorted[:, -2]  

    weights = (u_ij - u_ij_prime) 
    return float(np.sum(weights * s) / np.sum(weights))




# Discrete clustering metrics (for comparison with true labels)
def safe_silhouette(X, labels):
    """
    NOT IDEAL FOR GK CLUSTERING, since it uses Euclidean distance by default.
    This function returns the mean Silhouette Coefficient over all samples.
    silhouette_score, but returns nan instead of raising when there are fewer than 2 (or more than N-1) distinct clusters.
    Range [-1, 1]; higher = crisper.
    """
    k = len(np.unique(labels))
    if k < 2 or k > len(labels) - 1:
        return np.nan
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return np.nan
    



# Distance helpers
def distance(X, u=None, centers=None, distance_metric="sqeuclidean", P=None):
        # return cdist(X, self.centers, metric="euclidean")
        if distance_metric == "sqeuclidean":
            return cdist(X, centers, metric="sqeuclidean")
        else:
            return distance_gk(X, u, centers)
            # return cdist(X, self.centers, metric="mahalanobis", VI=self.inverse_covariance(X, self.centers))

def distance_gk(X, u, centers, P=None, reg=1e-10, squared=True):
        """Mahalanobis distance (inner part of E-GK, before radius subtraction).
        
        A_i  = |P_i|^(1/n) * P_i^{-1}
        d_ik = sqrt( (x_k - v_i)^T A_i (x_k - v_i) )
        
        Returns
        -------
        D : (N, C) array of Mahalanobis distances (NON-squared).
        """
        X = np.asarray(X, dtype=float)
        N, n = X.shape
        C = centers.shape[0]

        if P is None:
            P = covariance_matrix(X, u, centers)        

        D = np.empty((N, C))
        for i in range(C):
            diff = X - centers[i]   # (N, n)
            Pi = P[i] + reg * np.eye(n)  
            vol = np.abs(np.linalg.det(Pi)) ** (1.0 / n)  # |P_i|^(1/n)
            A = vol * np.linalg.inv(Pi)       # norm matrix A_i
            maha2 = np.maximum(np.einsum("kj,jl,kl->k", diff, A, diff), 0.0)  # (x-v)^T A (x-v)
            D[:, i] = maha2 if squared else np.sqrt(maha2)  
        return D   # (N, C)

def covariance_matrix(X, u, centers):
    """Fuzzy covariance P_j for every cluster

        P_j = sum_i (u_ij^m) (x_i - v_j)(x_i - v_j)^T   / sum_i (u_ij^m)

    Returns P : (C, n, n) array, one covariance matrix per cluster
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    C, n = centers.shape  # n = features
    um = np.asarray(u, dtype=float) ** 2                 

    P = np.empty((C, n, n))
    for j in range(C):
        diff = X - centers[j]                            
        w = um[:, j]                               
        denom = w.sum()  # sum of the whole column j
        denom = denom if denom > 1e-10 else 1e-10
        # P[j] = (diff * w[:, None]).T @ diff / denom     # BORRAR

        # diff[:, :, None] becomes a stack of N matrices, each sized n x 1
        # diff[:, None, :] becomes a stack of N matrices, each sized 1 x n
        # The @ operator multiplies them to create N matrices of shape (n, n)
        # w[:, None, None] reshapes weights to (N, 1, 1) 
        outer_products = diff[:, :, None] @ diff[:, None, :]
        weighted_products = w[:, None, None] * outer_products # each corresponding weight for the corresponding outer product
        P[j] = np.sum(weighted_products, axis=0) / denom  # collapse the N dimension
    
    return P

def covariance_matrix(X, u, centers):
    """Fuzzy covariance P_j for every cluster

        P_j = sum_i (u_ij^m) (x_i - v_j)(x_i - v_j)^T   / sum_i (u_ij^m)

    Returns P : (C, n, n) array, one covariance matrix per cluster
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    C, n = centers.shape  # C = clusters, n = features
    um = np.asarray(u, dtype=float) ** 2                 

    P = np.empty((C, n, n))
    for j in range(C):
        diff = X - centers[j]  # Shape: (N, n)                            
        w = um[:, j]           # Shape: (N,)                      
        denom = w.sum() 
        denom = denom if denom > 1e-10 else 1e-10
        
        # 1. The `diff @ diff` product (Batched Outer Product)
        # diff[:, :, None] makes N column vectors of shape (n, 1)
        # diff[:, None, :] makes N row vectors of shape (1, n)
        # The @ operator multiplies them to create N matrices of shape (n, n)
        outer_products = diff[:, :, None] @ diff[:, None, :]
        
        # 2. Multiplication by u (Apply the fuzzy weights)
        # w[:, None, None] reshapes weights to (N, 1, 1) so they broadcast perfectly
        weighted_products = w[:, None, None] * outer_products
        
        # 3. The sum (Collapse the N dimension and divide by denominator)
        P[j] = np.sum(weighted_products, axis=0) / denom
        
    return P