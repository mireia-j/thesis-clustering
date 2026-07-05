"""Utility functions for fuzzy clustering."""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, silhouette_samples, homogeneity_completeness_v_measure)
from scipy.spatial.distance import cdist
import copy

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
# BOOTSTRAP METRICS

# Partition Coefficient (PC), Modified Partition Coefficient (MPC), Partition Entropy (PE), Xie-Beni Index (XBI)
# Fukuyama-Sugeno Index (FKI), Fuzzy Silhouette (FS), Homogeneity, Completeness, V-measure
# Adjusted Rand Score (ARS), Adjusted Mutual Information (AMI), Silhouette Score
METRICS = ["PC", "MPC", "PE", "XBI", "FKI", "FS", "Hom", "Comp", "V-measure", "ARS", "AMI", "Silh"]

GRIDS = {
    "quick":  dict(m=[2.0],            m2=[1.1],
                   lam=[1.0],
                   m1=[2.0], lam1=[1.0], lam2=[1.0], alpha=["original"]),
    "medium": dict(m=[1.1, 1.5, 2.0],       m2=[1.05, 1.1, 1.3, 1.5],
                   lam=[0.1, 1.0, 10.0],
                   m1=[1.1, 1.5, 2.0], lam1=[0.1, 1.0, 10.0], lam2=[0.1, 1.0, 10.0],
                   alpha=["original", "new", 0.25, 0.5, 0.75]),
    "large":  dict(m=[1.05, 1.1, 1.3, 1.5, 2.0], m2=[1.05, 1.1, 1.3, 1.5, 1.8, 2.0],
                     lam=[0.01, 0.1, 1.0, 10.0, 100.0], 
                     m1=[1.3, 1.5, 1.8, 2.0], lam1=[0.01, 0.1, 1.0, 10.0, 100.0], lam2=[0.01, 0.1, 1.0, 10.0, 100.0],
                     alpha=["original", "new", 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    "paper":  dict(m=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   m2=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   lam=[10.0 ** e for e in range(-15, 16, 2)],
                   m1=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   lam1=[10.0 ** e for e in range(-15, 16, 2)],
                   lam2=[10.0 ** e for e in range(-15, 16, 2)]),
}

def bootstrap_metrics(model, X, y, n_boot=30, random_state=0, verbose=False):
    """
    Pass the model without fitting it yet. 
    """
    rng = np.random.RandomState(random_state)
    N = X.shape[0]
    distance_metric = model.distance_metric
    rows = []
    n_failed = 0
    for i in range(n_boot):
        idx = rng.randint(0, N, size=N)
        Xb, yb = X[idx], y[idx]

        m_i = copy.deepcopy(model)
        m_i.random_state = i**2 + 1 # initialize different U(0) every time
        m_i.fit(Xb)
        u = m_i.u
        if u is None or not np.all(np.isfinite(u)):     # dead resample -> all NaN
            n_failed += 1
            rows.append({k: np.nan for k in METRICS})
            continue

        labels = m_i.labels()
        centers = m_i.centers

        h, c, v = homogeneity_completeness_v_measure(yb, labels)
        rows.append({
            "PC": partition_coefficient(u),
            "MPC": modified_partition_coefficient(u),
            "PE": partition_entropy(u),
            "XBI": xie_beni(Xb, u, centers, distance_metric=distance_metric),
            "FKI": fukuyama_sugeno(Xb, u, centers, distance_metric=distance_metric),
            "FS": fuzzy_silhouette(Xb, u, centers),
            "Hom": h, "Comp": c, "V-measure": v,
            "ARS": adjusted_rand_score(yb, labels),
            "AMI": adjusted_mutual_info_score(yb, labels),
            "Silh": safe_silhouette(Xb, labels)
        })

    if verbose and n_failed:
        print(f"  [{n_failed}/{n_boot} resamples produced NaN memberships]")

    df = pd.DataFrame(rows)
    return {m: (df[m].mean(), df[m].std()) for m in METRICS}   # mean/std skip NaN

def create_df_FCM(model, data):
    """
    Pass the model without fitting it yet, data is a tuple (X, y)
    Create a DataFrame of metrics for FCM and eFCM.
    We are not trying with various parameters, just m=2
    Apply bootstrap and initialize U(0) differently for each resample
    """
    results = []
    m_i = copy.deepcopy(model)
    metrics = bootstrap_metrics(m_i, data[0], data[1],
                                        n_boot=30, random_state=0)

    row = {"param_m": 2}
    for metric, (mean_val, std_val) in metrics.items():
        row[f"{metric}_mean"] = mean_val
        row[f"{metric}_std"] = std_val
    results.append(row)

    return pd.DataFrame(results)

def create_df_QFCM(model, data, size):
    """
    Create a DataFrame of metrics for QFCM and YFCM.
    We are varying m and lambda
    Apply bootstrap and initialize U(0) differently for each resample
    size: "quick", "medium", "large", "paper"
    """
    results = []
    for param_lam in GRIDS[size]["lam"]:
      for param_m in GRIDS[size]["m"]:
        m_i = copy.deepcopy(model)
        m_i.m = param_m
        m_i.lambda_ = param_lam
        metrics = bootstrap_metrics(m_i, data[0], data[1],
                                        n_boot=100, random_state=0)

        row = {"param_m": param_m, "param_lam": param_lam}
        for metric, (mean_val, std_val) in metrics.items():
          row[f"{metric}_mean"] = mean_val
          row[f"{metric}_std"] = std_val
        results.append(row)

    return pd.DataFrame(results)

def create_df_eQFCM(model, data, size):
    """
    Create a DataFrame of metrics for eQFCM and YFCM.
    We are varying m1, m2 and lambda1, lamda2
    Apply bootstrap and initialize U(0) differently for each resample
    size: "quick", "medium", "large", "paper"
    """
    results = []
    for param_lam in GRIDS[size]["lam"]:
      for param_m in GRIDS[size]["m"]:
        m_i = copy.deepcopy(model)
        m_i.m = param_m
        m_i.lambda_ = param_lam
        metrics = bootstrap_metrics(m_i, data[0], data[1],
                                        n_boot=100, random_state=0)

        row = {"param_m": param_m, "param_lam": param_lam}
        for metric, (mean_val, std_val) in metrics.items():
          row[f"{metric}_mean"] = mean_val
          row[f"{metric}_std"] = std_val
        results.append(row)

    return pd.DataFrame(results)

def create_df_eQFCM(model, data, size):
    """
    Create a DataFrame of metrics for eQFCM and YFCM.
    We are varying m1, m2 and lambda1, lamda2
    Apply bootstrap and initialize U(0) differently for each resample
    size: "quick", "medium", "large", "paper"
    """
    results = []
    # carry the fixed settings from the passed model; reconstruct per config
    fixed = dict(n_clusters=model.n_clusters_init, merge=model.merge,
                 distance_metric=model.distance_metric, epsilon_1=model.epsilon_1,
                 max_iter=model.max_iter, error=model.error)

    for param_m1 in GRIDS[size]["m1"]:
        for param_m2 in GRIDS[size]["m2"]:
            for param_lam1 in GRIDS[size]["lam1"]:
                for param_lam2 in GRIDS[size]["lam2"]:
                    m_i = copy.deepcopy(model)
                    m_i.m1 = param_m1
                    m_i.m2 = param_m2
                    m_i.lambda1 = param_lam1
                    m_i.lambda2 = param_lam2
                    # eQFCM(m1=param_m1, m2=param_m2, lambda1=param_lam1, lambda2=param_lam2, **fixed)
                    metrics = bootstrap_metrics(m_i, data[0], data[1],
                                                n_boot=30, random_state=0)

                    row = {"param_m1": param_m1, "param_m2": param_m2,
                           "param_lam1": param_lam1, "param_lam2": param_lam2}
                    for metric, (mean_val, std_val) in metrics.items():
                        row[f"{metric}_mean"] = mean_val
                        row[f"{metric}_std"] = std_val
                    results.append(row)
    return pd.DataFrame(results)

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
    """FKI = sum_ij mu^m ||x_i - v_j||^2  -  sum_ij mu^m ||v_j - xbar||^2.
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



# BORRAR
# print(f"The ARI is {adjusted_rand_score(X_aniso[1], model_fcm_1.labels())}")
# print(f"The PC is {partition_coefficient(model_fcm_1.u)}")
# print(f"The MPC is {modified_partition_coefficient(model_fcm_1.u)}")
# print(f"The PE is {partition_entropy(model_fcm_1.u)}")
# print(f"The XBI is {xie_beni(X_aniso[0], model_fcm_1.u, model_fcm_1.centers)}")
# print(f"The FKI is {fukuyama_sugeno(X_aniso[0], model_fcm_1.u, model_fcm_1.centers)}")
# print(f"The FS is {fuzzy_silhouette(X_aniso[0], model_fcm_1.u, model_fcm_1.centers)}")
# print(f"The Hom is {homogeneity_completeness_v_measure(X_aniso[1], model_fcm_1.labels())[0]}")
# print(f"The Comp is {homogeneity_completeness_v_measure(X_aniso[1], model_fcm_1.labels())[1]}")
# print(f"The V-measure is {homogeneity_completeness_v_measure(X_aniso[1], model_fcm_1.labels())[2]}")
# print(f"The ARS is {adjusted_rand_score(X_aniso[1], model_fcm_1.labels())}")
# print(f"The AMI is {adjusted_mutual_info_score(X_aniso[1], model_fcm_1.labels())}")
# print(f"The Silh is {safe_silhouette(X_aniso[0], model_fcm_1.labels())}")