import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

# Import custom clustering algorithms safely
try:
    from fcm import FCM
    from efcm import eFCM
    from qfcm import QFCM
    from yfcm import YFCM
    from eqfcm import eQFCM
except ImportError:
    from fuzzyclust_algorithms.fcm import FCM
    from fuzzyclust_algorithms.efcm import eFCM
    from fuzzyclust_algorithms.qfcm import QFCM
    from fuzzyclust_algorithms.yfcm import YFCM
    from fuzzyclust_algorithms.eqfcm import eQFCM

# ==============================================================================
# 1. DATASET GENERATION
# ==============================================================================
n = 1500
rng = np.random.RandomState(170)

# Euclidean Datasets
blobs1 = datasets.make_blobs(n_samples=n, random_state=170)
blobs2 = datasets.make_blobs(
    n_samples=[200, 200, 100, 200],
    centers=np.array([[1, 1], [8, 2], [6, 8], [3, 9]]),
    cluster_std=0.5,
    random_state=0
)
varied1 = datasets.make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=170)
varied2 = datasets.make_blobs(
    n_samples=n,
    centers=np.array([[-5, -5], [16, 5], [11, 13], [5, 9]]),
    cluster_std=[1.0, 2.5, 0.7, 2.0],
    random_state=0
)

blobs1_scaled = (StandardScaler().fit_transform(np.asarray(blobs1[0], dtype=float)), blobs1[1])
blobs2_scaled = (StandardScaler().fit_transform(np.asarray(blobs2[0], dtype=float)), blobs2[1])
varied1_scaled = (StandardScaler().fit_transform(np.asarray(varied1[0], dtype=float)), varied1[1])
varied2_scaled = (StandardScaler().fit_transform(np.asarray(varied2[0], dtype=float)), varied2[1])

TOY_EUC = {
    "blobs1": {"data": blobs1_scaled, "C_upper": 10},
    "blobs2": {"data": blobs2_scaled, "C_upper": 10},
    "varied1": {"data": varied1_scaled, "C_upper": 10},
    "varied2": {"data": varied2_scaled, "C_upper": 10}
}

# Mahalanobis Datasets
Xa, y_aniso = datasets.make_blobs(n_samples=n, random_state=170)
aniso = (Xa @ [[0.6, -0.6], [-0.4, 0.8]], y_aniso)

def make_elliptical_3(n_samples=500, random_state=0):
    rng = np.random.RandomState(random_state)
    centers = np.array([[2.0, 8.0], [8.0, 2.0], [6.0, 6.0]])
    def cov(sx, sy, deg):
        t = np.deg2rad(deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return R @ np.diag([sx**2, sy**2]) @ R.T
    covs = [cov(1.4, 0.5, 30), cov(1.3, 0.4, -20), cov(1.5, 0.6, 75)]
    sizes = [167, 167, 166]
    X, y = [], []
    for k, (mu, C, s) in enumerate(zip(centers, covs, sizes)):
        X.append(rng.multivariate_normal(mu, C, size=s))
        y.append(np.full(s, k))
    return np.vstack(X), np.concatenate(y)

X_elliptical_3, y_elliptical_3 = make_elliptical_3()
elliptical = (X_elliptical_3, y_elliptical_3)

def _segment(p0, p1, n_pts, noise_val, rng_obj):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    t = rng_obj.uniform(0.0, 1.0, size=n_pts)[:, None]
    return p0 + t * (p1 - p0) + rng_obj.normal(0.0, noise_val, size=(n_pts, 2))

def make_lines(segments, rng_obj):
    X, y = [], []
    for lbl, (p0, p1, n_pts, noise_val) in enumerate(segments):
        X.append(_segment(p0, p1, n_pts, noise_val, rng_obj))
        y.append(np.full(n_pts, lbl))
    return np.vstack(X), np.concatenate(y)

def _letter(strokes, x0, w, h, y0, npts, noise_val):
    return [((x0 + a[0]*w, y0 + a[1]*h), (x0 + b[0]*w, y0 + b[1]*h), npts, noise_val)
            for (a, b) in strokes]

rng = np.random.RandomState(0)
fig10 = [
    ((0.05, 0.15), (0.05, 0.80), 120, 0.015),
    ((0.10, 0.10), (1.00, 1.00), 200, 0.020),
    ((0.20, 0.90), (0.95, 0.30), 150, 0.025),
    ((0.25, 0.08), (1.15, 0.18), 100, 0.012),
    ((0.55, 0.35), (0.75, 1.00),  80, 0.030),
]
five_lines = make_lines(fig10, rng)

N_strokes = [((0,0),(0,1)), ((0,1),(1,0)), ((1,0),(1,1))]
Z_strokes = [((0,1),(1,1)), ((1,1),(0,0)), ((0,0),(1,0))]
H_strokes = [((0,0),(0,1)), ((1,0),(1,1)), ((0,0.5),(1,0.5))]
K_strokes = [((0,0),(0,1)), ((0,0.5),(1,1)), ((0,0.5),(1,0))]

rng = np.random.RandomState(1)
letters = [
    (N_strokes, 0.30,  95, 0.018),
    (Z_strokes, 0.80,  70, 0.025),
    (H_strokes, 1.30, 115, 0.015),
    (K_strokes, 1.80,  60, 0.030),
]
fig11 = []
for strokes, x0, npts, noise_val in letters:
    fig11 += _letter(strokes, x0, w=0.35, h=1.0, y0=0.0, npts=npts, noise_val=noise_val)
four_letters = make_lines(fig11, rng)

TOY_MAH = {
    "aniso": {"data": aniso, "C_upper": 10},
    "elliptical": {"data": elliptical, "C_upper": 10},
    "five_lines": {"data": five_lines, "C_upper": 15},
    "four_letters": {"data": four_letters, "C_upper": 20},
}

# ==============================================================================
# 2. CONFIGURATIONS & MODEL FACTORY
# ==============================================================================
BEST_COMBO = {
    "FCM_euc": {"m": 2, "alpha": "original", "dist": "euclidean"},
    "FCM_mah": {"m": 2, "alpha": 0.9, "dist": "mahalanobis"},
    "eFCM_euc": {"m": 2, "alpha": "original", "dist": "euclidean"},
    "eFCM_mah": {"m": 2, "alpha": "new", "dist": "mahalanobis"},
    "QFCM_euc": {"m": 1.1, "lambda": 100, "alpha": 0.1, "dist": "euclidean"},
    "QFCM_mah": {"m": 1.5, "lambda": 100, "alpha": 0.8, "dist": "mahalanobis"},
    "YFCM_euc": {"m": 1.3, "lambda": 10, "alpha": 0.2, "dist": "euclidean"},
    "YFCM_mah": {"m": 2, "lambda": 100, "alpha": "new", "dist": "mahalanobis"},
    "eQFCM_euc": {"m1": 1.1, "m2": 1.1, "lambda1": 100, "lambda2": 100, "alpha": 0.1, "dist": "euclidean"},
    "eQFCM_mah": {"m1": 1.5, "m2": 1.05, "lambda1": 100, "lambda2": 100, "alpha": 0.9, "dist": "mahalanobis"},
}

def get_model_instance(algo_name, dist_metric, n_clusters):
    """Dynamically initializes the algorithm using the pre-defined optimal parameters."""
    key = f"{algo_name}_{dist_metric[:3]}"
    params = BEST_COMBO[key]
    
    base_kwargs = {
        "n_clusters": n_clusters,
        "merge": True,
        "random_state": 0,
        "distance_metric": params["dist"],
        "threshold_merge": params["alpha"]
    }
    
    if algo_name == "FCM":
        return FCM(m=params["m"], **base_kwargs)
    elif algo_name == "eFCM":
        return eFCM(m=params["m"], **base_kwargs)
    elif algo_name == "QFCM":
        return QFCM(m=params["m"], lambda_=params["lambda"], **base_kwargs)
    elif algo_name == "YFCM":
        return YFCM(m=params["m"], lambda_=params["lambda"], **base_kwargs)
    elif algo_name == "eQFCM":
        return eQFCM(
            m1=params["m1"], m2=params["m2"],
            lambda1=params["lambda1"], lambda2=params["lambda2"],
            **base_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

# ==============================================================================
# 3. EXPERIMENTAL PIPELINE EXECUTION (UPDATED)
# ==============================================================================
algorithms = ["FCM", "eFCM", "QFCM", "YFCM", "eQFCM"]
results = []

# Process Euclidean datasets
for ds_name, ds_info in TOY_EUC.items():
    # Extract feature matrix X (data[0]) and true labels y_true (data[1])
    X, y_true = ds_info["data"]
    c_init = ds_info["C_upper"]
    
    for algo in algorithms:
        model = get_model_instance(algo, "euclidean", c_init)
        
        # Fit the model on the feature matrix
        model.fit(X)
        
        # Retrieve predicted labels using your custom method
        y_pred = model.labels()
        
        _, _, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
        ars_score = adjusted_rand_score(y_true, y_pred)
        
        results.append({
            "Distance": "euclidean",
            "Algorithm": algo,
            "Dataset": ds_name,
            "C": c_init,
            "C_tilde": getattr(model, "n_clusters_", getattr(model, "n_clusters", c_init)),
            "V": np.round(v_measure, 4),
            "ARS": np.round(ars_score, 4),
            "_n_samples": X.shape[0]  # Used temporarily for deterministic sorting
        })

# Process Mahalanobis datasets
for ds_name, ds_info in TOY_MAH.items():
    X, y_true = ds_info["data"]
    c_init = ds_info["C_upper"]
    
    # Scale feature matrix if not already standardized
    if not isinstance(X, np.ndarray) or np.abs(X.mean()) > 1.0:
        X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    
    for algo in algorithms:
        model = get_model_instance(algo, "mahalanobis", c_init)
        
        # Fit the model on the standardized feature matrix
        model.fit(X)
        
        # Retrieve predicted labels using your custom method
        y_pred = model.labels()
        
        _, _, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
        ars_score = adjusted_rand_score(y_true, y_pred)
        
        results.append({
            "Distance": "mahalanobis",
            "Algorithm": algo,
            "Dataset": ds_name,
            "C": c_init,
            "C_tilde": getattr(model, "n_clusters_", getattr(model, "n_clusters", c_init)),
            "V": np.round(v_measure, 4),
            "ARS": np.round(ars_score, 4),
            "_n_samples": X.shape[0]
        })

# ==============================================================================
# 4. STRUCTURING & EXPORT (UPDATED CUSTOM ORDER)
# ==============================================================================
df_results = pd.DataFrame(results)

# 1. Define the exact desired order by combining the dictionary keys
ordered_datasets = list(TOY_EUC.keys()) + list(TOY_MAH.keys())
# Result: ['blobs1', 'blobs2', 'varied1', 'varied2', 'aniso', 'elliptical', 'five_lines', 'four_letters']

# 2. Convert the 'Dataset' column into an ordered Categorical type
df_results["Dataset"] = pd.Categorical(
    df_results["Dataset"], 
    categories=ordered_datasets, 
    ordered=True
)

# 3. Sort hierarchically and drop the temporary _n_samples column
df_results = (
    df_results.sort_values(
        by=["Distance", "Algorithm", "Dataset"],
        ascending=[True, True, True]
    )
    .drop(columns=["_n_samples"])
    .reset_index(drop=True)
)

# Ensure output directory exists and save
output_path = "Toy_results_BEST_COMBO/final_toy_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Export to CSV (pandas automatically converts the categorical column back to text)
df_results.to_csv(output_path, index=False)

print(f"Experimental pipeline finished successfully. Results saved to: {output_path}")
print("\nFirst 10 rows of generated table:")
print(df_results.head(10).to_string(index=False))