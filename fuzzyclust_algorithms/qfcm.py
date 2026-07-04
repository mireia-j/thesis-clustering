import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             silhouette_score)

try:
    from base import plot_2, BaseKanzawa   
except ImportError:
    from fuzzyclust_algorithms.base import plot_2, BaseKanzawa

class QFCM(BaseKanzawa):
    """q-divergence-based FCM (Miyamoto et al., 2008)"""

    def factor(self, D):
        # 1 - lambda(1-m) d_ik = 1 + lambda(m-1) d_ik  (>= 1 > 0)
        return 1.0 - self.lambda_ * (1.0 - self.m) * D

    def update_alpha(self, u, D):
        # eq. 11:  alpha_i ∝ [ sum_k u_ik^m (1 - lambda(1-m) d_ik) ]^(1/m)
        S = ((u ** self.m) * self.factor(D)).sum(axis=0)       # (C,)
        a = S ** (1.0 / self.m)
        return a / a.sum()

    def update_membership(self, D):
        # eq. 13:  u_ik ∝ alpha_i (1 - lambda(1-m) d_ik)^(1/(1-m))
        power = 1.0 / (1.0 - self.m)
        weighted = self.alpha[None, :] * self.factor(D) ** power   # (N, C)
        return weighted / weighted.sum(axis=1, keepdims=True)
    


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = QFCM(n_clusters=3, merge=True, random_state=0)
    model.fit(X)

    print(f"started with {model.n_clusters_init} clusters, "
          f"ended with {model.n_clusters}")
    print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    # plot_2(model, X, show_centers=True)

    # EXAMPLE 2 (Mahalanobis distance)
    n_samples = 1500
    X, y_aniso = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    # model_2 = QFCM(n_clusters=10, merge=True, random_state=0, distance_metric='mahalanobis', epsilon_1=0.00005)
    # model_2.fit(X_aniso)
    # plot_2(model_2, X_aniso)


    # EXAMPLE 3 (Mahalanobis distance)

    def line_1(x):
        y = (1-4*x)/5
        return np.column_stack((x, y))  # shape (n, 2)
    
    def line_2(x):
        y = 2*x
        return np.column_stack((x, y))  # shape (n, 2)
    
    x_values = np.random.uniform(-1, 1, 100)
    line_eq_1 = line_1(x_values)
    line_eq_2 = line_2(x_values)

    X_3 = np.vstack((line_eq_1, line_eq_2))

    # model_3 = QFCM(n_clusters=10, merge=True, random_state=100, distance_metric='mahalanobis', epsilon_1=0.00005)
    # model_3.fit(X_3)
    # plot_2(model_3, X_3)


    def bootstrap_metrics_2(X, y, n_boot=100, random_state=0, param_m=1.5, param_lam=1.0):
        """fit_predict_fn(Xb) -> hard labels for the resampled data Xb.
        Returns {metric: (mean, std)} over n_boot resamples."""
        rng = np.random.RandomState(random_state)
        N = X.shape[0]
        rows = []
        for _ in range(n_boot):
            idx = rng.randint(0, N, size=N)          # sample N indices WITH replacement
            Xb, yb = X[idx], y[idx]
            model = QFCM(n_clusters=3, merge=False, random_state=0, m=param_m, lamda_=param_lam)
            model.fit(X)
            labels = model.labels()             # run your GK / n-PyGK here
            rows.append({
                "ARS": adjusted_rand_score(yb, labels),
                "AMI": adjusted_mutual_info_score(yb, labels),
                "Silhouette": silhouette_score(Xb, labels),
                # ... add PC, XBI, FSI, homogeneity, completeness, v-measure
            })
        
        df = pd.DataFrame(rows)
        return {m: (df[m].mean(), df[m].std()) for m in df.columns}

    
    results = []

for param_lam in [0.1, 1.0, 10.0]:
    for param_m in [1.1, 1.5, 2.0]:

        metrics = bootstrap_metrics_2(X_aniso, y_aniso, n_boot=100, random_state=0, param_m=param_m, param_lam=param_lam)

        row = {"param_m": param_m, "param_lam": param_lam}

        for metric, (mean_val, std_val) in metrics.items():
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val

        results.append(row)

df_results = pd.DataFrame(results)
print(df_results)