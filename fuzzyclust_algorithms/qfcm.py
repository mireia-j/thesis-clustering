import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets
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
    # print(f"started with {model.n_clusters_init} clusters, f"ended with {model.n_clusters}")
    # print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    # plot_2(model, X)



    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    model_2 = QFCM(n_clusters=10, merge=True, random_state=0, distance_metric='mahalanobis', threshold_merge="original")
    model_2.fit(X_aniso)
    plot_2(model_2, X_aniso)