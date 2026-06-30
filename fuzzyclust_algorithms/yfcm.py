import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from base import BaseKanzawa, plot_2

class YFCM(BaseKanzawa):
    """Y-type / penalized FCM (Yang, 1993). Algorithm 1, YFCM branch."""

    def update_alpha(self, u, D):
        # eq. 10:  alpha_i = sum_k u_ik^m / sum_i sum_k u_ik^m
        mass = (u ** self.m).sum(axis=0)
        return mass / mass.sum()

    def update_membership(self, D):
        # eq. 12:  u_ik ∝ (lambda d_ik - ln(alpha_i))^(1/(1-m))
        power = 1.0 / (1.0 - self.m)
        term = self.lambda_ * D - np.log(self.alpha)[None, :]   # (N, C), > 0
        num = term ** power
        return num / num.sum(axis=1, keepdims=True)
    


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = YFCM(n_clusters=5, merge=False, random_state=0)
    model.fit(X)

    print(f"started with {model.n_clusters_init} clusters, "
          f"ended with {model.n_clusters}")
    print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    plot_2(model, X, show_centers=True)