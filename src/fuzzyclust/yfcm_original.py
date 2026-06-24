import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.base import BaseFuzzyClustering, plot_2


class YFCM(BaseFuzzyClustering):
    """Y-type Fuzzy C-Means / penalized FCM (Yang, 1993).

    Like ordinary FCM but with a per-cluster size controller alpha_i
    (sum_i alpha_i = 1) and a penalty governed by the regularizing
    parameter lambda. Implements Algorithm 1 (YFCM branch) of
    Kanzawa (2024).

    Parameter mapping to the paper:
        n_clusters -> C
        m          -> B-parameter / fuzzifier (> 1)
        lambda_    -> regularizing parameter lambda (> 0)
        error      -> stopping threshold on ||U_new - U_old||
    """

    def __init__(self, n_clusters=10, m=2, lambda_=1.0, max_iter=150,
                 error=1e-5, random_state=42):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state)
        self.lambda_ = lambda_
        self.alpha = None          # cluster size controllers, shape (C,)

    # --- Step 3: cluster size controller (eq. 10) ---------------------
    def _update_alpha(self, u, D):
        #   alpha_i = sum_k u_ik^m / sum_i sum_k u_ik^m
        mass = (u ** self.m).sum(axis=0)        # (C,)
        return mass / mass.sum()

    # --- Step 4: membership / fuzzy classification function (eq. 12, 20)
    def _update_membership(self, D):
        #   u_ik ∝ (lambda * d_ik - ln(alpha_i))^(1/(1-m))
        power = 1.0 / (1.0 - self.m)
        term = self.lambda_ * D - np.log(self.alpha)[None, :]    # (N, C), > 0
        num = term ** power
        return num / num.sum(axis=1, keepdims=True)

    # --- Algorithm 1 -------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        u = self._init_membership(X.shape[0])                 # Step 1
        self.alpha = np.full(self.n_clusters, 1.0 / self.n_clusters)
        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)            # Step 2 (eq. 9)
            D = cdist(X, self.centers, metric="sqeuclidean")  # d_ik (eq. 2)
            self.alpha = self._update_alpha(u, D)             # Step 3 (eq. 10)
            u = self._update_membership(D)                    # Step 4 (eq. 12)
            if np.linalg.norm(u - u_old) < self.error:        # Step 5
                break
        self.u = u
        return self

    # Make predict()/labels() use the fuzzy classification function.
    def _predict(self, X):
        D = cdist(X, self.centers, metric="sqeuclidean")
        return self._update_membership(D)


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    def disc(n, cx, cy, r):
        rad = r * np.sqrt(rng.rand(n))
        ang = 2 * np.pi * rng.rand(n)
        return np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]

    X = np.vstack([disc(20, 2, 5, 1), disc(80, 7, 5, 2)])   # paper's Fig. 1
    true_labels = np.array([0] * 20 + [1] * 80)

    model = YFCM(n_clusters=2, m=2, lambda_=1.0).fit(X)
    print("alpha:", np.round(model.alpha, 3))
    print("centers:\n", np.round(model.centers, 2))
    print("ARI:", round(adjusted_rand_score(true_labels, model.labels()), 4))
    plot_2(model, X)
