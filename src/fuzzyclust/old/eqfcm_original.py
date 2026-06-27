import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.base import BaseFuzzyClustering, plot_2


class eQFCM(BaseFuzzyClustering):
    """Extended q-divergence-based Fuzzy C-Means (Kanzawa, 2024).

    A four-parameter fuzzy clustering model that contains both QFCM and YFCM
    as special cases. The single fuzzifier of QFCM is split so that one
    parameter acts on the membership and another on the cluster-size
    controller; likewise the regularizing parameter is split in two:

        m1 -> fuzzifier acting on the membership      (> 1)
        m2 -> fuzzifier acting on the size controller (> 1)
        lambda1, lambda2 -> the two regularizing parameters (> 0)

    Reductions proved in the paper:
        m1 = m2 and lambda1 = lambda2           -> QFCM
        m1 = m, lambda1 = lambda,
        m2 -> 1+, lambda2 -> +inf               -> YFCM

    Implements Algorithm 2. (n_clusters -> C, error -> stop threshold on
    ||U_new - U_old||.)
    """

    def __init__(self, n_clusters=10, m1=2, m2=2, lambda1=1.0, lambda2=1.0,
                 max_iter=150, error=1e-5, random_state=42):
        # base `m` is the center/membership exponent, which here is m1
        super().__init__(n_clusters=n_clusters, m=m1, max_iter=max_iter,
                         error=error, random_state=random_state)
        self.m1 = m1
        self.m2 = m2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = None          # cluster size controllers, shape (C,)

    # Common factor 1 - lambda1*(1-m2)*d_ik = 1 + lambda1*(m2-1)*d_ik (>= 1)
    def _factor1(self, D):
        return 1.0 - self.lambda1 * (1.0 - self.m2) * D

    # --- Step 3: cluster size controller (eq. 36) --------------------
    def _update_alpha(self, u, D):
        #   alpha_i ∝ [ sum_k u_ik^m1 (1 - lambda1(1-m2) d_ik) ]^(1/m2)
        S = ((u ** self.m1) * self._factor1(D)).sum(axis=0)        # (C,)
        a = S ** (1.0 / self.m2)
        return a / a.sum()

    # --- Step 4: membership / fuzzy classification function (eq. 38, 43)
    def _update_membership(self, D):
        #   inner_ik = lambda2(m1-1)[ alpha_i^(1-m2)(1 - lambda1(1-m2)d_ik) - 1 ]
        #              + lambda1(m2-1)
        #   u_ik ∝ inner_ik^(1/(1-m1))
        bracket = (self.alpha[None, :] ** (1.0 - self.m2)) * self._factor1(D) - 1.0
        inner = (self.lambda2 * (self.m1 - 1.0) * bracket
                 + self.lambda1 * (self.m2 - 1.0))
        num = inner ** (1.0 / (1.0 - self.m1))
        return num / num.sum(axis=1, keepdims=True)

    # --- Algorithm 2 -------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        u = self._init_membership(X.shape[0])                 # Step 1
        self.alpha = np.full(self.n_clusters, 1.0 / self.n_clusters)
        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)            # Step 2 (eq. 34)
            D = cdist(X, self.centers, metric="sqeuclidean")  # d_ik (eq. 2)
            self.alpha = self._update_alpha(u, D)             # Step 3 (eq. 36)
            u = self._update_membership(D)                    # Step 4 (eq. 38)
            if np.linalg.norm(u - u_old) < self.error:        # Step 5
                break
        self.u = u
        return self

    # Make predict()/labels() use the fuzzy classification function (eq. 43).
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

    # QFCM-equivalent setting (m1=m2, lambda1=lambda2)
    model = eQFCM(n_clusters=2, m1=2, m2=2, lambda1=1.0, lambda2=1.0).fit(X)
    print("alpha:", np.round(model.alpha, 3))
    print("centers:\n", np.round(model.centers, 2))
    print("ARI:", round(adjusted_rand_score(true_labels, model.labels()), 4))
    plot_2(model, X)
