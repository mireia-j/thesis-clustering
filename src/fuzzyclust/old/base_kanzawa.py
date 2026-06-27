import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.base import BaseFuzzyClustering, plot_2


class BaseKanzawa(BaseFuzzyClustering):
    """Base for the Fuzzy C-Means (Kanzawa et al., 2024).

    Modified FCM regularized with the q-divergence between the membership
    and a per-cluster size controller alpha_i (sum_i alpha_i = 1).
    Implements Algorithm 1 (QFCM branch) of Kanzawa (2024).

    Parameter mapping to the paper:
        n_clusters -> C
        m          -> B-parameter / q-divergence parameter (> 1)
        lambda_    -> regularizing parameter lambda (> 0)
        error      -> stopping threshold on ||U_new - U_old||
    """

    def __init__(self, n_clusters=10, m=2, lambda_=1.0, max_iter=150,
                 error=1e-5, random_state=42):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state)

        self.lambda_ = lambda_
        self.alpha = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        u = self._init_membership(X.shape[0])                 # Step 1
        self.alpha = np.full(self.n_clusters, 1.0 / self.n_clusters)
        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)            # Step 2 
            D = cdist(X, self.centers, metric="sqeuclidean")  
            self.alpha = self.update_alpha(u, D)             # Step 3 (eq. 11)
            u = self.update_membership(D)                    # Step 4 (eq. 13)
            if np.linalg.norm(u - u_old) < self.error:        # Step 5
                break
        self.u = u
        return self

    def update_alpha(self, u, D):
        pass

    def update_membership(self, D):
        pass
    
    # Make predict()/labels() use the fuzzy classification function.
    def _predict(self, X):
        D = cdist(X, self.centers, metric="sqeuclidean")
        return self.update_membership(D)
    

class QFCM(BaseKanzawa):
    """q-divergence-based Fuzzy C-Means (Miyamoto et al., 2008).

    Modified FCM regularized with the q-divergence between the membership
    and a per-cluster size controller alpha_i (sum_i alpha_i = 1).
    Implements Algorithm 1 (QFCM branch) of Kanzawa (2024).

    Parameter mapping to the paper:
        n_clusters -> C
        m          -> B-parameter / q-divergence parameter (> 1)
        lambda_    -> regularizing parameter lambda (> 0)
        error      -> stopping threshold on ||U_new - U_old||
    """

    def __init__(self, n_clusters=10, m=2, lambda_=1.0, max_iter=150,
                 error=1e-5, random_state=42, alpha=None):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state, lambda_=lambda_, alpha=alpha)

    def factor(self, D):
        return 1.0 - self.lambda_ * (1.0 - self.m) * D

    def update_alpha(self, u, D):
        #   alpha_i ∝ [ sum_k u_ik^m (1 - lambda(1-m) d_ik) ]^(1/m)
        S = ((u ** self.m) * self.factor(D)).sum(axis=0)       # (C,)
        a = S ** (1.0 / self.m)
        return a / a.sum()

    def update_membership(self, D):
        #   u_ik ∝ alpha_i (1 - lambda(1-m) d_ik)^(1/(1-m))
        power = 1.0 / (1.0 - self.m)
        weighted = self.alpha[None, :] * self.factor(D) ** power   # (N, C)
        return weighted / weighted.sum(axis=1, keepdims=True)
    
    # Make predict()/labels() use the fuzzy classification function.
    def _predict(self, X):
        D = cdist(X, self.centers, metric="sqeuclidean")
        return self.update_membership(D)


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    def disc(n, cx, cy, r):
        rad = r * np.sqrt(rng.rand(n))
        ang = 2 * np.pi * rng.rand(n)
        return np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]

    X = np.vstack([disc(20, 2, 5, 1), disc(80, 7, 5, 2)])   # paper's Fig. 1
    true_labels = np.array([0] * 20 + [1] * 80)

    model = QFCM(n_clusters=2, m=2, lambda_=1.0).fit(X)
    print("alpha:", np.round(model.alpha, 3))
    print("centers:\n", np.round(model.centers, 2))
    print("ARI:", round(adjusted_rand_score(true_labels, model.labels()), 4))
    plot_2(model, X)