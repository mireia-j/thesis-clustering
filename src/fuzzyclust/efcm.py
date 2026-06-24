import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.base import BaseFuzzyClustering, plot_2

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.base import BaseFuzzyClustering, plot_2


class eFCM(BaseFuzzyClustering):
    """Extended Fuzzy C-Means (Kaymak & Setnes, 2002).
    
        n_clusters  -> M^(0), the initial (over)estimate of clusters
        m           -> fuzziness exponent m (> 1)
        error       -> eps_2, termination threshold on ||U^(l) - U^(l-1)||
        epsilon_1   -> eps_1, threshold on the change of max similarity
        merge       -> turn the merging step on/off (off = fixed-size E-FCM)
    """

    def __init__(self, n_clusters=10, m=2, max_iter=150, error=1e-3,
                 epsilon_1=0.01, merge=True, random_state=42):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state)
        self.epsilon_1 = epsilon_1          # eps_1
        self.merge = merge
        # filled in by fit():
        self.radii = None                   # r_i for each final cluster
        self.n_clusters_init = n_clusters   # remember the starting M^(0)
        self.beta_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]

        n_clusters = self.n_clusters_init = self.n_clusters
        u = self._init_membership(N)        # U^(0)
        beta = 1.0                          # beta^(0)
        prev_max_sim = 1.0                  # S^(0) (paper initialises to 1)
        centers = self.next_centers(X, u)
        radii = np.zeros(n_clusters)

        for _ in range(self.max_iter):
            u_prev = u

            centers = self.next_centers(X, u)
            radii = self.covariance_radius(X, u, centers, beta, n_clusters) # Step 2
            d2_mod = self.modified_sq_distances(X, centers, radii) # Step 3
            u = self.update_membership(d2_mod) # Step 4

            merged = False
            if self.merge and n_clusters > 1:
                (i_, j_), max_sim = self.most_similar_pair(u)   # Step 5
                if abs(max_sim - prev_max_sim) < self.epsilon_1:   # Step 6
                    alpha = 1.0 / (n_clusters - 1)       
                    if max_sim > alpha:
                        u[:, i_] = u[:, i_] + u[:, j_]    # merge j* into i*
                        u = np.delete(u, j_, axis=1)
                        n_clusters -= 1
                        merged = True
                    else:
                        beta = min(n_clusters, beta + 1)  # enlarge prototypes
                prev_max_sim = max_sim
            
            # DUDA: esto es así?
            # Termination: ||U^(l) - U^(l-1)|| < eps_2 (only when shape stable)
            if not merged and u.shape == u_prev.shape:
                if np.linalg.norm(u - u_prev) < self.error:
                    centers = self.next_centers(X, u)
                    radii = self.covariance_radius(X, u, centers, beta,
                                                    n_clusters)
                    break

        self.u = u
        self.centers = centers
        self.radii = radii
        self.n_clusters = n_clusters
        self.beta_ = beta
        return self

    def covariance_radius(self, X, u, centers, beta, n_clusters):
        """Step 2: fuzzy covariance P_i (eq. 2) and volume-prototype radius.

        For E-FCM the radius is the geometric-mean radius that preserves the
        cluster volume (eq. 17), scaled by the gradual-growth factor beta/M:

            r_i = (beta / M) * sqrt(|P_i|^(1/n))
        """
        n = X.shape[1]                             # number of features
        radii = np.zeros(n_clusters)
        for i in range(n_clusters):
            um = u[:, i] ** self.m                 # (u_ik)^m, shape (N,)
            denom = um.sum()
            if denom <= 1e-12:           
                radii[i] = 0.0
                continue
            diff = X - centers[i]                   # (x_k - v_i), shape (N, n)
            P = (um[:, None] * diff).T @ diff / denom    # eq. (2), (n, n)
            det = max(np.linalg.det(P), 0.0)
            radii[i] = (beta / n_clusters) * det ** (1.0 / (2 * n))

            # BORRAR
            if i == 1:
                print(f"Cluster {i}: det(P)={det:.4f}, radius={radii[i]:.4f}")
        return radii

    def modified_sq_distances(self, X, centers, radii):
        """
            d~_ik^2 = max(0, ||x_k - v_i||^2 - r_i^2)
        """
        d2 = cdist(X, centers, metric="sqeuclidean")        # ||x_k - v_i||^2
        return np.maximum(0.0, d2 - radii[None, :] ** 2)

    def update_membership(self, d2_mod):
        N, C = d2_mod.shape
        power = 1.0 / (self.m - 1)
        u = np.zeros((N, C))

        inside = d2_mod <= 0.0                  # phi_k: prototypes containing x_k
        has_inside = inside.any(axis=1)

        # points inside >=1 prototype: split membership equally among them
        if has_inside.any():
            z = inside[has_inside].astype(float)
            u[has_inside] = z / z.sum(axis=1, keepdims=True)

        # ordinary points: standard fuzzy rule on the modified distances
        normal = ~has_inside
        if normal.any():
            d = d2_mod[normal]                  # all entries > 0 here
            ratio = (d[:, :, None] / d[:, None, :]) ** power
            u[normal] = 1.0 / ratio.sum(axis=2)
        return u

    @staticmethod
    def most_similar_pair(u):
        """Step 5: fuzzy inclusion similarity (eq. 10) and the most similar
        pair (i*, j*)."""
        C = u.shape[1]
        col_sum = u.sum(axis=0)
        best_pair, best_sim = (0, 1), -np.inf
        for i in range(C):
            for j in range(i + 1, C):
                num = np.minimum(u[:, i], u[:, j]).sum()
                den = min(col_sum[i], col_sum[j])
                sim = num / den if den > 0 else 0.0
                if sim > best_sim:
                    best_sim, best_pair = sim, (i, j)
        return best_pair, best_sim

    # Make predict()/labels() (inherited) volume-aware for new data.
    def _predict(self, X):
        d2_mod = self.modified_sq_distances(X, self.centers, self.radii)
        return self.update_membership(d2_mod)




if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = eFCM(n_clusters=10, m=2, error=1e-3, epsilon_1=0.01)
    model.fit(X)

    print(f"started with {model.n_clusters_init} clusters, "
          f"ended with {model.n_clusters}")
    print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))
    plot_2(model, X)