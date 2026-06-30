import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from base import BaseFuzzyClustering, plot_2


class eFCM(BaseFuzzyClustering):
    """Extended Fuzzy C-Means (Kaymak & Setnes, 2002).
    
        n_clusters  -> M^(0), the initial (over)estimate of clusters
        m           -> fuzziness exponent m (> 1)
        error       -> eps_2, termination threshold on ||U^(l) - U^(l-1)||
        epsilon_1   -> eps_1, threshold on the change of max similarity
        merge       -> turn the merging step on/off (off = fixed-size E-FCM)
    """
        
    def __init__(self, n_clusters=10, m=2, max_iter=150,
                 error=1e-5, random_state=42, merge=False, epsilon_1=0.01):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state, merge=merge, epsilon_1=epsilon_1)

        self.radii = None                  
        self.beta_ = 1.0   # beta^(0)


    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        u = self._init_membership(N)        # U^(0)                        

        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)
            self.radii = self.covariance_radius(X, u) # Step 2
            d2_mod = self.modified_sq_distances(X) # Step 3
            u = self.update_membership(d2_mod) # Step 4
            u, merged = self.merge_clusters(u)                     
            if not merged and u.shape == u_old.shape:         
                if np.linalg.norm(u - u_old) < self.error:
                    break

        self.u = u
        return self

    def covariance_radius(self, X, u):
        """Step 2: fuzzy covariance P_i (eq. 2) and volume-prototype radius.

        For E-FCM the radius is the geometric-mean radius that preserves the
        cluster volume (eq. 17), scaled by the gradual-growth factor beta/M:

            r_i = (beta / M) * sqrt(|P_i|^(1/n))
        """
        n = X.shape[1]                             # number of features
        radii = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            um = u[:, i] ** self.m                 # (u_ik)^m, shape (N,)
            denom = um.sum()
            if denom <= 1e-12:           
                radii[i] = 0.0
                continue
            diff = X - self.centers[i]                   # (x_k - v_i), shape (N, n)
            P = (um[:, None] * diff).T @ diff / denom    # eq. (2), (n, n)
            det = max(np.linalg.det(P), 0.0)
            radii[i] = (self.beta_ / self.n_clusters) * det ** (1.0 / (2 * n))

        return radii

    def modified_sq_distances(self, X):
        """
            d~_ik^2 = max(0, ||x_k - v_i||^2 - r_i^2)
        """
        d2 = self.distance_squared(X)        
        return np.maximum(0.0, d2 - self.radii[None, :] ** 2)

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

    def on_no_merge(self):
        """eFCM overrides to enlarge prototypes (beta)"""
        self.beta_ = min(self.n_clusters, self.beta_ + 1) 




if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = eFCM(n_clusters=3, merge=False)
    model.fit(X)

    print(f"started with {model.n_clusters_init} clusters, "
          f"ended with {model.n_clusters}")
    print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    plot_2(model, X)