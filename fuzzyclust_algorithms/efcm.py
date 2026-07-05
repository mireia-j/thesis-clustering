import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets

try:
    from base import plot_2, BaseFuzzyClustering   
except ImportError:
    from fuzzyclust_algorithms.base import plot_2, BaseFuzzyClustering


class eFCM(BaseFuzzyClustering):
    """Extended Fuzzy C-Means (Kaymak & Setnes, 2002).
    
        n_clusters  -> M^(0), the initial (over)estimate of clusters
        m           -> fuzziness exponent m (> 1)
        error       -> eps_2, termination threshold on ||U^(l) - U^(l-1)||
        epsilon_1   -> eps_1, threshold on the change of max similarity
        merge       -> turn the merging step on/off (off = fixed-size E-FCM)
    """
        
    def __init__(self, n_clusters=10, m=2, max_iter=150,
                 error=1e-5, random_state=42, merge=False, epsilon_1=0.01, threshold_merge="original", distance_metric='euclidean'):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state, merge=merge, epsilon_1=epsilon_1, threshold_merge=threshold_merge, distance_metric=distance_metric)

        self.radii = None                  
        self.beta_ = 1.0   # beta^(0)


    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        u = self._init_membership(N)        # U^(0)                        

        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)
            P = self.covariance_matrix(X, u) 
            self.radii = self.radius_centers(P) 
            d2_mod = self.modified_sq_distances(X, u, P=P) 
            u = self.update_membership(d2_mod) # Step 4
            u, merged = self.merge_clusters(u)                     
            if not merged and u.shape == u_old.shape:         
                if np.linalg.norm(u - u_old) < self.error:
                    break

        self.u = u
        return self
    
    def radius_centers(self, P):
        """Volume-prototype radius r_i (Step 2).
    
            r_i = beta * sqrt( |P_i|^(1/n) / M )
    
        Parameters
        ----------
        P    : (C, n, n) covariances from covariance_matrix().
    
        Returns
        -------
        r : (C,) array of radii.
        """
        P = np.asarray(P, dtype=float)
        C, n, _ = P.shape
        vol = np.abs(np.linalg.det(P)) ** (1.0 / n)          # |P_i|^(1/n), (C,)
        return (self.beta_ / C) * np.sqrt(vol)

    def modified_sq_distances(self, X, u, P=None): # REVISAR
        """E-GK modified distance: Mahalanobis minus volume-prototype radius.
        
        d~_ik = max(0, d_ik - r_i)
        
        Returns
        -------
        D : (N, C) array.
        """
        d = self.distance(X, u, P=P)                    # (N, C)
        C = self.centers.shape[0]
        
        # Subtract radius per cluster (broadcast r from (C,) to (N, C))
        return np.maximum(0.0, d - self.radii[np.newaxis, :])
        # or equivalently, explicitly per cluster:
        # D = np.empty_like(d)
        # for i in range(C):
        #     D[:, i] = np.maximum(0.0, d[:, i] - self.radii[i])
        # return D

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

    # model = eFCM(n_clusters=3, merge=False)
    # model.fit(X)

    # print(f"started with {model.n_clusters_init} clusters, "
    #       f"ended with {model.n_clusters}")
    # print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    # plot_2(model, X)



    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    model_2 = eFCM(n_clusters=10, merge=True, random_state=0, distance_metric='mahalanobis', threshold_merge=0.5)
    model_2.fit(X_aniso)
    plot_2(model_2, X_aniso)


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

    # model_3 = eFCM(n_clusters=2, merge=False, random_state=42, distance_metric='mahalanobis')
    # model_3.fit(X_3)
    # plot_2(model_3, X_3)