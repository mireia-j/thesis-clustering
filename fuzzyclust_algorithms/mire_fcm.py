import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from base import BaseFuzzyClustering

class FCM(BaseFuzzyClustering):
    """Fuzzy C-Means."""
    def fit(self, X):
        u = self._init_membership(X.shape[0])               
        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)           
            D = self.distance(X)  
            u = self.update_membership(D)
            u, merged = self.merge_clusters(u)                     
            if not merged and u.shape == u_old.shape:         
                if np.linalg.norm(u - u_old) < self.error:
                    break
        self.u = u
        return self
    
    def update_membership(self, D):
        power = float(1 / (self.m - 1))
        dist = D ** power
        denominator_ = dist.reshape((D.shape[0], 1, -1)).repeat(dist.shape[-1], axis=1)
        denominator_ = dist[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)
    


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = FCM(n_clusters=10, merge=True)
    model.fit(X)

    print(f"started with {model.n_clusters_init} clusters, "
          f"ended with {model.n_clusters}")
    print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))