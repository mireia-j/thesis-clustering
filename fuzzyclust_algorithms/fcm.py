import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets
import matplotlib.pyplot as plt

try:
    from base import plot_2, BaseFuzzyClustering   
except ImportError:
    from fuzzyclust_algorithms.base import plot_2, BaseFuzzyClustering


class FCM(BaseFuzzyClustering):
    """Fuzzy C-Means."""
    def fit(self, X):
        u = self._init_membership(X.shape[0])               
        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)           
            D = self.distance(X, u)  
            u = self.update_membership(D)
            u, merged = self.merge_clusters(u)                     
            if not merged and u.shape == u_old.shape:         
                if np.linalg.norm(u - u_old) < self.error:
                    break
        self.u = u
        self.centers = self.next_centers(X, u)
        return self
    
    def update_membership(self, D):
        power = float(2 / (self.m - 1))
        dist = D ** power
        denominator_ = dist.reshape((D.shape[0], 1, -1)).repeat(dist.shape[-1], axis=1)
        denominator_ = dist[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)
    


if __name__ == "__main__":
    # EXAMPLE 1
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # model = FCM(n_clusters=5, merge=False)
    # model.fit(X)

    # print(f"started with {model.n_clusters_init} clusters, " f"ended with {model.n_clusters}")
    # print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    # plot_2(model, X)



    # EXAMPLE 2 (Mahalanobis distance)
    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    # model_2 = FCM(n_clusters=10, merge=True, random_state=42, distance_metric='mahalanobis', epsilon_1=0.00005)
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

    model_3 = FCM(n_clusters=2, merge=False, random_state=100, distance_metric='mahalanobis', epsilon_1=0.00005)
    model_3.fit(X_3)
    plot_2(model_3, X_3)

    

