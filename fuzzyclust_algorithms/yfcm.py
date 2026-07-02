import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets

try:
    from base import plot_2, BaseKanzawa   
except ImportError:
    from fuzzyclust_algorithms.base import plot_2, BaseKanzawa

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

    # model = YFCM(n_clusters=5, merge=False, random_state=0)
    # model.fit(X)

    # print(f"started with {model.n_clusters_init} clusters, "f"ended with {model.n_clusters}")
    # print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    # plot_2(model, X, show_centers=True)

    # EXAMPLE 2 (Mahalanobis distance)
    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    # model_2 = YFCM(n_clusters=10, merge=True, random_state=100, distance_metric='mahalanobis')
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

    model_3 = YFCM(n_clusters=6, merge=True, random_state=10, distance_metric='mahalanobis', epsilon_1=0.00005)
    model_3.fit(X_3)
    plot_2(model_3, X_3)