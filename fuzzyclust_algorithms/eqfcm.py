import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets

try:
    from base import plot_2, BaseKanzawa   
except ImportError:
    from fuzzyclust_algorithms.base import plot_2, BaseKanzawa

class eQFCM(BaseKanzawa):
    """Extended QFCM (Kanzawa, 2024). Algorithm 2, four fuzzification params.

    Reduces to QFCM when m1=m2 and lambda1=lambda2, and to YFCM when
    m1=m, lambda1=lambda, m2->1+, lambda2->+inf.
    """

    def __init__(self, n_clusters=10, m1=2, m2=2, lambda1=1.0, lambda2=1.0,
                 max_iter=150, error=1e-5, random_state=42, merge=False, epsilon_1=0.01, distance_metric='euclidean'):
        # base `m` (used by next_centers) is the membership exponent m1
        super().__init__(n_clusters=n_clusters, m=m1, lambda_=lambda1,
                         max_iter=max_iter, error=error,
                         random_state=random_state, merge=merge, epsilon_1=epsilon_1, distance_metric=distance_metric)
        self.m1 = m1
        self.m2 = m2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def factor1(self, D):
        # 1 - lambda1(1-m2) d_ik  (>= 1 > 0)
        return 1.0 - self.lambda1 * (1.0 - self.m2) * D

    def update_alpha(self, u, D):
        # eq. 36
        S = ((u ** self.m1) * self.factor1(D)).sum(axis=0)
        a = S ** (1.0 / self.m2)
        return a / a.sum()

    def update_membership(self, D):
        # eq. 38
        bracket = (self.alpha[None, :] ** (1.0 - self.m2)) * self.factor1(D) - 1.0
        inner = (self.lambda2 * (self.m1 - 1.0) * bracket
                 + self.lambda1 * (self.m2 - 1.0))
        num = inner ** (1.0 / (1.0 - self.m1))
        return num / num.sum(axis=1, keepdims=True)
    


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    model = eQFCM(n_clusters=3, merge=False, random_state=0)
    model.fit(X)

    print(f"started with {model.n_clusters_init} clusters, "
          f"ended with {model.n_clusters}")
    print("ARI vs true labels:", adjusted_rand_score(true_labels, model.labels()))

    # plot_2(model, X, show_centers=True)

    # EXAMPLE 2 (Mahalanobis distance)
    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    # model_2 = eQFCM(n_clusters=3, merge=False, random_state=42, distance_metric='mahalanobis')
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

    model_3 = eQFCM(n_clusters=2, merge=False, random_state=100, distance_metric='mahalanobis', epsilon_1=0.00005)
    model_3.fit(X_3)
    plot_2(model_3, X_3)