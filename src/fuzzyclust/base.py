import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class BaseFuzzyClustering:

    def __init__(self, n_clusters=10, m=2, max_iter=150, error=1e-5,
                 random_state=42):
        self.n_clusters = n_clusters
        self.m = m                  # fuzzifier (>1)
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

        self.u = None               # membership matrix, shape (N, C)
        self.centers = None         # cluster centers, shape (C, n_features)


    def fit(self, X):
        pass

    def _init_membership(self, n_samples):
        rng = np.random.RandomState(self.random_state)
        u = rng.rand(n_samples, self.n_clusters)
        return u / u.sum(axis=1, keepdims=True)

    def next_centers(self, X, u): 
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X):
        """Fuzzy membership of each row of X to each cluster."""
        return self._predict(X)

    def _predict(self, X):
        power = float(1 / (self.m - 1))
        dist = cdist(X, self.centers) ** power
        denominator_ = dist.reshape((X.shape[0], 1, -1)).repeat(dist.shape[-1], axis=1)
        denominator_ = dist[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)
    
        # denom = dist[:, :, np.newaxis] / dist[:, np.newaxis, :]
        # return 1.0 / denom.sum(axis=2)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X)
        return np.argmax(u, axis=-1)
    
    def labels(self):
        """Hard labels for the data passed to fit()"""
        if self.u is None:
            raise ValueError("Not fitted yet. Call fit() first.")
        return np.argmax(self.u, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels()
    


def plot_2(result_fuzzy, X, show_centers=True):
    """Fit on X (must have exactly 2 features) and scatter the points
    coloured by their assigned cluster."""
    if result_fuzzy.centers.shape[1] != 2:
        raise ValueError(
            f"plot() needs 2-D data of shape (n_samples, 2); got {X.shape}."
        )
     
    if result_fuzzy.centers is None:
        raise ValueError("Not fitted yet. Call fit() first.")

    labels = result_fuzzy.labels()        
    if len(labels) != X.shape[0]:
        raise ValueError("Number of samples in X does not match number labels.")

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15, alpha=0.7)

    if show_centers:
        ax.scatter(result_fuzzy.centers[:, 0], result_fuzzy.centers[:, 1],
                    c="red", marker="X", s=200, edgecolors="black",
                    label="centers")
        ax.legend()

    ax.set_title(f"{type(result_fuzzy).__name__} — {result_fuzzy.n_clusters} clusters")
    plt.show()
    return ax


