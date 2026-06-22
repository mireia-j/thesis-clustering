"""Base class for fuzzy clustering algorithms, with Fuzzy C-Means as the
first concrete implementation.

Design idea
-----------
Every fuzzy clustering algorithm ends up producing a *membership matrix*
`u` of shape (n_samples, n_clusters). Once that exists, the useful
operations are the same regardless of how `u` was computed:

    - hard labels        -> argmax(u, axis=1)
    - predict new points -> membership w.r.t. the cluster centers

What differs between algorithms is *how* `u` is produced. FCM does it with
an iterative centers <-> memberships update; agglomerative hierarchical
clustering does it by merging from the bottom up and cutting the tree.

So the base class owns the shared state + behaviour and exposes a single
abstract hook, `fit`, that each algorithm implements its own way.
"""

import numpy as np
from scipy.spatial.distance import cdist

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

    def next_centers(self, X, u): # CHECK
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X):
        """Fuzzy membership of each row of X to each cluster."""
        return self._predict(X)

    def _predict(self, X):
        power = float(2 / (self.m - 1))
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
        return self.labels


