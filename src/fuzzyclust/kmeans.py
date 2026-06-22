"""
K-means clustering.

Reference
---------
S. Lloyd, "Least squares quantization in PCM," IEEE Trans. Inf. Theory,
vol. 28, no. 2, pp. 129-137, 1982.

Thin wrapper around ``sklearn.cluster.KMeans`` to fit this package's
``BaseClusterer`` API. This means notebooks can iterate over k-means,
FCM, and GK uniformly without special-casing the crisp algorithm.

Why wrap rather than reimplement?
---------------------------------
sklearn's implementation uses Elkan's algorithm (faster than naive Lloyd
on most data), handles edge cases robustly, and is the de-facto reference.
Reimplementing it would mean a slower, less-tested baseline — unfair to
k-means in any performance comparison. The thesis still cites Lloyd 1982
and explains the algorithm; the implementation is just plumbing.

Differences from FCM/GK
-----------------------
* K-means is *crisp*: each point belongs to exactly one cluster.
* ``memberships_`` returns a one-hot encoding of ``labels_`` so the API
  matches FCM/GK. Notebooks can call ``model.memberships_`` uniformly.
* There is no fuzziness parameter ``m`` and no fuzzy objective.
* The objective stored in ``inertia_`` is the within-cluster sum of
  squared distances (sklearn's terminology).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans as _SKLearnKMeans

from fuzzyclust.base import BaseClusterer, FitHistory


class KMeans(BaseClusterer):
    """K-means clustering (Lloyd's algorithm, via sklearn).

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default=300
        Maximum number of Lloyd iterations. Default matches sklearn's
        default (higher than FCM's because k-means iterations are cheap).
    tol : float, default=1e-4
        Convergence threshold on the Frobenius norm of the change in
        cluster centers between iterations. Matches sklearn's default.
    random_state : int or None, default=None
    init : {'kmeans++', 'random'}, default='kmeans++'
        Center initialization. ``kmeans++`` (Arthur & Vassilvitskii 2007)
        is almost always the better choice; ``random`` is included for
        comparison studies.
    n_init : int, default=10
        Number of independent runs with different random seeds. The best
        run (lowest inertia) is kept. K-means is highly sensitive to
        initialization, so multiple restarts are standard practice.

    Attributes
    ----------
    centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    memberships_ : ndarray of shape (n_clusters, n_samples)
        One-hot encoding of ``labels_``. Provided for API compatibility
        with the fuzzy algorithms.
    n_iter_ : int
        Iterations of the best run (not summed across restarts).
    inertia_ : float
        Final within-cluster sum of squared distances.
    history_ : FitHistory
        Only the final inertia is recorded (sklearn doesn't expose
        per-iteration history). The list has length 1.
    """

    name = "KMeans"

    _VALID_INITS = {"kmeans++", "random"}

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init: str = "kmeans++",
        n_init: int = 10,
    ):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        if init not in self._VALID_INITS:
            raise ValueError(
                f"Unknown init={init!r}. Must be one of {self._VALID_INITS}."
            )
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")

        self.init = init
        self.n_init = n_init
        self.inertia_: Optional[float] = None

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({"init": self.init, "n_init": self.n_init})
        return params

    # ------------------------------------------------------------------ #
    # Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "KMeans":
        X = self._validate_data(X)

        # Translate our 'kmeans++' to sklearn's 'k-means++'.
        sklearn_init = "k-means++" if self.init == "kmeans++" else "random"

        backend = _SKLearnKMeans(
            n_clusters=self.n_clusters,
            init=sklearn_init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        backend.fit(X)

        # Store results in our API conventions.
        self.centers_ = backend.cluster_centers_
        self.labels_ = backend.labels_.astype(int)
        self.memberships_ = self._labels_to_onehot(self.labels_, self.n_clusters)
        self.n_clusters_ = self.n_clusters
        self.n_iter_ = int(backend.n_iter_)
        self.inertia_ = float(backend.inertia_)

        # K-means doesn't expose per-iteration history through sklearn,
        # so we just record the final state.
        self.history_ = FitHistory()
        self.history_.objective.append(self.inertia_)
        self.history_.n_clusters.append(self.n_clusters)
        self.history_.max_u_change.append(0.0)

        return self

    # ------------------------------------------------------------------ #
    # Transform / predict                                                 #
    # ------------------------------------------------------------------ #

    def _compute_memberships(self, X: np.ndarray) -> np.ndarray:
        """One-hot memberships for new points (nearest center)."""
        labels = self._nearest_center(X, self.centers_)
        return self._labels_to_onehot(labels, self.n_clusters)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Override the base predict to skip the unnecessary one-hot round-trip."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._nearest_center(X, self.centers_)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _nearest_center(X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Argmin over centers of squared Euclidean distance."""
        XX = np.sum(X * X, axis=1)[:, None]    # (N, 1)
        VV = np.sum(V * V, axis=1)[None, :]    # (1, c)
        cross = X @ V.T                        # (N, c)
        D2 = XX + VV - 2.0 * cross
        return np.argmin(D2, axis=1).astype(int)

    @staticmethod
    def _labels_to_onehot(labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """One-hot encode labels as a (c, N) membership matrix."""
        N = labels.shape[0]
        U = np.zeros((n_clusters, N), dtype=float)
        U[labels, np.arange(N)] = 1.0
        return U
