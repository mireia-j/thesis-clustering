"""
Fuzzy c-means (FCM) clustering.

Reference
---------
J. C. Bezdek, *Pattern Recognition with Fuzzy Objective Function Algorithms*,
Plenum, 1981.

Minimizes

    J_m(U, V) = sum_{i=1..c} sum_{k=1..N} u_{ik}^m * ||x_k - v_i||^2

subject to sum_i u_{ik} = 1, via alternating optimization. FCM only
differs from generic fuzzy clustering in (a) using Euclidean distance,
(b) not maintaining any per-cluster state between iterations. So almost
all logic lives in ``FuzzyClusterer``; this class just plugs in those two
choices.
"""

from __future__ import annotations

import numpy as np

from fuzzyclust.base import FuzzyClusterer


class FCM(FuzzyClusterer):
    """Fuzzy c-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters ``c``.
    m : float, default=2.0
        Fuzziness exponent. ``m -> 1`` gives crisp clustering;
        ``m -> infinity`` gives uniform memberships. ``m = 2`` is standard.
    max_iter : int, default=100
    tol : float, default=1e-3
    random_state : int or None, default=None
    init : {'random_memberships', 'kmeans++'}, default='random_memberships'
        ``random_memberships``: standard Bezdek FCM initialization with a
        Dirichlet-sampled U. ``kmeans++``: seed centers with the k-means++
        heuristic and derive U from them; usually converges faster.

    Attributes
    ----------
    centers_ : ndarray of shape (n_clusters, n_features)
    memberships_ : ndarray of shape (n_clusters, n_samples)
    labels_ : ndarray of shape (n_samples,)
    n_iter_ : int
    objective_ : float
    history_ : FitHistory
    """

    name = "FCM"

    _VALID_INITS = {"random_memberships", "kmeans++"}

    def _validate_init(self, init: str) -> None:
        if init not in self._VALID_INITS:
            raise ValueError(
                f"Unknown init={init!r}. Must be one of {self._VALID_INITS}."
            )

    # ------------------------------------------------------------------ #
    # Distance metric: Euclidean                                          #
    # ------------------------------------------------------------------ #

    def _squared_distances(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """||x_k - v_i||^2, broadcast-style.

        Returns shape (c, N).
        """
        XX = np.sum(X * X, axis=1)[None, :]    # (1, N)
        VV = np.sum(V * V, axis=1)[:, None]    # (c, 1)
        cross = V @ X.T                        # (c, N)
        D2 = VV + XX - 2.0 * cross
        return np.maximum(D2, 0.0)

    # ------------------------------------------------------------------ #
    # Initialization                                                      #
    # ------------------------------------------------------------------ #

    def _initialize(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        c = self.n_clusters

        if self.init == "random_memberships":
            U = self._random_memberships(c, n_samples, rng)
            V = self._update_centers(X, U)
        else:  # 'kmeans++'
            V = self._kmeanspp_centers(X, c, rng)
            D2 = self._squared_distances(X, V)
            U = self._memberships_from_distances(D2)

        return U, V
