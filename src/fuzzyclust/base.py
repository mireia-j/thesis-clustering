"""
Base interfaces for clustering algorithms in this package.

Class hierarchy
---------------
    BaseClusterer (ABC)
    ├── KMeans (planned)
    └── FuzzyClusterer (ABC)
        ├── FCM
        ├── GK
        └── ExtendedFuzzyClusterer (planned, for E-FCM and E-GK)

``BaseClusterer`` defines the universal API (``fit``, ``predict``,
``transform``, ``fit_predict``) that works for any clustering algorithm,
crisp or fuzzy.

``FuzzyClusterer`` adds the shared machinery for objective-function-based
fuzzy clustering: the alternating optimization loop, the membership update
from a distance matrix (Bezdek 1981 eq. 2), and the weighted center update
(eq. 1). Subclasses only need to implement ``_squared_distances`` and any
per-iteration bookkeeping (e.g. covariance update for GK).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Numerical floor on distances to avoid 1/0 when a point coincides
# with a center. Shared across all fuzzy algorithms.
_EPS = 1e-10


@dataclass
class FitHistory:
    """Per-iteration diagnostics recorded during ``fit``.

    Useful for plotting convergence curves and debugging the merging step
    in the extended algorithms.
    """

    objective: list[float] = field(default_factory=list)
    n_clusters: list[int] = field(default_factory=list)
    max_u_change: list[float] = field(default_factory=list)
    merges: list[tuple[int, int, int]] = field(default_factory=list)
    # Each merge entry is (iteration, cluster_i, cluster_j).


# ====================================================================== #
# BaseClusterer: universal API                                            #
# ====================================================================== #


class BaseClusterer(ABC):
    """Abstract base class for all clustering algorithms.

    Defines the public API. Subclasses implement ``fit`` and
    ``_compute_memberships``; everything else follows.

    Parameters
    ----------
    n_clusters : int
        Number of clusters. For algorithms with merging (E-FCM, E-GK)
        this is the *initial* number; the final number is
        ``self.n_clusters_``.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-3
        Termination threshold on ``||U^(l) - U^(l-1)||_F``.
    random_state : int or None, default=None
        Seed for reproducible initialization.

    Attributes
    ----------
    centers_ : ndarray of shape (n_clusters_, n_features)
    labels_ : ndarray of shape (n_samples,)
        Hard cluster assignment (argmax of memberships).
    memberships_ : ndarray of shape (n_clusters_, n_samples)
        For crisp algorithms this is one-hot.
    n_clusters_ : int
    n_iter_ : int
    history_ : FitHistory
    """

    name: str = "BaseClusterer"

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
    ):
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.memberships_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.n_iter_: Optional[int] = None
        self.history_: Optional[FitHistory] = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """Run the clustering on ``X`` and populate the ``*_`` attributes."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard cluster labels for ``X``."""
        self._check_fitted()
        U = self.transform(X)
        return np.argmax(U, axis=0)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience: ``fit`` then return ``labels_``."""
        return self.fit(X).labels_  # type: ignore[return-value]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute the membership matrix for new data ``X``."""
        self._check_fitted()
        return self._compute_memberships(np.asarray(X, dtype=float))

    # ------------------------------------------------------------------ #
    # Hooks for subclasses                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _compute_memberships(self, X: np.ndarray) -> np.ndarray:
        """Compute U for new points using the fitted model."""

    # ------------------------------------------------------------------ #
    # Optional attributes (overridden where relevant)                     #
    # ------------------------------------------------------------------ #

    @property
    def covariances_(self) -> Optional[np.ndarray]:
        """Cluster covariance matrices (GK, E-GK). ``None`` otherwise."""
        return None

    @property
    def radii_(self) -> Optional[np.ndarray]:
        """Volume-prototype radii (E-FCM, E-GK). ``None`` otherwise."""
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if self.centers_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call fit(X) first."
            )

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            )
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples ({X.shape[0]}) must be >= n_clusters "
                f"({self.n_clusters})"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        return X

    def _init_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    def get_params(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }


# ====================================================================== #
# FuzzyClusterer: shared logic for objective-function fuzzy clustering    #
# ====================================================================== #


class FuzzyClusterer(BaseClusterer):
    """Base for objective-function-based fuzzy clustering (FCM, GK, ...).

    Implements the standard alternating-optimization loop:

        1. Update centers from memberships (weighted means)         eq. (1)
        2. Update per-iteration state (e.g. covariances for GK)     [hook]
        3. Compute squared distances (subclass-specific)
        4. Update memberships from distances                        eq. (2)
        5. Check convergence

    Subclasses customize the distance metric and per-iteration state.
    They also choose their initialization strategy.

    Parameters
    ----------
    n_clusters : int
        Number of fuzzy clusters ``c``.
    m : float, default=2.0
        Fuzziness exponent (>1).
    max_iter, tol, random_state
        See ``BaseClusterer``.
    init : str, default='random_memberships'
        Initialization strategy. Concrete subclasses define accepted values.
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        init: str = "random_memberships",
    ):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        if m <= 1.0:
            raise ValueError(f"m must be > 1, got {m}")
        self._validate_init(init)
        self.m = m
        self.init = init
        self.objective_: Optional[float] = None

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({"m": self.m, "init": self.init})
        return params

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _squared_distances(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Return (c, N) squared distances d_{ik}^2 from points to centers.

        FCM uses Euclidean; GK uses (x-v)^T A_i (x-v) with cluster-specific A_i.
        """

    def _post_center_update(self, X: np.ndarray, U: np.ndarray) -> None:
        """Hook called after each center update, before distance computation.

        GK uses this to refresh per-cluster covariance matrices. FCM is
        a no-op. Default: do nothing.
        """

    @abstractmethod
    def _initialize(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return an initial ``(U, V)`` pair for the optimization."""

    def _validate_init(self, init: str) -> None:
        """Raise if ``init`` is not a value this subclass supports.

        Default: accept anything. Concrete subclasses override to be strict.
        """
        return

    # ------------------------------------------------------------------ #
    # Shared optimization loop                                            #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "FuzzyClusterer":
        X = self._validate_data(X)
        rng = self._init_rng()
        U, V = self._initialize(X, rng)

        # State (e.g. covariances) may need to be set before the first
        # distance computation.
        self._post_center_update(X, U)

        history = FitHistory()
        prev_U = U.copy()
        it = 0
        J = float("inf")

        for it in range(1, self.max_iter + 1):
            V = self._update_centers(X, U)
            self._post_center_update(X, U)
            D2 = self._squared_distances(X, V)
            U = self._memberships_from_distances(D2)

            J = float(np.sum((U ** self.m) * D2))
            change = float(np.linalg.norm(U - prev_U))

            history.objective.append(J)
            history.n_clusters.append(self.n_clusters)
            history.max_u_change.append(change)

            if change < self.tol:
                break
            prev_U = U.copy()

        self.centers_ = V
        self.memberships_ = U
        self.labels_ = np.argmax(U, axis=0)
        self.n_clusters_ = self.n_clusters
        self.n_iter_ = it
        self.objective_ = J
        self.history_ = history
        return self

    # ------------------------------------------------------------------ #
    # Shared update equations                                             #
    # ------------------------------------------------------------------ #

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Eq. (1): v_i = sum_k u_{ik}^m x_k / sum_k u_{ik}^m."""
        Um = U ** self.m  # (c, N)
        denom = np.maximum(Um.sum(axis=1, keepdims=True), _EPS)
        return (Um @ X) / denom

    def _memberships_from_distances(self, D2: np.ndarray) -> np.ndarray:
        """Eq. (2): u_{ik} = 1 / sum_j (d_{ik}^2 / d_{jk}^2)^(1/(m-1)).

        Handles the zero-distance edge case: a point that coincides with
        one or more centers splits its membership equally among them.
        """
        c, N = D2.shape
        U = np.zeros((c, N), dtype=float)

        zero_mask = D2 < _EPS  # (c, N)
        coincides = zero_mask.any(axis=0)  # (N,)

        good = ~coincides
        if good.any():
            D2_good = D2[:, good]
            exponent = 1.0 / (self.m - 1.0)
            inv = D2_good ** (-exponent)
            U[:, good] = inv / inv.sum(axis=0, keepdims=True)

        if coincides.any():
            for k in np.where(coincides)[0]:
                hits = zero_mask[:, k]
                U[hits, k] = 1.0 / hits.sum()

        return U

    # ------------------------------------------------------------------ #
    # Initialization helpers shared by subclasses                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _random_memberships(
        c: int, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample a random valid U from a Dirichlet(1, ..., 1)."""
        return rng.dirichlet(alpha=np.ones(c), size=n_samples).T

    @staticmethod
    def _kmeanspp_centers(
        X: np.ndarray, c: int, rng: np.random.Generator
    ) -> np.ndarray:
        """k-means++ seeding for initial centers."""
        n_samples = X.shape[0]
        first = rng.integers(n_samples)
        centers = [X[first]]
        d2 = np.sum((X - centers[0]) ** 2, axis=1)
        for _ in range(1, c):
            probs = d2 / d2.sum()
            idx = rng.choice(n_samples, p=probs)
            centers.append(X[idx])
            new_d2 = np.sum((X - centers[-1]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)
        return np.asarray(centers)

    # ------------------------------------------------------------------ #
    # transform / predict                                                 #
    # ------------------------------------------------------------------ #

    def _compute_memberships(self, X: np.ndarray) -> np.ndarray:
        """Memberships for new points, using the fitted model."""
        D2 = self._squared_distances(X, self.centers_)
        return self._memberships_from_distances(D2)
