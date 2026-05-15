"""
Gustafson-Kessel (GK) clustering.

Reference
---------
D. E. Gustafson and W. C. Kessel, "Fuzzy clustering with a fuzzy
covariance matrix," in Proc. IEEE Conf. Decision Control, 1979, pp. 761-766.

GK extends FCM by adapting the distance metric to each cluster's shape
via the (fuzzy) cluster covariance matrix P_i. The distance is

    d^2(x_k, v_i) = (x_k - v_i)^T A_i (x_k - v_i),
    A_i = (rho_i * det(P_i))^(1/n) * inv(P_i)            (eq. 12 in paper)

where rho_i (typically 1) imposes a volume constraint det(A_i) = rho_i
that prevents the metric from degenerating to a point.

Per iteration:
    1. Update centers V from U                                  (eq. 1)
    2. Update covariances P_i from U and V                      (eq. 2)
    3. Compute distances using A_i
    4. Update U from distances                                  (eq. 2)

Numerical considerations
------------------------
Two well-known failure modes are handled here:

* **Singular P_i** when a cluster has very few effective members. Fixed
  by adding ``reg * I`` (Tikhonov-style regularization) before inversion.
* **Eigenvalue collapse**: when one eigenvalue of P_i becomes much smaller
  than the largest, the cluster grows arbitrarily elongated. Fixed by
  bounding the condition number: small eigenvalues are clipped up to
  ``lambda_max / max_condition``.

These are standard tricks; see e.g. Babuska et al. (2002) "Improved
covariance estimation for Gustafson-Kessel clustering".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fuzzyclust.base import FuzzyClusterer, _EPS


class GK(FuzzyClusterer):
    """Gustafson-Kessel fuzzy clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    m : float, default=2.0
        Fuzziness exponent (>1).
    max_iter : int, default=100
    tol : float, default=1e-3
    random_state : int or None, default=None
    init : {'random_memberships', 'kmeans++'}, default='kmeans++'
        See ``FCM``. Default differs from FCM: GK benefits from a
        reasonable initial geometry so kmeans++ is recommended.
    rho : float or array-like of shape (n_clusters,), default=1.0
        Volume constraint det(A_i) = rho_i. A scalar applies to all
        clusters. The standard choice rho_i = 1 makes the metric
        volume-preserving.
    reg : float, default=1e-6
        Tikhonov regularization added to each covariance matrix before
        inversion, as a fraction of trace(P_i)/n. Prevents singularities.
    max_condition : float, default=1e15
        Maximum allowed condition number of P_i. Eigenvalues smaller than
        ``lambda_max / max_condition`` are clipped up. Prevents eigenvalue
        collapse. Set to ``np.inf`` to disable.

    Attributes
    ----------
    centers_ : ndarray of shape (n_clusters, n_features)
    memberships_ : ndarray of shape (n_clusters, n_samples)
    labels_ : ndarray of shape (n_samples,)
    covariances_ : ndarray of shape (n_clusters, n_features, n_features)
        Final cluster covariance matrices P_i.
    n_iter_ : int
    objective_ : float
    history_ : FitHistory
    """

    name = "GK"

    _VALID_INITS = {"random_memberships", "kmeans++"}

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        init: str = "kmeans++",
        rho=1.0,
        reg: float = 1e-6,
        max_condition: float = 1e15,
    ):
        super().__init__(
            n_clusters=n_clusters,
            m=m,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            init=init,
        )
        if reg < 0:
            raise ValueError(f"reg must be >= 0, got {reg}")
        if max_condition <= 1:
            raise ValueError(f"max_condition must be > 1, got {max_condition}")

        # rho may be a scalar (broadcast to all clusters) or an array.
        rho_arr = np.atleast_1d(np.asarray(rho, dtype=float))
        if rho_arr.size == 1:
            rho_arr = np.full(n_clusters, float(rho_arr[0]))
        elif rho_arr.size != n_clusters:
            raise ValueError(
                f"rho must be scalar or length n_clusters ({n_clusters}), "
                f"got shape {rho_arr.shape}"
            )
        if np.any(rho_arr <= 0):
            raise ValueError(f"rho must be positive, got {rho_arr}")

        self.rho = rho_arr
        self.reg = reg
        self.max_condition = max_condition

        # Per-iteration state: covariances P_i (set by _post_center_update).
        self._covariances: Optional[np.ndarray] = None
        # Cached norm matrices A_i, recomputed when P_i changes.
        self._A: Optional[np.ndarray] = None

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "rho": self.rho.tolist() if self.rho.size > 1 else float(self.rho[0]),
            "reg": self.reg,
            "max_condition": self.max_condition,
        })
        return params

    def _validate_init(self, init: str) -> None:
        if init not in self._VALID_INITS:
            raise ValueError(
                f"Unknown init={init!r}. Must be one of {self._VALID_INITS}."
            )

    # ------------------------------------------------------------------ #
    # Covariance update (the heart of GK)                                 #
    # ------------------------------------------------------------------ #

    def _post_center_update(self, X: np.ndarray, U: np.ndarray) -> None:
        """Refresh the per-cluster covariance P_i and the norm matrix A_i.

        Implements equation (2) of the paper with regularization and
        condition-number bounding for numerical stability.
        """
        c, N = U.shape
        n = X.shape[1]

        # If centers aren't computed yet (first call before fit loop),
        # compute them from U so we can build covariances.
        if self.centers_ is None:
            V = self._update_centers(X, U)
        else:
            V = self.centers_

        Um = U ** self.m                          # (c, N)
        denom = np.maximum(Um.sum(axis=1), _EPS)  # (c,)

        P = np.empty((c, n, n))
        for i in range(c):
            diff = X - V[i]                       # (N, n)
            weighted = (Um[i][:, None] * diff)    # (N, n)
            P[i] = (weighted.T @ diff) / denom[i] # (n, n)
            P[i] = self._stabilize(P[i])

        self._covariances = P
        self._A = self._compute_norm_matrices(P)

    def _stabilize(self, P_i: np.ndarray) -> np.ndarray:
        """Regularize and clip eigenvalues to avoid singularity / collapse."""
        n = P_i.shape[0]

        # Symmetrize (defensive: numerical drift can break symmetry).
        P_i = 0.5 * (P_i + P_i.T)

        # Tikhonov regularization, scaled to the matrix magnitude.
        trace = np.trace(P_i)
        if trace > 0:
            P_i = P_i + (self.reg * trace / n) * np.eye(n)
        else:
            # Cluster has effectively no spread; fall back to identity.
            P_i = P_i + self.reg * np.eye(n)

        # Bound the condition number.
        if np.isfinite(self.max_condition):
            eigvals, eigvecs = np.linalg.eigh(P_i)
            lam_max = eigvals.max()
            lam_min_allowed = lam_max / self.max_condition
            eigvals = np.maximum(eigvals, lam_min_allowed)
            P_i = (eigvecs * eigvals) @ eigvecs.T

        return P_i

    def _compute_norm_matrices(self, P: np.ndarray) -> np.ndarray:
        """A_i = (rho_i * det(P_i))^(1/n) * inv(P_i)  (eq. 12)."""
        c, n, _ = P.shape
        A = np.empty_like(P)
        for i in range(c):
            det_P = np.linalg.det(P[i])
            # det_P should be positive after stabilization, but guard anyway.
            det_P = max(det_P, _EPS)
            scale = (self.rho[i] * det_P) ** (1.0 / n)
            A[i] = scale * np.linalg.inv(P[i])
        return A

    # ------------------------------------------------------------------ #
    # Distance metric                                                     #
    # ------------------------------------------------------------------ #

    def _squared_distances(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """d^2(x_k, v_i) = (x_k - v_i)^T A_i (x_k - v_i)."""
        c = V.shape[0]
        N = X.shape[0]
        D2 = np.empty((c, N))

        # During transform() on new data, _A reflects the *fitted* state.
        # During fit(), _post_center_update keeps _A in sync.
        A = self._A
        for i in range(c):
            diff = X - V[i]                       # (N, n)
            # (N, n) @ (n, n) -> (N, n); then sum elementwise products.
            D2[i] = np.einsum("ni,ij,nj->n", diff, A[i], diff)

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
            # We need an initial U before we can build covariances. Use a
            # Euclidean-distance soft assignment as a warm start.
            XX = np.sum(X * X, axis=1)[None, :]
            VV = np.sum(V * V, axis=1)[:, None]
            D2 = np.maximum(VV + XX - 2.0 * (V @ X.T), 0.0)
            U = self._memberships_from_distances(D2)

        return U, V

    # ------------------------------------------------------------------ #
    # Public read of fitted covariances                                   #
    # ------------------------------------------------------------------ #

    @property
    def covariances_(self) -> Optional[np.ndarray]:
        return self._covariances
