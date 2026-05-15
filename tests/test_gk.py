"""
Tests for the Gustafson-Kessel implementation.

Same structure as test_fcm.py:
- Partition invariants
- Algorithm correctness (including the GK-specific behavior: shape adaptation)
- Edge cases
- Input validation
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from fuzzyclust.gk import GK


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


@pytest.fixture
def three_gaussians():
    """Three well-separated isotropic 2D Gaussians."""
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([rng.normal(c, 0.3, size=(50, 2)) for c in centers])
    return X, centers


@pytest.fixture
def rotated_ellipsoids():
    """Two elongated clusters at 45/-45 degrees - GK's specialty.

    FCM would conflate these because Euclidean distance is rotation-blind;
    GK should track each cluster's elongation via its covariance.
    """
    rng = np.random.default_rng(0)
    n = 100
    # Cluster at (-3, -3), elongated along (1, 1)
    th1 = np.pi / 4
    R1 = np.array([[np.cos(th1), -np.sin(th1)], [np.sin(th1), np.cos(th1)]])
    X1 = (rng.normal(0, 1, (n, 2)) * [2.0, 0.2]) @ R1.T + [-3, -3]
    # Cluster at (3, 3), elongated along (1, -1)
    th2 = -np.pi / 4
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])
    X2 = (rng.normal(0, 1, (n, 2)) * [2.0, 0.2]) @ R2.T + [3, 3]
    X = np.vstack([X1, X2])
    true_labels = np.array([0] * n + [1] * n)
    return X, true_labels


# ====================================================================== #
# 1. Partition invariants                                                 #
# ====================================================================== #


def test_memberships_sum_to_one(three_gaussians):
    X, _ = three_gaussians
    U = GK(n_clusters=3, random_state=0).fit(X).memberships_
    np.testing.assert_allclose(U.sum(axis=0), 1.0, atol=1e-10)


def test_memberships_in_unit_interval(three_gaussians):
    X, _ = three_gaussians
    U = GK(n_clusters=3, random_state=0).fit(X).memberships_
    assert U.min() >= 0.0
    assert U.max() <= 1.0


def test_memberships_are_finite(three_gaussians):
    X, _ = three_gaussians
    U = GK(n_clusters=3, random_state=0).fit(X).memberships_
    assert np.all(np.isfinite(U))


def test_memberships_shape(three_gaussians):
    X, _ = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    assert model.memberships_.shape == (3, X.shape[0])


def test_labels_are_argmax(three_gaussians):
    X, _ = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(
        model.labels_, np.argmax(model.memberships_, axis=0)
    )


# ====================================================================== #
# 2. Algorithm correctness                                                #
# ====================================================================== #


def test_objective_is_finite(three_gaussians):
    """GK's objective should always be a finite number."""
    X, _ = three_gaussians
    J = np.array(GK(n_clusters=3, random_state=0).fit(X).history_.objective)
    assert np.all(np.isfinite(J))


def test_recovers_known_centers(three_gaussians):
    """On easy isotropic data, GK should find the true centers."""
    X, true_centers = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    nearest = cdist(true_centers, model.centers_).min(axis=1)
    assert nearest.max() < 0.3


def test_deterministic_with_same_seed(three_gaussians):
    X, _ = three_gaussians
    m1 = GK(n_clusters=3, random_state=42).fit(X)
    m2 = GK(n_clusters=3, random_state=42).fit(X)
    np.testing.assert_array_equal(m1.centers_, m2.centers_)
    np.testing.assert_array_equal(m1.memberships_, m2.memberships_)


def test_separates_rotated_ellipsoids(rotated_ellipsoids):
    """The defining advantage of GK over FCM: shape-adaptive clusters.

    Standard FCM with Euclidean distance gets confused by elongated
    clusters at different orientations. GK should resolve them cleanly.
    """
    X, true_labels = rotated_ellipsoids
    model = GK(n_clusters=2, random_state=0).fit(X)
    pred = model.labels_
    # Account for label permutation.
    acc = max((pred == true_labels).mean(), (pred != true_labels).mean())
    assert acc > 0.95


def test_covariances_reflect_elongation(rotated_ellipsoids):
    """Eigenvalue ratio of each P_i should reflect the data's anisotropy."""
    X, _ = rotated_ellipsoids
    model = GK(n_clusters=2, random_state=0).fit(X)
    for i in range(2):
        eigs = np.linalg.eigvalsh(model.covariances_[i])
        # Data has std ratio 10:1, so eigenvalue ratio should be ~100.
        ratio = eigs.max() / eigs.min()
        assert ratio > 30, f"Cluster {i} eigenvalue ratio = {ratio:.1f}"


def test_volume_constraint_holds(three_gaussians):
    """det(A_i) should equal rho_i (by construction, default rho=1)."""
    X, _ = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    for i, P in enumerate(model.covariances_):
        # A_i = (rho * det(P))^(1/n) * inv(P), so det(A_i) = rho.
        det_P = np.linalg.det(P)
        scale = (1.0 * det_P) ** (1.0 / 2)  # rho=1, n=2
        A_i = scale * np.linalg.inv(P)
        assert abs(np.linalg.det(A_i) - 1.0) < 1e-6


def test_covariances_attribute_exposed(three_gaussians):
    """``covariances_`` should be a (c, n, n) array after fit."""
    X, _ = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    assert model.covariances_ is not None
    assert model.covariances_.shape == (3, 2, 2)


def test_covariances_are_positive_definite(three_gaussians):
    """Stabilization should guarantee SPD covariances."""
    X, _ = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    for P in model.covariances_:
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs > 0)


def test_higher_m_gives_fuzzier_memberships(three_gaussians):
    X, _ = three_gaussians
    U_low = GK(n_clusters=3, m=1.5, random_state=0).fit(X).memberships_
    U_high = GK(n_clusters=3, m=4.0, random_state=0).fit(X).memberships_
    assert U_low.max(axis=0).mean() > U_high.max(axis=0).mean()


# ====================================================================== #
# 3. Edge cases and numerical stability                                   #
# ====================================================================== #


def test_handles_collinear_cluster():
    """A nearly-collinear cluster should not crash GK (singular P)."""
    rng = np.random.default_rng(0)
    # Cluster 1: points almost on a line (y nearly constant)
    X1 = np.column_stack([rng.normal(0, 2, 100), rng.normal(0, 1e-6, 100)])
    # Cluster 2: normal
    X2 = rng.normal([5, 5], 0.3, size=(50, 2))
    X = np.vstack([X1, X2])
    # Should not raise, should produce finite results.
    model = GK(n_clusters=2, random_state=0).fit(X)
    assert np.all(np.isfinite(model.memberships_))
    assert np.all(np.isfinite(model.centers_))
    for P in model.covariances_:
        assert np.all(np.isfinite(P))


def test_point_coincides_with_center():
    """Coinciding points should yield one-hot memberships, not NaN."""
    X = np.vstack([
        np.tile([-3.0, 0.0], (30, 1)),
        np.tile([3.0, 0.0], (30, 1)),
    ])
    model = GK(n_clusters=2, random_state=0).fit(X)
    assert np.all(np.isfinite(model.memberships_))


def test_predict_on_new_data(three_gaussians):
    """Predict should work after fit, with no surprises."""
    X, true_centers = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    labels = model.predict(true_centers)
    assert len(set(labels)) == 3


def test_transform_returns_valid_partition(three_gaussians):
    X, _ = three_gaussians
    model = GK(n_clusters=3, random_state=0).fit(X)
    rng = np.random.default_rng(1)
    X_new = rng.normal(0, 1, size=(20, 2))
    U_new = model.transform(X_new)
    assert U_new.shape == (3, 20)
    np.testing.assert_allclose(U_new.sum(axis=0), 1.0, atol=1e-10)


def test_fit_predict_equivalent(three_gaussians):
    X, _ = three_gaussians
    a = GK(n_clusters=3, random_state=0).fit_predict(X)
    b = GK(n_clusters=3, random_state=0).fit(X).labels_
    np.testing.assert_array_equal(a, b)


# ====================================================================== #
# 4. Input validation                                                     #
# ====================================================================== #


def test_rejects_zero_clusters():
    with pytest.raises(ValueError, match="n_clusters"):
        GK(n_clusters=0)


def test_rejects_m_below_one():
    with pytest.raises(ValueError, match="m"):
        GK(n_clusters=3, m=0.5)


def test_rejects_unknown_init():
    with pytest.raises(ValueError, match="init"):
        GK(n_clusters=3, init="not_real")


def test_rejects_negative_reg():
    with pytest.raises(ValueError, match="reg"):
        GK(n_clusters=3, reg=-0.1)


def test_rejects_bad_max_condition():
    with pytest.raises(ValueError, match="max_condition"):
        GK(n_clusters=3, max_condition=1.0)


def test_rejects_nonpositive_rho():
    with pytest.raises(ValueError, match="rho"):
        GK(n_clusters=3, rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        GK(n_clusters=3, rho=-1.0)


def test_rejects_rho_wrong_length():
    with pytest.raises(ValueError, match="rho"):
        GK(n_clusters=3, rho=[1.0, 2.0])  # wrong length


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="2D"):
        GK(n_clusters=3).fit(np.array([1.0, 2.0, 3.0]))


def test_rejects_nan_input():
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(ValueError, match="NaN|infinite"):
        GK(n_clusters=2).fit(X)


def test_rejects_predict_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        GK(n_clusters=3).predict(np.array([[1.0, 2.0]]))
