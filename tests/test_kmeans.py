"""
Tests for the KMeans wrapper.

K-means is a thin wrapper around sklearn, so we focus on:
- API conformance with BaseClusterer (so notebooks work uniformly)
- The one-hot membership encoding (the key adapter trick)
- Determinism with seeds
- Input validation
- Sanity checks on correctness, not exhaustive coverage (sklearn already
  tests its own implementation)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from fuzzyclust.kmeans import KMeans


# ====================================================================== #
# Fixtures                                                                #
# ====================================================================== #


@pytest.fixture
def three_gaussians():
    """Three well-separated 2D Gaussian clusters with known centers."""
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([rng.normal(c, 0.3, size=(50, 2)) for c in centers])
    return X, centers


# ====================================================================== #
# 1. API conformance (the key reason for wrapping)                        #
# ====================================================================== #


def test_memberships_are_one_hot(three_gaussians):
    """K-means memberships should be exactly 0 or 1, never fractional."""
    X, _ = three_gaussians
    U = KMeans(n_clusters=3, random_state=0).fit(X).memberships_
    assert np.all((U == 0.0) | (U == 1.0))


def test_memberships_sum_to_one(three_gaussians):
    """Each point belongs to exactly one cluster."""
    X, _ = three_gaussians
    U = KMeans(n_clusters=3, random_state=0).fit(X).memberships_
    np.testing.assert_array_equal(U.sum(axis=0), 1.0)


def test_memberships_shape(three_gaussians):
    """U has shape (n_clusters, n_samples), matching FCM/GK convention."""
    X, _ = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert model.memberships_.shape == (3, X.shape[0])


def test_labels_match_memberships(three_gaussians):
    """labels_ should equal argmax of memberships_."""
    X, _ = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(
        model.labels_, np.argmax(model.memberships_, axis=0)
    )


def test_has_required_attributes_after_fit(three_gaussians):
    """All BaseClusterer attributes should be populated after fit."""
    X, _ = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert model.centers_ is not None
    assert model.labels_ is not None
    assert model.memberships_ is not None
    assert model.n_clusters_ == 3
    assert model.n_iter_ is not None
    assert model.history_ is not None
    assert model.inertia_ is not None


# ====================================================================== #
# 2. Correctness                                                          #
# ====================================================================== #


def test_recovers_known_centers(three_gaussians):
    """K-means should find true centers on well-separated isotropic data."""
    X, true_centers = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    nearest = cdist(true_centers, model.centers_).min(axis=1)
    assert nearest.max() < 0.2


def test_deterministic_with_same_seed(three_gaussians):
    """Same seed => identical clustering."""
    X, _ = three_gaussians
    m1 = KMeans(n_clusters=3, random_state=42).fit(X)
    m2 = KMeans(n_clusters=3, random_state=42).fit(X)
    np.testing.assert_array_equal(m1.labels_, m2.labels_)
    np.testing.assert_array_equal(m1.centers_, m2.centers_)


def test_inertia_is_finite(three_gaussians):
    """Final inertia should be a positive finite number."""
    X, _ = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert np.isfinite(model.inertia_)
    assert model.inertia_ > 0


def test_more_clusters_reduce_inertia(three_gaussians):
    """Adding clusters cannot increase total within-cluster variance."""
    X, _ = three_gaussians
    inertia_3 = KMeans(n_clusters=3, random_state=0).fit(X).inertia_
    inertia_5 = KMeans(n_clusters=5, random_state=0).fit(X).inertia_
    assert inertia_5 <= inertia_3


# ====================================================================== #
# 3. Edge cases                                                           #
# ====================================================================== #


def test_predict_on_new_data(three_gaussians):
    """Predicting on the true centers should give three distinct labels."""
    X, true_centers = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = model.predict(true_centers)
    assert len(set(labels)) == 3


def test_transform_returns_valid_partition(three_gaussians):
    """transform() returns one-hot memberships for new data."""
    X, _ = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    rng = np.random.default_rng(1)
    X_new = rng.normal(0, 1, size=(20, 2))
    U_new = model.transform(X_new)
    assert U_new.shape == (3, 20)
    assert np.all((U_new == 0.0) | (U_new == 1.0))
    np.testing.assert_array_equal(U_new.sum(axis=0), 1.0)


def test_fit_predict_equivalent(three_gaussians):
    X, _ = three_gaussians
    a = KMeans(n_clusters=3, random_state=0).fit_predict(X)
    b = KMeans(n_clusters=3, random_state=0).fit(X).labels_
    np.testing.assert_array_equal(a, b)


def test_single_cluster():
    """c=1: the unique center should be the data mean."""
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 1.0, size=(100, 2))
    model = KMeans(n_clusters=1, random_state=0).fit(X)
    np.testing.assert_allclose(model.centers_[0], X.mean(axis=0), atol=1e-6)
    np.testing.assert_array_equal(model.memberships_, 1.0)


def test_predict_is_consistent_with_fit(three_gaussians):
    """Calling predict() on the training data should return labels_."""
    X, _ = three_gaussians
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(model.predict(X), model.labels_)


# ====================================================================== #
# 4. Comparison with the fuzzy algorithms (sanity, not correctness)       #
# ====================================================================== #


def test_uniform_api_with_fcm_and_gk(three_gaussians):
    """All three algorithms should share the same attributes after fit."""
    from fuzzyclust import FCM, GK
    X, _ = three_gaussians

    for AlgoClass in [KMeans, FCM, GK]:
        algo = AlgoClass(n_clusters=3, random_state=0).fit(X)
        # Every model exposes these.
        assert algo.centers_.shape == (3, 2)
        assert algo.labels_.shape == (X.shape[0],)
        assert algo.memberships_.shape == (3, X.shape[0])
        # Memberships are a valid (possibly degenerate) partition.
        np.testing.assert_allclose(algo.memberships_.sum(axis=0), 1.0)


# ====================================================================== #
# 5. Input validation                                                     #
# ====================================================================== #


def test_rejects_zero_clusters():
    with pytest.raises(ValueError, match="n_clusters"):
        KMeans(n_clusters=0)


def test_rejects_unknown_init():
    with pytest.raises(ValueError, match="init"):
        KMeans(n_clusters=3, init="not_real")


def test_rejects_zero_n_init():
    with pytest.raises(ValueError, match="n_init"):
        KMeans(n_clusters=3, n_init=0)


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="2D"):
        KMeans(n_clusters=3).fit(np.array([1.0, 2.0, 3.0]))


def test_rejects_nan_input():
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(ValueError, match="NaN|infinite"):
        KMeans(n_clusters=2).fit(X)


def test_rejects_too_few_samples():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="n_clusters"):
        KMeans(n_clusters=3).fit(X)


def test_rejects_predict_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        KMeans(n_clusters=3).predict(np.array([[1.0, 2.0]]))
