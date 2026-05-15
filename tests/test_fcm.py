"""
Tests for the FCM implementation.

Each test encodes one fact we know about the algorithm. If a refactor
breaks any of these facts, the test goes red and points at the problem
before it propagates into experimental results.

Organized into four sections:
- Partition invariants     (memberships are a valid fuzzy partition)
- Algorithm correctness    (the optimization does what it should)
- Edge cases               (degenerate inputs don't blow up)
- Input validation         (bad arguments raise clear errors)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from fuzzyclust.fcm import FCM


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


@pytest.fixture
def four_gaussians_uneven():
    """The data from Table I of Kaymak & Setnes (2002).

    Used to confirm that standard FCM *fails* on uneven data
    distributions: this is the documented weakness that motivates E-FCM,
    and a regression check that our FCM still exhibits it (i.e. we
    haven't accidentally fixed it the wrong way).
    """
    rng = np.random.default_rng(0)
    centers = np.array([[-0.5, -0.4], [0.1, 0.2], [0.5, 0.7], [0.6, -0.3]])
    stds = [(0.2, 0.2), (0.1, 0.1), (0.2, 0.1), (0.2, 0.25)]
    sizes = [150, 30, 30, 50]
    X = np.vstack([
        rng.normal(c, s, size=(n, 2))
        for c, s, n in zip(centers, stds, sizes)
    ])
    return X, centers


# ====================================================================== #
# 1. Partition invariants                                                 #
# ====================================================================== #


def test_memberships_sum_to_one(three_gaussians):
    """Equation (5) in the paper: sum_i u_{ik} = 1 for every point k."""
    X, _ = three_gaussians
    U = FCM(n_clusters=3, random_state=0).fit(X).memberships_
    np.testing.assert_allclose(U.sum(axis=0), 1.0, atol=1e-10)


def test_memberships_in_unit_interval(three_gaussians):
    """0 <= u_{ik} <= 1 for all i, k."""
    X, _ = three_gaussians
    U = FCM(n_clusters=3, random_state=0).fit(X).memberships_
    assert U.min() >= 0.0
    assert U.max() <= 1.0


def test_memberships_are_finite(three_gaussians):
    """No NaN or inf, even with the inverse-distance update formula."""
    X, _ = three_gaussians
    U = FCM(n_clusters=3, random_state=0).fit(X).memberships_
    assert np.all(np.isfinite(U))


def test_memberships_shape(three_gaussians):
    """U has shape (n_clusters, n_samples), not the other way around."""
    X, _ = three_gaussians
    model = FCM(n_clusters=3, random_state=0).fit(X)
    assert model.memberships_.shape == (3, X.shape[0])


def test_labels_are_argmax(three_gaussians):
    """labels_ should equal argmax of memberships_."""
    X, _ = three_gaussians
    model = FCM(n_clusters=3, random_state=0).fit(X)
    expected = np.argmax(model.memberships_, axis=0)
    np.testing.assert_array_equal(model.labels_, expected)


# ====================================================================== #
# 2. Algorithm correctness                                                #
# ====================================================================== #


def test_objective_is_nonincreasing(three_gaussians):
    """Alternating optimization must never increase J."""
    X, _ = three_gaussians
    history = FCM(n_clusters=3, random_state=0).fit(X).history_
    J = np.array(history.objective)
    # Allow tiny floating-point noise.
    assert np.all(np.diff(J) <= 1e-9), (
        f"Objective increased at iterations: "
        f"{np.where(np.diff(J) > 1e-9)[0].tolist()}"
    )


def test_objective_strictly_decreases_for_first_iterations(three_gaussians):
    """On non-trivial data, J should drop substantially before convergence."""
    X, _ = three_gaussians
    J = np.array(FCM(n_clusters=3, random_state=0).fit(X).history_.objective)
    # The first iteration should have a much higher J than the last.
    assert J[0] > J[-1] * 1.05  # at least 5% reduction overall


def test_recovers_known_centers(three_gaussians):
    """On well-separated data, found centers should match the truth."""
    X, true_centers = three_gaussians
    model = FCM(n_clusters=3, random_state=0, init="kmeans++").fit(X)
    nearest = cdist(true_centers, model.centers_).min(axis=1)
    assert nearest.max() < 0.2


def test_recovers_centers_random_init(three_gaussians):
    """Random-membership init should also work on easy data, given enough seeds."""
    X, true_centers = three_gaussians
    # Try a few seeds; at least one should converge to the truth.
    errors = []
    for seed in range(5):
        model = FCM(n_clusters=3, random_state=seed,
                    init="random_memberships").fit(X)
        errors.append(cdist(true_centers, model.centers_).min(axis=1).max())
    assert min(errors) < 0.2


def test_deterministic_with_same_seed(three_gaussians):
    """Same seed => bit-identical results."""
    X, _ = three_gaussians
    m1 = FCM(n_clusters=3, random_state=42).fit(X)
    m2 = FCM(n_clusters=3, random_state=42).fit(X)
    np.testing.assert_array_equal(m1.centers_, m2.centers_)
    np.testing.assert_array_equal(m1.memberships_, m2.memberships_)


def test_different_seeds_give_different_inits(three_gaussians):
    """Different seeds shouldn't all collapse to the same iteration count."""
    X, _ = three_gaussians
    iters = {
        FCM(n_clusters=3, random_state=s,
            init="random_memberships").fit(X).n_iter_
        for s in range(5)
    }
    # We expect at least *some* variation in iteration count.
    assert len(iters) > 1


def test_kmeanspp_converges_faster_than_random(three_gaussians):
    """k-means++ init should typically reach convergence in fewer steps."""
    X, _ = three_gaussians
    kpp = FCM(n_clusters=3, random_state=0, init="kmeans++").fit(X).n_iter_
    rnd = FCM(n_clusters=3, random_state=0,
              init="random_memberships").fit(X).n_iter_
    assert kpp <= rnd


def test_higher_m_gives_fuzzier_memberships(three_gaussians):
    """As m increases, memberships should become more uniform."""
    X, _ = three_gaussians
    U_low = FCM(n_clusters=3, m=1.5, random_state=0).fit(X).memberships_
    U_high = FCM(n_clusters=3, m=4.0, random_state=0).fit(X).memberships_
    # The maximum membership per point measures "crispness": closer to 1
    # means more decisive, closer to 1/c means fuzzier.
    assert U_low.max(axis=0).mean() > U_high.max(axis=0).mean()


def test_fcm_fails_on_uneven_data(four_gaussians_uneven):
    """Regression check: standard FCM should miss the small clusters.

    This is the failure mode documented in Fig. 3(a) of the paper. If
    this test ever starts passing, it likely means we accidentally
    implemented something other than vanilla FCM.
    """
    X, true_centers = four_gaussians_uneven
    model = FCM(n_clusters=4, random_state=42, init="kmeans++").fit(X)
    nearest = cdist(true_centers, model.centers_).min(axis=1)
    # At least one true center should be far from any found center.
    assert nearest.max() > 0.2


# ====================================================================== #
# 3. Edge cases                                                           #
# ====================================================================== #


def test_point_coincides_with_center():
    """Data points exactly at cluster centers => one-hot memberships."""
    X = np.array([[-3.0, 0.0]] * 30 + [[3.0, 0.0]] * 30, dtype=float)
    model = FCM(n_clusters=2, random_state=0, init="kmeans++").fit(X)
    # Every point coincides with one of the centers.
    assert np.all((model.memberships_ == 1.0) | (model.memberships_ == 0.0))
    assert np.all(model.memberships_.sum(axis=0) == 1.0)


def test_single_cluster():
    """c=1 should put the unique center at the mean of the data."""
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 1.0, size=(100, 2))
    model = FCM(n_clusters=1, random_state=0).fit(X)
    np.testing.assert_allclose(model.centers_[0], X.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(model.memberships_, 1.0)


def test_more_clusters_than_distinct_points():
    """If c equals n_samples, every point gets its own cluster."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    model = FCM(n_clusters=3, random_state=0, init="kmeans++").fit(X)
    # Centers should be a permutation of the points.
    assert cdist(X, model.centers_).min(axis=1).max() < 1e-6


def test_predict_on_new_data(three_gaussians):
    """Points near a true center should be assigned to that cluster."""
    X, true_centers = three_gaussians
    model = FCM(n_clusters=3, random_state=0, init="kmeans++").fit(X)
    labels = model.predict(true_centers)
    # The three true centers should land in three distinct clusters.
    assert len(set(labels)) == 3


def test_transform_returns_valid_partition(three_gaussians):
    """transform() on new data should return a valid U matrix."""
    X, _ = three_gaussians
    model = FCM(n_clusters=3, random_state=0).fit(X)
    rng = np.random.default_rng(1)
    X_new = rng.normal(0, 1, size=(20, 2))
    U_new = model.transform(X_new)
    assert U_new.shape == (3, 20)
    np.testing.assert_allclose(U_new.sum(axis=0), 1.0, atol=1e-10)


def test_fit_predict_equivalent_to_fit_then_labels(three_gaussians):
    """fit_predict(X) should equal fit(X).labels_."""
    X, _ = three_gaussians
    labels_a = FCM(n_clusters=3, random_state=0).fit_predict(X)
    labels_b = FCM(n_clusters=3, random_state=0).fit(X).labels_
    np.testing.assert_array_equal(labels_a, labels_b)


# ====================================================================== #
# 4. Input validation                                                     #
# ====================================================================== #


def test_rejects_zero_clusters():
    with pytest.raises(ValueError, match="n_clusters"):
        FCM(n_clusters=0)


def test_rejects_m_equals_one():
    with pytest.raises(ValueError, match="m"):
        FCM(n_clusters=3, m=1.0)


def test_rejects_m_below_one():
    with pytest.raises(ValueError, match="m"):
        FCM(n_clusters=3, m=0.5)


def test_rejects_unknown_init():
    with pytest.raises(ValueError, match="init"):
        FCM(n_clusters=3, init="not_a_real_strategy")


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="2D"):
        FCM(n_clusters=3).fit(np.array([1.0, 2.0, 3.0]))


def test_rejects_nan_input():
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(ValueError, match="NaN|infinite"):
        FCM(n_clusters=2).fit(X)


def test_rejects_too_few_samples():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="n_clusters"):
        FCM(n_clusters=3).fit(X)


def test_rejects_predict_before_fit():
    """Using predict() before fit() should give a clear error."""
    with pytest.raises(RuntimeError, match="fit"):
        FCM(n_clusters=3).predict(np.array([[1.0, 2.0]]))