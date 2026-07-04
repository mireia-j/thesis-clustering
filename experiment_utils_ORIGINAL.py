"""
experiment_utils.py
===================
Reusable harness for benchmarking the fuzzy-clustering family
(FCM, eFCM, QFCM, YFCM, eQFCM) across:

    * datasets (toy + real)
    * distance metrics : 'euclidean' | 'mahalanobis'
    * merge            : True | False
    * fuzzification params (m / lambda / m1,m2,lambda1,lambda2)
    * several random inits per config  -> best chosen by MIN OBJECTIVE

The algorithm classes are passed in via a registry dict, so this module has
no hard dependency on your package and can be unit-tested with stubs.

Typical use (in a notebook):

    from experiment_utils import *
    REGISTRY = {'FCM': FCM, 'eFCM': eFCM, 'QFCM': QFCM, 'YFCM': YFCM, 'eQFCM': eQFCM}
    df = run_benchmark(REGISTRY, dataset_names=['iris','wine'], size='medium')
    best = select_best_seed(df)          # one row per config (min objective)
    table = best_ari_table(best)         # best ARI per dataset x algorithm
"""

import inspect
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# --------------------------------------------------------------------------- #
#  1. Datasets                                                                 #
# --------------------------------------------------------------------------- #
# Real datasets. sklearn built-ins load offline; the rest use OpenML (needs
# internet -- fine on Colab). Names chosen to MATCH the Kanzawa (2024) paper:
#   - 'breast' -> Wisconsin ORIGINAL (breast-w, 699/9/2), NOT sklearn's WDBC.
#   - 'heart'  -> Statlog heart (270/13/2).
#   - 'bupa'   -> liver-disorders (345/6/2); see caveat below.
def _finalize(X, y):
    X = np.asarray(X, dtype=float)
    y = LabelEncoder().fit_transform(np.asarray(y).ravel())
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], y[ok], int(len(np.unique(y)))


def load_dataset(name):
    """Return (X, y, n_classes) for a named dataset."""
    from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                                  fetch_openml)
    key = name.lower()

    if key == "iris":
        d = load_iris();           return _finalize(d.data, d.target)
    if key == "wine":
        d = load_wine();           return _finalize(d.data, d.target)

    # --- OpenML datasets (internet required) --------------------------------
    if key in ("breast", "bcw", "breast-w", "breast_w"):
        d = fetch_openml("breast-w", version=1, as_frame=True)
        df = d.frame.dropna()
        return _finalize(df.drop(columns=["Class"]).values, df["Class"].values)

    if key in ("heart", "heart-statlog"):
        d = fetch_openml("heart-statlog", version=1, as_frame=True)
        df = d.frame.dropna()
        tgt = df.columns[-1]
        X = df.drop(columns=[tgt]).apply(pd.to_numeric, errors="coerce")
        return _finalize(X.values, df[tgt].values)

    if key in ("bupa", "liver", "liver-disorders"):
        # CAVEAT: the historical Bupa "selector" column was a train/test split
        # flag, not a biological class. It is nonetheless the conventional
        # 2-class target (as in the paper). Verify this matches your intent.
        d = fetch_openml("liver-disorders", version=1, as_frame=True)
        df = d.frame.dropna()
        tgt = df.columns[-1]
        X = df.drop(columns=[tgt]).apply(pd.to_numeric, errors="coerce")
        return _finalize(X.values, df[tgt].values)

    if key == "glass":
        d = fetch_openml("glass", version=1, as_frame=True)
        df = d.frame.dropna()
        tgt = "Type" if "Type" in df.columns else df.columns[-1]
        X = df.drop(columns=[tgt]).apply(pd.to_numeric, errors="coerce")
        return _finalize(X.values, df[tgt].values)

    if key == "sonar":
        d = fetch_openml("sonar", version=1, as_frame=True)
        df = d.frame.dropna()
        tgt = "Class" if "Class" in df.columns else df.columns[-1]
        X = df.drop(columns=[tgt]).apply(pd.to_numeric, errors="coerce")
        return _finalize(X.values, df[tgt].values)

    # A WDBC fallback so 'breast_wdbc' still works if you want sklearn's version
    if key in ("breast_wdbc", "wdbc"):
        d = load_breast_cancer();  return _finalize(d.data, d.target)

    raise ValueError(f"Unknown dataset '{name}'")


DEFAULT_REAL = ["breast", "heart", "iris", "wine", "bupa", "glass", "sonar"]


# --------------------------------------------------------------------------- #
#  2. Per-algorithm objective (Eqs. 4/6/7/27, d_ik = squared distance)         #
# --------------------------------------------------------------------------- #
def _squared_center_distance(model, X):
    """Un-floored SQUARED distance from each point to each center, d_ik^2 (N,C).

    Reliable for both metrics regardless of the `distance_squared` GK convention:
    `model.distance(...)` returns the non-squared distance (Euclidean ||.||, or the
    GK sqrt-Mahalanobis), so squaring it yields exactly d_ik^2 in both cases:
        euclidean    -> ||x_k - v_i||^2
        mahalanobis  -> |P_i|^(1/n) (x_k - v_i)^T P_i^{-1} (x_k - v_i)
    """
    d = np.asarray(model.distance(X, model.u))     # non-squared, (N, C)
    return d ** 2


def compute_objective(model, X):
    """Objective value of a *fitted* model, used only to pick the best seed
    WITHIN one config (never to compare across configs/algorithms).

    d_ik^2 is computed via `_squared_center_distance`, so the mahalanobis case is
    handled correctly even if `distance_squared` still returns the non-squared GK
    value."""
    u = model.u
    name = type(model).__name__
    D = _squared_center_distance(model, X)                 # d_ik^2, (N, C)

    if name == "eQFCM":                                   # Eq. (27)
        um1 = u ** model.m1
        a1m2 = model.alpha[None, :] ** (1.0 - model.m2)
        data = np.sum(a1m2 * um1 * D)
        reg1 = (1.0 / model.lambda1) / (model.m2 - 1.0) * np.sum(um1 * (a1m2 - 1.0))
        reg2 = (1.0 / model.lambda2) / (model.m1 - 1.0) * np.sum(um1 - u)
        return float(data + reg1 + reg2)

    if name == "QFCM":                                    # Eq. (6)
        um = u ** model.m
        a1m = model.alpha[None, :] ** (1.0 - model.m)
        data = np.sum(a1m * um * D)
        reg = (1.0 / model.lambda_) / (model.m - 1.0) * np.sum(a1m * um - u)
        return float(data + reg)

    if name == "YFCM":                                    # Eq. (7)
        um = u ** model.m
        data = np.sum(um * D)
        reg = -(1.0 / model.lambda_) * np.sum(um * np.log(model.alpha)[None, :])
        return float(data + reg)

    if name == "eFCM":                                    # Eq. (4), Kaymak 2002
        # J~ = sum_ik u_ik^m (d_ik^2 - r_i^2), using the *un-floored* d^2 and the
        # radii the model converged with (self.radii holds the non-squared r_i).
        # NB: this can be negative (points inside a prototype have d^2 < r^2) --
        # that is correct; do NOT floor it here.
        r = getattr(model, "radii", None)
        if r is None:                                     # safety fallback
            P = model.covariance_matrix(X, u)
            r = model.radius_centers(P)
        r2 = np.asarray(r, dtype=float) ** 2              # (C,)
        return float(np.sum((u ** model.m) * (D - r2[None, :])))

    # plain FCM (no volume prototype):  J = sum_ik u_ik^m d_ik^2
    return float(np.sum((u ** getattr(model, "m", 2.0)) * D))


# --------------------------------------------------------------------------- #
#  3. Parameter grids                                                          #
# --------------------------------------------------------------------------- #
# Membership fuzzifier m/m1 behaves like the classic FCM fuzzifier.
# m2 (cluster-size-controller exponent) MUST be in (1, 2): the paper shows
# m2 > 2 gives wrong clusters and m2 <= 1 gives division-by-zero NaNs.
# lambda enters only through lambda*d, so its useful range tracks the distance
# scale -> swept logarithmically. Standardized data keeps d ~ O(1).
GRIDS = {
    "quick":  dict(m=[2.0],            m2=[1.1],
                   lam=[1.0],
                   m1=[2.0], lam1=[1.0], lam2=[1.0]),
    "medium": dict(m=[1.1, 2.0],       m2=[1.05, 1.1],
                   lam=[0.1, 1.0, 10.0],
                   m1=[1.1, 2.0], lam1=[0.1, 10.0], lam2=[0.1, 10.0]),
    "paper":  dict(m=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   m2=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   lam=[10.0 ** e for e in range(-15, 16, 2)],
                   m1=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   lam1=[10.0 ** e for e in range(-15, 16, 2)],
                   lam2=[10.0 ** e for e in range(-15, 16, 2)]),
}


def build_param_grid(algo, size="medium"):
    """List of algorithm-specific fuzzification-parameter dicts."""
    g = GRIDS[size]
    if algo in ("FCM", "eFCM"):
        return [{"m": mm} for mm in g["m"]]
    if algo in ("QFCM", "YFCM"):
        return [{"m": mm, "lambda_": ll} for mm in g["m"] for ll in g["lam"]]
    if algo == "eQFCM":
        return [{"m1": a, "m2": b, "lambda1": c, "lambda2": d}
                for a in g["m1"] for b in g["m2"]
                for c in g["lam1"] for d in g["lam2"]]
    raise ValueError(f"No grid for algorithm '{algo}'")


# --------------------------------------------------------------------------- #
#  4. Runner                                                                   #
# --------------------------------------------------------------------------- #
def _filter_kwargs(cls, kwargs):
    """Keep only kwargs the constructor actually accepts (robust to differing
    signatures, e.g. eFCM not taking lambda_)."""
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def _fit_once(cls, kwargs, X, y):
    """Fit one model; return metrics dict. Never raises."""
    rec = dict(objective=np.nan, ari=np.nan, nmi=np.nan,
               n_final=np.nan, status="ok", seconds=np.nan)
    try:
        t0 = time.time()
        model = cls(**_filter_kwargs(cls, kwargs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        rec["seconds"] = time.time() - t0

        u = getattr(model, "u", None)
        if u is None or not np.all(np.isfinite(u)):
            rec["status"] = "nan_membership"
            return rec, model
        labels = model.labels()
        rec["ari"] = adjusted_rand_score(y, labels)
        rec["nmi"] = normalized_mutual_info_score(y, labels)
        rec["n_final"] = int(getattr(model, "n_clusters", len(np.unique(labels))))
        rec["objective"] = compute_objective(model, X)
        if not np.isfinite(rec["objective"]):
            rec["status"] = "nan_objective"
    except Exception as e:                       # noqa: BLE001
        rec["status"] = f"error:{type(e).__name__}"
    return rec, None


def run_benchmark(registry,
                  dataset_names=DEFAULT_REAL,
                  algorithms=("FCM", "eFCM", "QFCM", "YFCM", "eQFCM"),
                  metrics=("euclidean", "mahalanobis"),
                  merges=(False, True),
                  size="medium",
                  n_seeds=5,
                  normalize=True,
                  over_estimate=10,
                  epsilon_1=0.01,
                  max_iter=150,
                  error=1e-5,
                  save_path=None,
                  verbose=True):
    """Full sweep. Returns a long-form DataFrame (one row per fitted model).

    `registry` : {algo_name: class}. Only algorithms present here are run.
    `save_path`: if given, CSV is (re)written after each dataset (crash-safe).
    """
    rows = []
    for dname in dataset_names:
        X, y, C = load_dataset(dname)
        Xs = StandardScaler().fit_transform(X) if normalize else X
        if verbose:
            print(f"[{dname}] N={X.shape[0]} M={X.shape[1]} C={C}")

        for algo in algorithms:
            if algo not in registry:
                continue
            cls = registry[algo]
            grid = build_param_grid(algo, size)

            for metric in metrics:
                for merge in merges:
                    n_clusters = over_estimate if merge else C
                    for cfg_id, params in enumerate(grid):
                        for seed in range(n_seeds):
                            kwargs = dict(params)
                            kwargs.update(n_clusters=n_clusters, merge=merge,
                                          distance_metric=metric,
                                          random_state=seed, epsilon_1=epsilon_1,
                                          max_iter=max_iter, error=error)
                            rec, _ = _fit_once(cls, kwargs, Xs, y)
                            rows.append(dict(
                                dataset=dname, algo=algo, metric=metric,
                                merge=merge, config_id=cfg_id, seed=seed,
                                params=str(params), n_true=C, **rec))
            if verbose:
                print(f"    {algo:6s} done ({len(grid)} configs x {n_seeds} seeds"
                      f" x {len(metrics)} metrics x {len(merges)} merge)")

        if save_path:                                   # crash-safe checkpoint
            pd.DataFrame(rows).to_csv(save_path, index=False)

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


# --------------------------------------------------------------------------- #
#  5. Summaries                                                                #
# --------------------------------------------------------------------------- #
def select_best_seed(df):
    """One row per (dataset, algo, metric, merge, config): the seed with the
    smallest objective (the paper's selection rule). NaN objectives dropped."""
    ok = df[np.isfinite(df["objective"])].copy()
    keys = ["dataset", "algo", "metric", "merge", "config_id"]
    idx = ok.groupby(keys)["objective"].idxmin()
    return ok.loc[idx].reset_index(drop=True)


def best_ari_table(best):
    """Pivot: best ARI per dataset x algorithm (best over all configs/metric/
    merge for that pair). Mirrors the paper's Table 2 style."""
    g = best.groupby(["dataset", "algo"])["ari"].max().reset_index()
    return g.pivot(index="dataset", columns="algo", values="ari")


def best_config_details(best):
    """For each dataset x algo, the full winning row (max ARI)."""
    idx = best.groupby(["dataset", "algo"])["ari"].idxmax()
    cols = ["dataset", "algo", "metric", "merge", "params", "ari", "nmi",
            "n_true", "n_final", "objective"]
    return best.loc[idx, cols].sort_values(["dataset", "algo"]).reset_index(drop=True)
