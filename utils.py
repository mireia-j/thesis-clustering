"""Utility functions for fuzzy clustering."""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, silhouette_samples, homogeneity_completeness_v_measure)
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml
import copy

try:
    from fcm import FCM
    from efcm import eFCM
    from qfcm import QFCM
    from yfcm import YFCM
    from eqfcm import eQFCM
except ImportError:
    from fuzzyclust_algorithms.fcm import FCM
    from fuzzyclust_algorithms.efcm import eFCM
    from fuzzyclust_algorithms.qfcm import QFCM
    from fuzzyclust_algorithms.yfcm import YFCM
    from fuzzyclust_algorithms.eqfcm import eQFCM
    

def plot_results(X, y, name_fig, title="Fuzzy Clustering Results"):
  """X is the original data, y is the true labels (discrete)."""
  plt.figure(figsize=(8, 6))
  scatter = plt.scatter(
      X[:, 0], X[:, 1],
      c=y,
      cmap="viridis",
      s=20,
      alpha=0.7
  )

  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.title(title)
  plt.axis("equal")
  plt.grid(True, alpha=0.3)

  # Legend
  handles, labels = scatter.legend_elements()
  plt.legend(handles, [f"Cluster {i}" for i in range(3)])

  plt.savefig(f"{name_fig}.png")


# _________________________________________________________________
# BOOTSTRAP METRICS

"""
The value of lambda cannot be negative.
The value of m1 cannot be 1.
The value of m cannot be 1 
The Kanzawa paper plays with lam for very small values (10^-15) and very big values (10^15).
The Kanzawa paper plays with m, m1, m2 for values around 1.
"""
GRIDS = {
    "quick":  dict(m=[2.0],  m1=[2.0],  m2=[1.1],
                   lam=[1.0],  lam1=[1.0], lam2=[1.0], alpha=["original"]),

    "medium": dict(m=[1.1, 1.5, 2.0],  m1=[1.1, 1.5, 2.0],  m2=[1.05, 1.1, 1.3, 1.5],
                   lam=[0.01, 0.1, 1.0, 10.0], lam1=[0.1, 1.0, 10.0], lam2=[0.1, 1.0, 10.0],
                   alpha=["original", "new", 0.25, 0.5, 0.75]),

    "large": dict(m=[1.1, 1.3, 1.5, 1.8, 2.0],  m1=[1.1, 1.3, 1.5, 1.8, 2.0],  m2=[1.05, 1.1, 1.3, 1.5, 1.8, 2.0],
                     lam=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  lam1=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  lam2=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                     alpha=["original", "new", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),

    "extra_large":  dict(m=[1.05, 1.1, 1.3, 1.5, 2.0],  
                         m1=[1.3, 1.5, 1.8, 2.0],  m2=[1.05, 1.1, 1.3, 1.5, 1.8, 2.0],
                         lam=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 
                         lam1=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 
                         lam2=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                         alpha=["original", "new", 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),

    "paper":  dict(m=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   m1=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   m2=[1 + 2.0 ** -k for k in range(5, 30, 2)],
                   lam=[10.0 ** e for e in range(-15, 16, 2)],
                   lam1=[10.0 ** e for e in range(-15, 16, 2)],
                   lam2=[10.0 ** e for e in range(-15, 16, 2)]),
}

METRICS = ["PC", "MPC", "PE", "MPE", "XBI", "FKI", "FS", "Hom", "Comp", "V-measure", "ARS", "AMI", "Silh"]

def load_openml_numpy(name, version=1):
    """
    Function for having all the loaded datasets (e.g. sonar, heart, etc.) in the same format
    which is (X, y) in the type of (numpy.ndarray, numpy.ndarray).

    Special case: for "liver-disorders" (bupa) the OpenML default target is the
    `drinks` count (a numeric field with ~16 distinct values), NOT a class label.
    Following McDermott & Forsyth, we binarize it at drinks >= 3 to get a 2-class
    target and discard the `selector` field.
    """
    X, y = fetch_openml(
        name=name,
        version=version,
        return_X_y=True,
        as_frame=False
    )

    # sparse -> dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X)
    y = np.asarray(y)

    # bupa: turn the raw `drinks` count into a 2-class target
    if name == "liver-disorders":
        y = (y.astype(float) >= 3).astype(int)

    return X, y

# BORRAR
def bootstrap_metrics_2(model, X, y, n_boot, apply_bootstrap, random_state=0, verbose=True, frac=0.9):
    """
    This function gets the model with its right parameters and returns the right mean and the std of the bootstrap.
    model: instance of model without fitting it yet
    apply_bootstrap: if it is True we apply the bootstrap. If False, we only apply different initialization.
    frac: fraction of the data (per class) to draw at each bootstrap iteration when apply_bootstrap=True.
    """
    print_params = False
    rng = np.random.RandomState(random_state)
    N = X.shape[0]
    distance_metric = model.distance_metric
    rows = []
    n_failed = 0
    show_n_final = model.merge # is for showing the initial and final values of n_clusters when merging

    # For bootstrap part: keep 90% of data and do stratified sampling
    classes = np.unique(y)
    class_indices = {c: np.flatnonzero(y == c) for c in classes}

    for i in range(n_boot):
        if apply_bootstrap:
            idx_parts = []
            for c in classes:
                c_idx = class_indices[c]
                n_c = c_idx.shape[0]
                n_draw = max(1, int(round(frac * n_c)))  # keep ~90% of each class, at least 1
                sampled = rng.choice(c_idx, size=n_draw, replace=True)
                idx_parts.append(sampled)
            idx = np.concatenate(idx_parts)
            rng.shuffle(idx)  # avoid keeping samples grouped by class
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y

        m_i = copy.deepcopy(model)
        m_i.random_state = i**2 + 1 # initialize different U(0) every time
        m_i.fit(Xb)
        u = m_i.u
        if u is None or not np.all(np.isfinite(u)):
            n_failed += 1
            failed = {k: np.nan for k in METRICS}
            if show_n_final:
                failed["n_initial"] = np.nan
                failed["n_final"] = np.nan
            rows.append(failed)
            continue

        labels = m_i.labels()
        centers = m_i.centers

        h, c, v = homogeneity_completeness_v_measure(yb, labels)
        row = {
            "PC": partition_coefficient(u),
            "MPC": modified_partition_coefficient(u),
            "PE": partition_entropy(u),
            "MPE": modified_partition_entropy(u),
            "XBI": xie_beni(Xb, u, centers, distance_metric=distance_metric),
            "FKI": fukuyama_sugeno(Xb, u, centers, distance_metric=distance_metric),
            "FS": fuzzy_silhouette(Xb, u),
            "Hom": h, "Comp": c, "V-measure": v,
            "ARS": adjusted_rand_score(yb, labels),
            "AMI": adjusted_mutual_info_score(yb, labels),
            "Silh": safe_silhouette(Xb, labels),
        }
        if show_n_final:
            row["n_initial"] = m_i.n_clusters_init   
            row["n_final"]   = m_i.n_clusters        
        rows.append(row)                             

    if verbose and n_failed:
        print(f"  [{n_failed}/{n_boot} resamples produced NaN memberships]")
        print_params = True

    df = pd.DataFrame(rows)
    info_metrics = {m: (df[m].mean(), df[m].std()) for m in METRICS}   
    if show_n_final:
        dict_final = {"n_initial": (df["n_initial"].mean(), df["n_initial"].std()), 
                      "n_final": (df["n_final"].mean(), df["n_final"].std())}
        return {**dict_final, **info_metrics}
    
    return (info_metrics, print_params)

def bootstrap_metrics(model, X, y, n_boot, apply_bootstrap, random_state=0,
                      verbose=True, frac=0.9, replace=False):
    """
    Returns the mean and std of the metrics across bootstrap/initialisation runs.
    model: instance of model, not yet fitted
    apply_bootstrap: if True apply (stratified) resampling; if False only vary init.
    frac: fraction of each class to draw when apply_bootstrap=True.
    replace: False -> stratified subsampling (keeps ~frac of DISTINCT points, no duplicates).
             True  -> stratified m-out-of-n bootstrap (with replacement, duplicates possible).
    """
    print_params = False
    rng = np.random.RandomState(random_state)
    N = X.shape[0]
    distance_metric = model.distance_metric
    rows = []
    n_failed = 0
    show_n_final = model.merge  # show initial vs final n_clusters when merging

    classes = np.unique(y)
    class_indices = {c: np.flatnonzero(y == c) for c in classes}

    for i in range(n_boot):
        if apply_bootstrap:
            idx_parts = []
            for c in classes:
                c_idx = class_indices[c]
                n_c = c_idx.shape[0]
                # keep ~frac of each class; never exceed the class size, never below 1
                n_draw = max(1, min(n_c, int(round(frac * n_c))))
                sampled = rng.choice(c_idx, size=n_draw, replace=replace)
                idx_parts.append(sampled)
            idx = np.concatenate(idx_parts)
            rng.shuffle(idx)  # avoid keeping samples grouped by class
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y

        m_i = copy.deepcopy(model)
        m_i.random_state = i**2 + 1  # different U(0) init every iteration
        m_i.fit(Xb)

        u = m_i.u
        if u is None or not np.all(np.isfinite(u)):
            n_failed += 1
            failed = {k: np.nan for k in METRICS}
            if show_n_final:
                failed["n_initial"] = np.nan
                failed["n_final"] = np.nan
            rows.append(failed)
            continue

        labels = m_i.labels()
        centers = m_i.centers
        h, c, v = homogeneity_completeness_v_measure(yb, labels)
        row = {
            "PC":  partition_coefficient(u),
            "MPC": modified_partition_coefficient(u),
            "PE":  partition_entropy(u),
            "MPE": modified_partition_entropy(u),
            "XBI": xie_beni(Xb, u, centers, distance_metric=distance_metric),
            "FKI": fukuyama_sugeno(Xb, u, centers, distance_metric=distance_metric),
            "FS":  fuzzy_silhouette(Xb, u),
            "Hom": h, "Comp": c, "V-measure": v,
            "ARS": adjusted_rand_score(yb, labels),
            "AMI": adjusted_mutual_info_score(yb, labels),
            "Silh": safe_silhouette(Xb, labels),
        }
        if show_n_final:
            row["n_initial"] = m_i.n_clusters_init
            row["n_final"]   = m_i.n_clusters
        rows.append(row)

    if verbose and n_failed:
        print(f"  [{n_failed}/{n_boot} resamples produced NaN memberships]")
        print_params = True

    df = pd.DataFrame(rows)
    info_metrics = {m: (df[m].mean(), df[m].std()) for m in METRICS}
    if show_n_final:
        dict_final = {
            "n_initial": (df["n_initial"].mean(), df["n_initial"].std()),
            "n_final":   (df["n_final"].mean(),   df["n_final"].std()),
        }
        return {**dict_final, **info_metrics}
    return (info_metrics, print_params)


def create_df_params(model, data, C, dist_metric, size, n_boot, apply_bootstrap):
    """
    This function is for once it gets a model it gets what params it needs to play with from grid 
    and applies the bootstrap to each one of them

    model: "FCM", "eFCM", "QFCM", "YFCM" or "eQFCM"
    C: number of authentic clusters
    """
    if model == "FCM":
        base_model = FCM(n_clusters=C, merge=False, distance_metric=dist_metric)
    elif model == "eFCM":
        base_model = eFCM(n_clusters=C, merge=False, distance_metric=dist_metric)
    elif model == "QFCM":
        base_model = QFCM(n_clusters=C, merge=False, distance_metric=dist_metric)
    elif model == "YFCM":
        base_model = YFCM(n_clusters=C, merge=False, distance_metric=dist_metric)
    elif model == "eQFCM":
        base_model = eQFCM(n_clusters=C, merge=False, distance_metric=dist_metric)
    else:
        raise ValueError("model argument must be a STRING of the form 'FCM', 'eFCM', 'QFCM', 'YFCM' or 'eQFCM' ")

    results = []

    if model == "FCM" or model == "eFCM": # only has m and we don't vary it (stays 2)
        for param_m in [2]:
            model_copy = copy.deepcopy(base_model)
            # Set params that will vary
            model_copy.m = param_m
            metrics, print_params = bootstrap_metrics(model_copy, data[0], data[1], n_boot, apply_bootstrap, random_state=0)
            if print_params:
                print(f"error in combination: m={param_m}")

            row = {"param_m": param_m}
            for metric, (mean_val, std_val) in metrics.items():
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
            results.append(row)
    
    elif model == "QFCM" or model == "YFCM": # we vary m and lambda
        for param_lam in GRIDS[size]["lam"]:
            for param_m in GRIDS[size]["m"]:
                model_copy = copy.deepcopy(base_model)
                # Set params that will vary
                model_copy.m = param_m
                model_copy.lambda_ = param_lam
                metrics, print_params = bootstrap_metrics(model_copy, data[0], data[1], n_boot, apply_bootstrap, random_state=0)
                if print_params:
                    print(f"error in combination: m={param_m}, lam={param_lam}")

                row = {"param_m": param_m, "param_lam": param_lam}
                for metric, (mean_val, std_val) in metrics.items():
                    row[f"{metric}_mean"] = mean_val
                    row[f"{metric}_std"] = std_val
                results.append(row)
    
    elif model == "eQFCM":
        for param_m1 in GRIDS[size]["m1"]:
            for param_m2 in GRIDS[size]["m2"]:
                for param_lam1 in GRIDS[size]["lam1"]:
                    for param_lam2 in GRIDS[size]["lam2"]:
                        model_copy = copy.deepcopy(base_model)
                        # Set params that will vary
                        model_copy.m1 = param_m1
                        model_copy.m2 = param_m2
                        model_copy.lambda1 = param_lam1
                        model_copy.lambda2 = param_lam2
                        metrics, print_params = bootstrap_metrics(model_copy, data[0], data[1], n_boot, apply_bootstrap, random_state=0)
                        if print_params:
                            print(f"error in combination: m1={param_m1}, m2={param_m2}, lam1={param_lam1}, lam2={param_lam2}")

                        row = {"param_m1": param_m1, "param_m2": param_m2, "param_lam1": param_lam1, "param_lam2": param_lam2}
                        for metric, (mean_val, std_val) in metrics.items():
                            row[f"{metric}_mean"] = mean_val
                            row[f"{metric}_std"] = std_val
                        results.append(row)
        
    return pd.DataFrame(results)


def rank_by_vmeasure(df, n, col='V-measure_mean', new_col='ranking_params', decimals=8):
    """
    Rank rows of `df` by the top-n distinct values of `col`, keeping only
    the parameter columns and the resulting ranking.

    The highest value of `col` gets ranking_params = n, the second-highest
    *distinct* value gets n-1, and so on down to 1 for the n-th best distinct
    value. Rows whose value is not among the top-n distinct values (or is NaN)
    get ranking_params = NaN. Ties share the same rank (dense ranking), so a
    tie never "costs" more than one of the n available slots.

    As a final override, any row whose raw `col` value is below 0.0001 has
    its `new_col` forced to 0, regardless of the rank computed above.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        Number of distinct top values to rank (i.e. the best row(s) get n,
        down to 1 for the n-th best distinct value).
    col : str
        Column to rank by (default 'V-measure_mean').
    new_col : str
        Name of the output ranking column (default 'ranking_params').
    decimals : int
        Number of decimals `col` is rounded to before ranking, so that
        values which differ only by floating-point noise are treated as
        exact ties. Set to None to disable rounding.

    Returns
    -------
    pd.DataFrame
        Only the columns starting with 'param_' plus `new_col`.
    """
    result = df.copy()

    if col not in result.columns:
        result[new_col] = np.nan
    else:
        col_values = result[col]
        if decimals is not None:
            col_values = col_values.round(decimals)

        # Dense rank, descending: 1 = highest value, ties share the same rank,
        # and no rank is skipped after a tie (unlike default 'average'/'min').
        dense_rank = col_values.rank(method='dense', ascending=False)

        # Only the top-n distinct values are kept.
        within_top_n = dense_rank <= n

        # Flip the scale so the best value gets n, and the n-th best gets 1.
        result[new_col] = (n - dense_rank + 1).where(within_top_n)

        # Override: values below 0.0001 always get ranking_params = 0,
        # even if they would otherwise have landed in the top-n.
        low_value_mask = result[col] < 0.0001
        result.loc[low_value_mask, new_col] = 0

    param_cols = [c for c in result.columns if c.startswith('param_')]
    return result[param_cols + [new_col]]


def sum_rankings(df_list, ranking_col='ranking_params'):
    """
    Combine a list of ranked dataframes (as produced by `rank_by_vmeasure`)
    into a single dataframe, summing `ranking_col` over rows that share the
    same combination of parameter values. The result is sorted so that the
    highest summed ranking appears first.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        Each dataframe must contain the same set of 'param_*' columns plus
        `ranking_col`. Rows with NaN in `ranking_col` (i.e. not among the
        top-n for that particular ranking) contribute 0 to the sum.
    ranking_col : str
        Name of the column to sum (default 'ranking_params').

    Returns
    -------
    pd.DataFrame
        One row per distinct parameter combination, with `ranking_col`
        replaced by the summed value across all dataframes in df_list,
        sorted in descending order of `ranking_col`.
    """
    if not df_list:
        raise ValueError("df_list is empty.")

    param_cols_sets = [
        frozenset(c for c in df.columns if c.startswith('param_'))
        for df in df_list
    ]
    if len(set(param_cols_sets)) > 1:
        raise ValueError(
            "All dataframes in df_list must share the same set of "
            f"'param_*' columns. Found differing sets: {param_cols_sets}"
        )
    param_cols = sorted(param_cols_sets[0])

    combined = pd.concat(
        [df[param_cols + [ranking_col]] for df in df_list],
        ignore_index=True
    )
    combined[ranking_col] = combined[ranking_col].fillna(0)

    result = (
        combined
        .groupby(param_cols, as_index=False, dropna=False)[ranking_col]
        .sum()
        .sort_values(ranking_col, ascending=False)
        .reset_index(drop=True)
    )

    return result

def filter_best_ranking(df, ranking_col='ranking_params'):
    """
    Filter a dataframe (as produced by `sum_rankings`) to keep only the
    row(s) with the highest value of `ranking_col`. If there is a tie at
    the top, all tied rows are kept.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `ranking_col`.
    ranking_col : str
        Name of the ranking column to filter by (default 'ranking_params').

    Returns
    -------
    pd.DataFrame
        Subset of `df` containing only the row(s) with the maximum value
        of `ranking_col`.
    """
    if df.empty:
        return df.copy()

    max_val = df[ranking_col].max()
    result = df[df[ranking_col] == max_val].reset_index(drop=True)

    return result


# BORRAR
def filter_extremes(df_initial, dist_metric):
    """Shortlist the most promising parameter rows from a metrics dataframe.

    1. Primary condition. Keep every row satisfying
         (MPC_mean >= 0.8 OR MPE_mean > 0.8)
         AND (V-measure_mean >= 0.8 OR ARS_mean >= 0.8 OR AMI_mean >= 0.8)
    2. Fallback. If the primary condition yields fewer than 4 rows or more than 15, ignore it
       and instead take the top-3 rows (highest mean) of each of MPC, MPE,
       V-measure, ARS and AMI. Duplicates allowed here; dropped at the end.
    3. Always add the top-3 rows (lowest mean) of XBI and FKI.
    4. Distance metric:
         - 'euclidean' : also add the top-3 rows (highest mean) of FS and Silh.
         - otherwise   : drop the FS/Silh columns (mean and std) entirely.
    5. Drop duplicate parameter combinations.

    Every column access is guarded, so missing columns (e.g. MPE_mean before
    it is wired into METRICS) are skipped rather than raising.
    """
    df = df_initial.copy()

    def top(col, n=3, largest=True):
        """Top-n rows by `col`, NaNs excluded. Empty frame if col absent."""
        if col not in df.columns:
            return df.iloc[0:0]
        sub = df.dropna(subset=[col])
        return sub.nlargest(n, col) if largest else sub.nsmallest(n, col)

    # Step 1: primary condition
    cond_crispness = pd.Series(False, index=df.index)
    if 'MPC_mean' in df.columns:
        cond_crispness |= (df['MPC_mean'] >= 0.8)
    if 'MPE_mean' in df.columns:
        cond_crispness |= (df['MPE_mean'] > 0.8)

    cond_agreement = pd.Series(False, index=df.index)
    for c in ('V-measure_mean', 'ARS_mean', 'AMI_mean'):
        if c in df.columns:
            cond_agreement |= (df[c] >= 0.8)

    df_cond = df[cond_crispness & cond_agreement]

    # Step 2: fallback if fewer than 4 rows or too many rows in the dataframe (more than 10)
    if len(df_cond) >= 4 and len(df_cond) <=15:
        base = df_cond
    else:
        base = pd.concat(
            [top(c, 3, largest=True)
             for c in ('MPC_mean', 'MPE_mean', 'V-measure_mean', 'ARS_mean', 'AMI_mean')]
        )

    # Step 3: always add XBI / FKI (minimisers)
    parts = [base, top('XBI_mean', 3, largest=False), top('FKI_mean', 3, largest=False)]

    # Step 4: distance-metric-specific FS / Silh handling
    if dist_metric == 'euclidean':
        parts += [top('FS_mean', 3, largest=True), top('Silh_mean', 3, largest=True)]

    result = pd.concat(parts)

    if dist_metric != 'euclidean':
        result = result.drop(columns=[c for c in ('FS_mean', 'FS_std', 'Silh_mean', 'Silh_std')
                                      if c in result.columns])

    # Step 5: drop duplicate parameter combinations
    param_cols = [c for c in result.columns
                  if c.startswith('param_') or c.startswith('params_')]
    if param_cols:
        result = result.drop_duplicates(subset=param_cols)
    else:
        result = result.drop_duplicates()

    return result.reset_index(drop=True)

PARAM_SETTERS = {
    "param_m":    lambda mdl, v: setattr(mdl, "m", v),
    "param_lam":  lambda mdl, v: setattr(mdl, "lambda_", v),
    "param_m1":   lambda mdl, v: setattr(mdl, "m1", v),
    "param_m2":   lambda mdl, v: setattr(mdl, "m2", v),
    "param_lam1": lambda mdl, v: setattr(mdl, "lambda1", v),
    "param_lam2": lambda mdl, v: setattr(mdl, "lambda2", v),
}

def create_df_threshold_alpha(model, data, n_clusters_merge, dist_metric, size, n_boot, apply_bootstrap, df):
    """
    model: "FCM", "eFCM", "QFCM", "YFCM" or "eQFCM"
    df: the results of a dataframe with the variation of model params and its metric results from the bootstrap process
    Takes a results df from one model, picks the best-metric parameter rows
    (via filter_extremes), then re-runs that model with merge=True while
    sweeping the alpha values in GRIDS[size]["alpha"].
    df: already has the parameters that we want to play with. 
    """
    # BORRAR
    # if model == "FCM" or model == "eFCM":
    #     df_filtered = df
    # else:
    #     # df_filtered = filter_extremes(df, dist_metric) # BORRAR
    #     df_filtered = filter_best_ranking(df)

    df_filtered = df.copy()

    param_cols = [c for c in df_filtered.columns if c.startswith("param_")]
    list_best_value = [
        {p: row[p] for p in param_cols if pd.notna(row[p])}
        for _, row in df_filtered.iterrows()
    ]

    if model == "FCM":
        base_model = FCM(n_clusters=n_clusters_merge, merge=True, distance_metric=dist_metric)
    elif model == "eFCM":
        base_model = eFCM(n_clusters=n_clusters_merge, merge=True, distance_metric=dist_metric)
    elif model == "QFCM":
        base_model = QFCM(n_clusters=n_clusters_merge, merge=True, distance_metric=dist_metric)
    elif model == "YFCM":
        base_model = YFCM(n_clusters=n_clusters_merge, merge=True, distance_metric=dist_metric)
    elif model == "eQFCM":
        base_model = eQFCM(n_clusters=n_clusters_merge, merge=True, distance_metric=dist_metric)
    else:
        raise ValueError("model argument must be a STRING of the form 'FCM', 'eFCM', 'QFCM', 'YFCM' or 'eQFCM' ")

    results = []
    for best_value in list_best_value:
        for alpha in GRIDS[size]["alpha"]:
            m_i = copy.deepcopy(base_model)        

            # Apply every parameter for THIS row at once, then set alpha.
            for param_name, actual_param in best_value.items():
                if param_name in PARAM_SETTERS:
                    PARAM_SETTERS[param_name](m_i, actual_param)
            m_i.threshold_merge = alpha                     

            metrics = bootstrap_metrics(m_i, data[0], data[1], n_boot, apply_bootstrap, random_state=0)

            row = dict(best_value)                 
            row["param_alpha"] = alpha
            for name, val in metrics.items():
                if isinstance(val, tuple):         
                    row[f"{name}_mean"], row[f"{name}_std"] = val
                else:                              
                    row[name] = val
            results.append(row)

    return pd.DataFrame(results)







# _________________________________________________________________

# Validity-index helpers

# Fuzzy clustering metrics
def partition_coefficient(u):
    """PC = (1/N) sum_ij mu_ij^2.  Range [1/c, 1]; higher = crisper."""
    if u.shape[1] <= 1: # C
        return 0.0
    return float(np.sum(u ** 2) / u.shape[0])

def modified_partition_coefficient(u):
    """MPC = 1 - C/(C-1) * (1 - PC).  Range [0, 1]; higher = crisper."""
    PC = partition_coefficient(u)
    C = u.shape[1]
    if C <= 1:
        return 0.0
    return float(1 - C/(C-1) * (1 - PC))

def partition_entropy(u):
    """PE = -(1/N) sum_ij u_ij * ln(u_ij). Range [0, ln(c)]; lower = crisper."""
    N = u.shape[0]
    u = np.clip(u, 1e-10, None)
    entropy_sum = np.sum(u * np.log2(u))
    return float(-(1.0 / N) * entropy_sum)

def modified_partition_entropy(u):
    """1 - PE/log(C).  Range [0, 1]; higher = crisper."""
    PE = partition_entropy(u)
    C = u.shape[1]
    if C <= 1:
        return 0.0                        
    return float(1.0 - PE / np.log2(C))

def xie_beni(X, u, centers, distance_metric="sqeuclidean"):
    """XBI = [sum_ij mu^m ||x_i - v_j||^2]   /   [N * min_{i!=j} ||v_i - v_j||^2].
    Lower = better (compactness / separation)"""
    if centers.shape[0] < 2:
        return np.nan
    um = u ** 2
    numerator = np.sum(um * distance(X, u, centers, distance_metric=distance_metric))
    # for the denominator we just use sqeuclidean since the distance is between centers, not between points and centers
    cc = cdist(centers, centers, "sqeuclidean") 
    np.fill_diagonal(cc, np.inf)
    min_sep = cc.min()
    return float(numerator / (X.shape[0] * min_sep)) if min_sep > 0 else np.nan

def fukuyama_sugeno(X, u, centers, distance_metric="sqeuclidean"):
    """FKI = sum_ij mu^m ||x_i - v_j||^2  -  sum_ij mu^m ||v_j - xbar||^2.
    Lower = better (compactness minus separation)."""
    um = u ** 2
    compact = np.sum(um * distance(X, u, centers, distance_metric=distance_metric))
    xbar = X.mean(axis=0) # global centroid
    sep_j = np.sum((centers - xbar) ** 2, axis=1)          
    separation = np.sum(um * sep_j[None, :])
    return float(compact - separation)

def fuzzy_silhouette(X, u):
    """
    NOT IDEAL FOR GK CLUSTERING, since it uses Euclidean distance by default.
    FS = sum_i (u_ij - u_ij')^alpha * s_i  /  sum_i (u_ij - u_ij')^alpha

    where s_i is the crisp silhouette of sample i, and u_ij, u_ij' are the
    largest and second-largest membership values of row i of U.
    If the number of clusters is not minimum 2, it does not get calculated (NA)

    Higher = better
    """
    X = np.asarray(X)
    u = np.asarray(u, dtype=float)
    labels = np.argmax(u, axis=1)

    # Safe check
    k = len(np.unique(labels)) # number of different clusters
    if k < 2 or k > len(labels) - 1:
        return np.nan
    # BORRAR
    # try:
    #     s = silhouette_samples(X, labels, metric="euclidean")  
    # except Exception:
    #     return np.nan
    s = silhouette_samples(X, labels, metric="euclidean")  
    
    # Two largest memberships per row
    u_sorted = np.sort(u, axis=1)
    u_ij = u_sorted[:, -1]        
    u_ij_prime = u_sorted[:, -2]  

    weights = (u_ij - u_ij_prime) 
    return float(np.sum(weights * s) / np.sum(weights))




# Discrete clustering metrics (for comparison with true labels)
def safe_silhouette(X, labels):
    """
    NOT IDEAL FOR GK CLUSTERING, since it uses Euclidean distance by default.
    This function returns the mean Silhouette Coefficient over all samples.
    silhouette_score, but returns nan instead of raising when there are fewer than 2 (or more than N-1) distinct clusters.
    Range [-1, 1]; higher = crisper.
    """
    k = len(np.unique(labels))
    if k < 2 or k > len(labels) - 1:
        return np.nan
    
    return float(silhouette_score(X, labels))
    



# Distance helpers
def distance(X, u=None, centers=None, distance_metric="sqeuclidean", P=None):
        # return cdist(X, self.centers, metric="euclidean")
        if distance_metric == "sqeuclidean":
            return cdist(X, centers, metric="sqeuclidean")
        else:
            return distance_gk(X, u, centers)
            # return cdist(X, self.centers, metric="mahalanobis", VI=self.inverse_covariance(X, self.centers))

def distance_gk(X, u, centers, P=None, reg=1e-10, squared=True):
        """Mahalanobis distance (inner part of E-GK, before radius subtraction).
        
        A_i  = |P_i|^(1/n) * P_i^{-1}
        d_ik = sqrt( (x_k - v_i)^T A_i (x_k - v_i) )
        
        Returns
        -------
        D : (N, C) array of Mahalanobis distances (NON-squared).
        """
        X = np.asarray(X, dtype=float)
        N, n = X.shape
        C = centers.shape[0]

        if P is None:
            P = covariance_matrix(X, u, centers)        

        D = np.empty((N, C))
        for i in range(C):
            diff = X - centers[i]   # (N, n)
            Pi = P[i] + reg * np.eye(n)  
            vol = np.abs(np.linalg.det(Pi)) ** (1.0 / n)  # |P_i|^(1/n)
            A = vol * np.linalg.inv(Pi)       # norm matrix A_i
            maha2 = np.maximum(np.einsum("kj,jl,kl->k", diff, A, diff), 0.0)  # (x-v)^T A (x-v)
            D[:, i] = maha2 if squared else np.sqrt(maha2)  
        return D   # (N, C)

def covariance_matrix(X, u, centers):
    """Fuzzy covariance P_j for every cluster

        P_j = sum_i (u_ij^m) (x_i - v_j)(x_i - v_j)^T   / sum_i (u_ij^m)

    Returns P : (C, n, n) array, one covariance matrix per cluster
    """
    # N samples, n features, C clusters
    X = np.asarray(X, dtype=float) # (N,n) 
    centers = np.asarray(centers, dtype=float)
    C, n = centers.shape  # (C, n)
    um = np.asarray(u, dtype=float) ** 2 # (N, C)             

    P = np.empty((C, n, n))
    for j in range(C):
        diff = X - centers[j]                            
        w = um[:, j]                               
        denom = w.sum()  # sum of the whole column j
        denom = denom if denom > 1e-10 else 1e-10
        # P[j] = (diff * w[:, None]).T @ diff / denom     # BORRAR

        # diff[:, :, None] becomes a stack of N matrices, each sized n x 1
        # diff[:, None, :] becomes a stack of N matrices, each sized 1 x n
        # The @ operator multiplies them to create N matrices of shape (n, n)
        # w[:, None, None] reshapes weights to (N, 1, 1) 
        outer_products = diff[:, :, None] @ diff[:, None, :]
        weighted_products = w[:, None, None] * outer_products # each corresponding weight for the corresponding outer product
        P[j] = np.sum(weighted_products, axis=0) / denom  # collapse the N dimension
    
    return P

