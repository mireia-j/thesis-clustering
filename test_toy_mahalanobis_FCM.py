import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# BORRAR
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, silhouette_samples, homogeneity_completeness_v_measure)

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

from utils import plot_results, create_df_params, create_df_threshold_alpha, bootstrap_metrics, filter_extremes


if __name__ == "__main__":

    folder_name = "Toy_results_Mahalanobis"
    distance_used = "mahalanobis"


    # _________________________________________________________________________________________________________________
    #
    # DATASETS MAHALANOBIS
    # _________________________________________________________________________________________________________________

    n = 1500
    rng = np.random.RandomState(170)

    Xa, y_aniso = datasets.make_blobs(n_samples=n, random_state=170)
    aniso = (Xa @ [[0.6, -0.6], [-0.4, 0.8]], y_aniso)              # Dataset 1 

    def make_elliptical_3(n_samples=500, random_state=0):
        rng = np.random.RandomState(random_state)
        centers = np.array([[2.0, 8.0], [8.0, 2.0], [6.0, 6.0]])

        # a rotated-ellipse covariance from (sx, sy, angle-in-degrees)
        def cov(sx, sy, deg):
            t = np.deg2rad(deg)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            return R @ np.diag([sx**2, sy**2]) @ R.T

        covs = [cov(1.4, 0.5,  30),    # long, tilted +30 deg
                cov(1.3, 0.4, -20),    # long, tilted -20 deg
                cov(1.5, 0.6,  75)]    # long, nearly vertical

        sizes = [167, 167, 166]

        X, y = [], []
        for k, (mu, C, s) in enumerate(zip(centers, covs, sizes)):
            X.append(rng.multivariate_normal(mu, C, size=s))
            y.append(np.full(s, k))
        return np.vstack(X), np.concatenate(y)

    X_elliptical_3, y_elliptical_3 = make_elliptical_3()
    elliptical = (X_elliptical_3, y_elliptical_3)              # Dataset 2
    
    # From EQFCM paper 2000

    def _segment(p0, p1, n, noise, rng):
        """n noisy points along the segment p0->p1 (isotropic Gaussian thickness)."""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        t = rng.uniform(0.0, 1.0, size=n)[:, None]
        return p0 + t * (p1 - p0) + rng.normal(0.0, noise, size=(n, 2))

    def make_lines(segments, rng):
        """segments: list of (p0, p1, n_points, noise) -> (X, labels), 1 label per segment."""
        X, y = [], []
        for lbl, (p0, p1, n, noise) in enumerate(segments):
            X.append(_segment(p0, p1, n, noise, rng)); y.append(np.full(n, lbl))
        return np.vstack(X), np.concatenate(y)

    def _letter(strokes, x0, w, h, y0, npts, noise):
        return [((x0 + a[0]*w, y0 + a[1]*h), (x0 + b[0]*w, y0 + b[1]*h), npts, noise)
                for (a, b) in strokes]

    # ---- Figure 10: five noisy lines (varied length / density / thickness) ----
    rng = np.random.RandomState(0)
    fig10 = [ #  p0            p1            n    noise
        ((0.05, 0.15), (0.05, 0.80), 120, 0.015),   # near-vertical, left
        ((0.10, 0.10), (1.00, 1.00), 200, 0.020),   # long main diagonal
        ((0.20, 0.90), (0.95, 0.30), 150, 0.025),   # anti-diagonal, thicker
        ((0.25, 0.08), (1.15, 0.18), 100, 0.012),   # shallow, bottom
        ((0.55, 0.35), (0.75, 1.00),  80, 0.030),   # short, steep, sparse+thick
    ]
    five_lines = make_lines(fig10, rng)             # Dataset 3

    # ---- Figure 11: four letters x three strokes = 12 clusters ----------------
    N = [((0,0),(0,1)), ((0,1),(1,0)), ((1,0),(1,1))]
    Z = [((0,1),(1,1)), ((1,1),(0,0)), ((0,0),(1,0))]
    H = [((0,0),(0,1)), ((1,0),(1,1)), ((0,0.5),(1,0.5))]
    K = [((0,0),(0,1)), ((0,0.5),(1,1)), ((0,0.5),(1,0))]

    rng = np.random.RandomState(1)
    letters = [ # strokes, x0,  npts, noise   -> different density & thickness
        (N, 0.30,  95, 0.018),
        (Z, 0.80,  70, 0.025),
        (H, 1.30, 115, 0.015),
        (K, 1.80,  60, 0.030),
    ]
    fig11 = []
    for strokes, x0, npts, noise in letters:
        fig11 += _letter(strokes, x0, w=0.35, h=1.0, y0=0.0, npts=npts, noise=noise)
    four_letters = make_lines(fig11, rng)           # Dataset 4

    # BORRAR?
    def line_1(x):
        y = (1-4*x)/5
        return np.column_stack((x, y))

    def line_2(x):
        y = 2*x
        return np.column_stack((x, y))

    x_values = np.random.uniform(-1, 1, 100)
    line_eq_1 = line_1(x_values)
    line_eq_2 = line_2(x_values)
    X_crossing = np.vstack((line_eq_1, line_eq_2))
    y_crossing = np.array([0] * 100 + [1] * 100)
    crossing_lines = (X_crossing, y_crossing)        # Dataset 5??

    TOY = {
        "aniso": {"data": aniso, "C_official": 3, "C_upper": 10},
        "elliptical": {"data": elliptical, "C_official": 3, "C_upper": 10},
        "five_lines": {"data": five_lines, "C_official": 5, "C_upper": 15},
        "four_letters": {"data": four_letters, "C_official": 12, "C_upper": 20},
        "crossing_lines": {"data": crossing_lines, "C_official": 2, "C_upper": 10},
    }


    # _________________________________________________________________________________________________________________
    #
    #    FCM
    # _________________________________________________________________________________________________________________

    algorithm_used = "FCM"
    file_name_algorithm = algorithm_used.lower()

    #  ----- ANISO -------

    data_used = "aniso"
 
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")

    #  ----- ELLIPTICAL -------

    data_used = "elliptical"
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")

    #  ----- FIVE LINES -------

    data_used = "five_lines"
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")
    
    #  ----- FOUR LETTERS -------

    data_used = "four_letters"
 
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")