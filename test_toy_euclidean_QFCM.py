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

    folder_name = "Toy_results_Euclidean"
    distance_used = "euclidean"


    # _________________________________________________________________________________________________________________
    #
    # DATASETS EUCLIDEAN
    # _________________________________________________________________________________________________________________

    n = 1500
    rng = np.random.RandomState(170)

    blobs1 = datasets.make_blobs(n_samples=n, random_state=170)
    blobs2= datasets.make_blobs(n_samples=[200, 200, 100, 200],
                centers=np.array([[1, 1], [8, 2], [6, 8], [3, 9]]), cluster_std=0.5, random_state=0)
    varied1 = datasets.make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    varied2 = datasets.make_blobs(n_samples=n, centers=np.array([[-5, -5], [16, 5], [11, 13], [5, 9]]), cluster_std=[1.0, 2.5, 0.7, 2.0],  random_state=0)

    blobs1_scaled = (StandardScaler().fit_transform(np.asarray(blobs1[0], dtype=float)), blobs1[1])
    blobs2_scaled = (StandardScaler().fit_transform(np.asarray(blobs2[0], dtype=float)), blobs2[1])
    varied1_scaled = (StandardScaler().fit_transform(np.asarray(varied1[0], dtype=float)), varied1[1])
    varied2_scaled = (StandardScaler().fit_transform(np.asarray(varied2[0], dtype=float)), varied2[1])

    TOY = {
        "blobs1": {"data_scaled": blobs1_scaled, "C_official": 3, "C_upper": 10},
        "blobs2": {"data_scaled": blobs2_scaled, "C_official": 4, "C_upper": 10},
        "varied1": {"data_scaled": varied1_scaled, "C_official": 3, "C_upper": 10},
        "varied2": {"data_scaled": varied2_scaled, "C_official": 4, "C_upper": 10}
    }


    # _________________________________________________________________________________________________________________
    #
    #    FCM
    # _________________________________________________________________________________________________________________

    algorithm_used = "QFCM"
    file_name_algorithm = algorithm_used.lower()

    #  ----- BLOBS 1 -------

    data_used = "blobs1"
 
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data_scaled"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data_scaled"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")

    #  ----- BLOBS 2 -------

    data_used = "blobs2"
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data_scaled"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data_scaled"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")

    #  ----- VARIED 1 -------

    data_used = "varied1"
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data_scaled"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data_scaled"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")
    
    #  ----- VARIED 2 -------

    data_used = "varied2"
 
    df_fcm_false = create_df_params(algorithm_used, TOY[data_used]["data_scaled"], TOY[data_used]["C_official"], distance_used, "large", 30, True)
    df_fcm_false.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mFalse.csv")
    df_fcm_true = create_df_threshold_alpha(algorithm_used,  TOY[data_used]["data_scaled"], TOY[data_used]["C_upper"], distance_used, "large", 30, True, df_fcm_false)
    df_fcm_true.to_csv(f"{folder_name}/{file_name_algorithm}_{data_used}_mTrue.csv")




 
    




