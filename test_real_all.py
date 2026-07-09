from sklearn.datasets import (load_iris, load_wine, load_breast_cancer, fetch_openml)

from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd
import time

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
    start_time = time.perf_counter()

    REAL_DATASETS = {
        # "wine": {"dataset_x_y": load_wine(return_X_y=True), "C_official": 3, "C_upper": 10},
        # "breast": {"dataset_x_y": load_breast_cancer(return_X_y=True), "C_official": 2, "C_upper": 10},
        # "iris": {"dataset_x_y": load_iris(return_X_y=True), "C_official": 3, "C_upper": 10},
        "heart": {"dataset_x_y": fetch_openml(name="heart", version=1, return_X_y=True), "C_official": 2, "C_upper": 10},
        "bupa": {"dataset_x_y": fetch_openml(name="liver-disorders", version=1, return_X_y=True), "C_official": 2, "C_upper": 10},
        "glass": {"dataset_x_y": fetch_openml(name="glass", version=1, return_X_y=True), "C_official": 6, "C_upper": 15},
        "sonar": {"dataset_x_y": fetch_openml(name="sonar", version=1, return_X_y=True), "C_official": 2, "C_upper": 10},
    }

    # BORRAR
    for data_name in REAL_DATASETS:
        data_used = REAL_DATASETS[data_name]["dataset_x_y"]
        # data_scaled = (StandardScaler().fit_transform(np.asarray(data_used[0], dtype=float)), data_used[1])
        n_real_clusters = REAL_DATASETS[data_name]["C_official"]
        n_upper_clusters = REAL_DATASETS[data_name]["C_upper"]
        print(f"data name: {data_name}, data used: {type(data_used)}, n real clusters: {n_real_clusters}, n upper clusters: {n_upper_clusters}")


    folder_name_euc = "Real_results_Euclidean"
    distance_euc = "euclidean"

    folder_name_mah = "Real_results_Mahalanobis"
    distance_mah = "mahalanobis"

    for data_name in REAL_DATASETS:
        # _________________________________________________________________________________________________________________
        #
        # DATASET
        # _________________________________________________________________________________________________________________

        data_used = REAL_DATASETS[data_name]["dataset_x_y"]
        # data_name = "iris"
        data_scaled = (StandardScaler().fit_transform(np.asarray(data_used[0], dtype=float)), data_used[1])
        n_real_clusters = REAL_DATASETS[data_name]["C_official"]
        n_upper_clusters = REAL_DATASETS[data_name]["C_upper"]
        
        # _________________________________________________________________________________________________________________
        #
        #    FCM
        # _________________________________________________________________________________________________________________

        algorithm_used = "FCM"
        file_name_algorithm = algorithm_used.lower()

        #  ----- Eucdlidean -------

        df_fcm_false = create_df_params(algorithm_used, data_scaled, n_real_clusters, distance_euc, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_scaled, n_upper_clusters, distance_euc, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mTrue.csv")

        #  ----- Mahalanobis -------

        df_fcm_false = create_df_params(algorithm_used, data_used, n_real_clusters, distance_mah, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_used, n_upper_clusters, distance_mah, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mTrue.csv")

        # _________________________________________________________________________________________________________________
        #
        #    eFCM
        # _________________________________________________________________________________________________________________

        algorithm_used = "eFCM"
        file_name_algorithm = algorithm_used.lower()

        #  ----- Euclidean -------

        df_fcm_false = create_df_params(algorithm_used, data_scaled, n_real_clusters, distance_euc, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_scaled, n_upper_clusters, distance_euc, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mTrue.csv")

        #  ----- Mahalanobis -------

        df_fcm_false = create_df_params(algorithm_used, data_used, n_real_clusters, distance_mah, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_used, n_upper_clusters, distance_mah, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mTrue.csv")

        # _________________________________________________________________________________________________________________
        #
        #    QFCM
        # _________________________________________________________________________________________________________________

        algorithm_used = "QFCM"
        file_name_algorithm = algorithm_used.lower()

        #  ----- Euclidean -------

        df_fcm_false = create_df_params(algorithm_used, data_scaled, n_real_clusters, distance_euc, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_scaled, n_upper_clusters, distance_euc, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mTrue.csv")

        #  ----- Mahalanobis -------

        df_fcm_false = create_df_params(algorithm_used, data_used, n_real_clusters, distance_mah, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_used, n_upper_clusters, distance_mah, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mTrue.csv")

        # _________________________________________________________________________________________________________________
        #
        #    YFCM
        # _________________________________________________________________________________________________________________

        algorithm_used = "YFCM"
        file_name_algorithm = algorithm_used.lower()

        #  ----- Eucdlidean -------

        df_fcm_false = create_df_params(algorithm_used, data_scaled, n_real_clusters, distance_euc, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_scaled, n_upper_clusters, distance_euc, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mTrue.csv")

        #  ----- Mahalanobis -------

        df_fcm_false = create_df_params(algorithm_used, data_used, n_real_clusters, distance_mah, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_used, n_upper_clusters, distance_mah, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mTrue.csv")

        # _________________________________________________________________________________________________________________
        #
        #    eQFCM
        # _________________________________________________________________________________________________________________

        algorithm_used = "eQFCM"
        file_name_algorithm = algorithm_used.lower()

        #  ----- Euclidean -------

        df_fcm_false = create_df_params(algorithm_used, data_scaled, n_real_clusters, distance_euc, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_scaled, n_upper_clusters, distance_euc, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mTrue.csv")

        #  ----- Mahalanobis -------

        df_fcm_false = create_df_params(algorithm_used, data_used, n_real_clusters, distance_mah, "large", 10, False)
        df_fcm_false.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mFalse.csv")
        df_fcm_true = create_df_threshold_alpha(algorithm_used, data_used, n_upper_clusters, distance_mah, "large", 10, False, df_fcm_false)
        df_fcm_true.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mTrue.csv")



    end_time = time.perf_counter()
    runtime = end_time - start_time
    print("-----------------------------------------")
    print(f"Execution time: {runtime:.6f} seconds") 
    print("-----------------------------------------")
