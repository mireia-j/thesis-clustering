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

from utils import create_df_params, create_df_threshold_alpha, load_openml_numpy, sum_rankings, rank_by_vmeasure, filter_best_ranking

if __name__ == "__main__":
    start_time = time.perf_counter()

    REAL_DATASETS = {
        "breast": {"dataset_x_y": load_breast_cancer(return_X_y=True), "C_official": 2, "C_upper": 10},
        "iris": {"dataset_x_y": load_iris(return_X_y=True), "C_official": 3, "C_upper": 10},
        "wine": {"dataset_x_y": load_wine(return_X_y=True), "C_official": 3, "C_upper": 10},
        "bupa": {"dataset_x_y": load_openml_numpy("liver-disorders"), "C_official": 2, "C_upper": 10},
        "glass": {"dataset_x_y": load_openml_numpy("glass"), "C_official": 6, "C_upper": 15},
        "heart": {"dataset_x_y": load_openml_numpy("heart"), "C_official": 2, "C_upper": 10},
        "seeds": {"dataset_x_y": load_openml_numpy("seeds"), "C_official": 3, "C_upper": 10},
        "sonar": {"dataset_x_y": load_openml_numpy("sonar"), "C_official": 2, "C_upper": 10}
    }

    ALGORITHMS = [
        "FCM",
        "eFCM",
        "QFCM",
        "YFCM",
        "eQFCM"
                  ]
    
    n_boot = 30
    apply_bootstrap = True

    folder_name_euc = "Real_results_Euclidean"
    distance_euc = "euclidean"

    folder_name_mah = "Real_results_Mahalanobis"
    distance_mah = "mahalanobis"

    folder_name_best = "Real_results_BEST_COMBO"

    for algorithm_used in ALGORITHMS:
        file_name_algorithm = algorithm_used.lower()

        # ---------------------- Trying m and lambda ----------------------
        df_params_m_lam_EUC = []
        df_params_m_lam_MAH = []

        for data_name in REAL_DATASETS:

            data_used = REAL_DATASETS[data_name]["dataset_x_y"]
            data_scaled = (StandardScaler().fit_transform(np.asarray(data_used[0], dtype=float)), data_used[1])
            n_real_clusters = REAL_DATASETS[data_name]["C_official"]

            #  ----- Euclidean -------
            df_fcm_false_EUC = create_df_params(algorithm_used, data_scaled, n_real_clusters, distance_euc, "large", n_boot, apply_bootstrap)
            df_fcm_false_EUC.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mFalse.csv")
            ranking_fcm_false_EUC = rank_by_vmeasure(df_fcm_false_EUC, 10)
            df_params_m_lam_EUC.append(ranking_fcm_false_EUC)

            #  ----- Mahalanobis -------
            df_fcm_false_MAH = create_df_params(algorithm_used, data_used, n_real_clusters, distance_mah, "large", n_boot, apply_bootstrap)
            df_fcm_false_MAH.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mFalse.csv")
            ranking_fcm_false_MAH = rank_by_vmeasure(df_fcm_false_MAH, 10)
            df_params_m_lam_MAH.append(ranking_fcm_false_MAH)

        # BORRAR
        df_best_m_lam_EUC = filter_best_ranking(sum_rankings(df_params_m_lam_EUC))
        df_best_m_lam_MAH = filter_best_ranking(sum_rankings(df_params_m_lam_MAH))


        # ---------------------- Trying threshold alpha ----------------------

        # 1. CREATE A LIST WITH ALL THE DATASETS FOR ALPHA
        df_params_alpha_EUC_ranking = []
        df_params_alpha_EUC_metrics = []
        df_params_alpha_MAH_ranking = []
        df_params_alpha_MAH_metrics = []

        for data_name in REAL_DATASETS:

            data_used = REAL_DATASETS[data_name]["dataset_x_y"]
            data_scaled = (StandardScaler().fit_transform(np.asarray(data_used[0], dtype=float)), data_used[1])
            n_upper_clusters = REAL_DATASETS[data_name]["C_upper"]

            #  ----- Euclidean -------
            df_fcm_true_EUC = create_df_threshold_alpha(algorithm_used, data_scaled, n_upper_clusters, distance_euc, "large", n_boot, apply_bootstrap, df_best_m_lam_EUC)
            df_fcm_true_EUC.to_csv(f"{folder_name_euc}/{file_name_algorithm}_{data_name}_mTrue.csv")
            new_cols_EUC = pd.DataFrame({
                "Dist": distance_euc,
                "algorithm": algorithm_used,
                "dataset": data_name
            }, index=df_fcm_true_EUC.index)
            df_fcm_true_EUC_extra_columns = pd.concat([new_cols_EUC, df_fcm_true_EUC], axis=1)
            df_params_alpha_EUC_metrics.append(df_fcm_true_EUC_extra_columns)
            ranking_fcm_true_EUC = rank_by_vmeasure(df_fcm_true_EUC, 3)
            df_params_alpha_EUC_ranking.append(ranking_fcm_true_EUC)

            #  ----- Mahalanobis -------
            df_fcm_true_MAH = create_df_threshold_alpha(algorithm_used, data_used, n_upper_clusters, distance_mah, "large", n_boot, apply_bootstrap, df_best_m_lam_MAH)
            df_fcm_true_MAH.to_csv(f"{folder_name_mah}/{file_name_algorithm}_{data_name}_mTrue.csv")
            new_cols_MAH = pd.DataFrame({
                "Dist": distance_mah,
                "algorithm": algorithm_used,
                "dataset": data_name
            }, index=df_fcm_true_MAH.index)
            df_fcm_true_MAH_extra_columns = pd.concat([new_cols_MAH, df_fcm_true_MAH], axis=1)
            df_params_alpha_MAH_metrics.append(df_fcm_true_MAH_extra_columns)
            ranking_fcm_true_MAH = rank_by_vmeasure(df_fcm_true_MAH, 3)
            df_params_alpha_MAH_ranking.append(ranking_fcm_true_MAH)

        # 2. GET THE OPTIMAL COMBINATION OF alpha, m, lambda
        df_best_alpha_EUC = filter_best_ranking(sum_rankings(df_params_alpha_EUC_ranking))
        df_best_alpha_MAH = filter_best_ranking(sum_rankings(df_params_alpha_MAH_ranking))

        # 3. FILTER IN THAT HAS ALL THE ALPHA COMBINATIONS AND METRICS BY THIS BEST COMBINATION

        #  ----- Euclidean -------
        param_cols_EUC = [c for c in df_best_alpha_EUC.columns if c.startswith("param_")]
        valid_combinations = set(
            map(tuple, df_best_alpha_EUC[param_cols_EUC].to_numpy())
        )
        filtered_dfs_EUC = []
        for df in df_params_alpha_EUC_metrics:
            mask = df[param_cols_EUC].apply(tuple, axis=1).isin(valid_combinations)
            filtered_dfs_EUC.append(df[mask])
        final_combination_EUC = pd.concat(filtered_dfs_EUC, ignore_index=True)
        final_combination_EUC.to_csv(f"{folder_name_best}/euc_{file_name_algorithm}.csv")

        #  ----- Mahalanobis -------
        param_cols_MAH = [c for c in df_best_alpha_MAH.columns if c.startswith("param_")]
        valid_combinations = set(
            map(tuple, df_best_alpha_MAH[param_cols_MAH].to_numpy())
        )
        filtered_dfs_MAH = []
        for df in df_params_alpha_MAH_metrics:
            mask = df[param_cols_MAH].apply(tuple, axis=1).isin(valid_combinations)
            filtered_dfs_MAH.append(df[mask])
        final_combination_MAH = pd.concat(filtered_dfs_MAH, ignore_index=True)
        final_combination_MAH.to_csv(f"{folder_name_best}/mah_{file_name_algorithm}.csv")

        # print("\nEuclidean:")
        # print(final_combination_EUC.iloc[:, [2, 3, 4, 8, 9, 30, 31]])
        # print("\nMahala:")
        # print(final_combination_MAH.iloc[:, [2, 3, 4, 8, 9, 30, 31]])


    