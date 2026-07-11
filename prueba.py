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
        "wine": {"dataset_x_y": load_wine(return_X_y=True), "C_official": 3, "C_upper": 10},
        "breast": {"dataset_x_y": load_breast_cancer(return_X_y=True), "C_official": 2, "C_upper": 10},
        "iris": {"dataset_x_y": load_iris(return_X_y=True), "C_official": 3, "C_upper": 10},
        "heart": {"dataset_x_y": load_openml_numpy("heart"), "C_official": 2, "C_upper": 10},
        "bupa": {"dataset_x_y": load_openml_numpy("liver-disorders"), "C_official": 2, "C_upper": 10},
        "glass": {"dataset_x_y": load_openml_numpy("glass"), "C_official": 6, "C_upper": 15},
        "sonar": {"dataset_x_y": load_openml_numpy("sonar"), "C_official": 2, "C_upper": 10}
    }


    # BORRAR
    for data_name in REAL_DATASETS:
        print("data ", data_name)
        data_used = REAL_DATASETS[data_name]["dataset_x_y"]
        size_data0 = REAL_DATASETS[data_name]["dataset_x_y"][0]
        size_data1 = REAL_DATASETS[data_name]["dataset_x_y"][1]
        vals, counts = np.unique(REAL_DATASETS[data_name]["dataset_x_y"][1], return_counts=True)
        print(f" dtype=  n_unique={len(vals)}")
        print("       ", dict(zip(vals.tolist(), counts.tolist())))
        n_real_clusters = REAL_DATASETS[data_name]["C_official"]
        n_upper_clusters = REAL_DATASETS[data_name]["C_upper"]
        # print(f"data name: {data_name}, data used: {type(data_used)}, n real clusters: {n_real_clusters}, n upper clusters: {n_upper_clusters}")
