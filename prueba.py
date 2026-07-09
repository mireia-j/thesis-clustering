import numpy as np
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer, fetch_openml)
from utils import plot_results, create_df_params, create_df_threshold_alpha, bootstrap_metrics, filter_extremes, load_openml_numpy

# REAL_DATASETS = {
#         "wine": {"dataset_x_y": load_wine(return_X_y=True), "C_official": 3, "C_upper": 10},
#         "breast": {"dataset_x_y": load_breast_cancer(return_X_y=True), "C_official": 2, "C_upper": 10},
#         "iris": {"dataset_x_y": load_iris(return_X_y=True), "C_official": 3, "C_upper": 10},
#         "heart": {"dataset_x_y": fetch_openml(name="heart", version=1, return_X_y=True), "C_official": 2, "C_upper": 10},
#         "bupa": {"dataset_x_y": fetch_openml(name="liver-disorders", version=1, return_X_y=True), "C_official": 2, "C_upper": 10},
#         "glass": {"dataset_x_y": fetch_openml(name="glass", version=1, return_X_y=True), "C_official": 6, "C_upper": 15},
#         "sonar": {"dataset_x_y": fetch_openml(name="sonar", version=1, return_X_y=True), "C_official": 2, "C_upper": 10},
#     }



REAL_DATASETS = {
    "wine": {
        "dataset_x_y": load_wine(return_X_y=True),
        "C_official": 3,
        "C_upper": 10,
    },
    "breast": {
        "dataset_x_y": load_breast_cancer(return_X_y=True),
        "C_official": 2,
        "C_upper": 10,
    },
    "iris": {
        "dataset_x_y": load_iris(return_X_y=True),
        "C_official": 3,
        "C_upper": 10,
    },
    "heart": {
        "dataset_x_y": load_openml_numpy("heart"),
        "C_official": 2,
        "C_upper": 10,
    },
    "bupa": {
        "dataset_x_y": load_openml_numpy("liver-disorders"),
        "C_official": 2,
        "C_upper": 10,
    },
    "glass": {
        "dataset_x_y": load_openml_numpy("glass"),
        "C_official": 6,
        "C_upper": 15,
    },
    "sonar": {
        "dataset_x_y": load_openml_numpy("sonar"),
        "C_official": 2,
        "C_upper": 10,
    },
}

for x in REAL_DATASETS.keys():
    print(f"Data 0: {type(REAL_DATASETS[x]["dataset_x_y"][0])}, Data 1: {type(REAL_DATASETS[x]["dataset_x_y"][1])}")
    print(f"Shape 0: {REAL_DATASETS[x]["dataset_x_y"][0].shape}, Data 1: {REAL_DATASETS[x]["dataset_x_y"][1].shape}")

