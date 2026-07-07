import numpy as np

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])
    


    X2 = np.array([[0.0, 0.5], 
                   [1.0, 1.5], 
                   [2.0, 2.5], 
                   [3.0, 3.5], 
                   [4.0, 4.5], 
                   [5.0, 5.5], 
                   [6.0, 6.5], 
                   [7.0, 7.5], 
                   [8.0, 8.5], 
                   [9.0, 9.5]])
    
    norm = X2@X2.T
    

    print(np.shape(norm))

    w = np.asarray(X2).shape[0]
    print(w)

results = []
model_fcm_try = FCM(10, merge=True, random_state=0, distance_metric='mahalanobis', random_state=0, m=2.0)
for param_alpha in ["original", "new", 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
  m_i = copy.deepcopy(model)
  m_i.m = param_m
  m_i.lambda_ = param_lam
  metrics = bootstrap_metrics(m_i, data[0], data[1],
                                        n_boot=100, random_state=0)

        row = {"param_m": param_m, "param_lam": param_lam}
        for metric, (mean_val, std_val) in metrics.items():
          row[f"{metric}_mean"] = mean_val
          row[f"{metric}_std"] = std_val
        results.append(row)