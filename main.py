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



