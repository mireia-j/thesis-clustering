import numpy as np

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])
    


    X2 = np.array([[0.0, 0.0]])

    print(X2.shape)
