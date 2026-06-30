import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

class BaseFuzzyClustering:

    def __init__(self, n_clusters=10, m=2, max_iter=150, error=1e-5,
                 random_state=42, merge=False, epsilon_1=0.01):
        self.n_clusters = n_clusters
        self.n_clusters_init = n_clusters  # starting M^(0)
        self.m = m                  # fuzzifier (>1)
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        
        # If merge=True, then we don't specify the exact number of clusters needed
        # We just set an upper bound of n_clusters. The algorithm will merge clusters based on similarity.
        self.merge = merge
        self.epsilon_1 = epsilon_1
        self.prev_max_sim = 1.0   # S^(0)

        self.u = None               # membership matrix, shape (N, C)
        self.centers = None         # cluster centers, shape (C, n_features)


    def fit(self, X):
        pass

    def _init_membership(self, n_samples):
        rng = np.random.RandomState(self.random_state)
        u = rng.rand(n_samples, self.n_clusters)
        return u / u.sum(axis=1, keepdims=True)

    def next_centers(self, X, u): 
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T
    
    def update_membership(self, D):
        raise NotImplementedError

    def distance(self, X):
        return cdist(X, self.centers, metric="euclidean")
    
    def distance_squared(self, X):
        return cdist(X, self.centers, metric="sqeuclidean")
    
    # For the merging of clusters
    @staticmethod
    def most_similar_pair(u):
        """Fuzzy-inclusion similarity (eq. 10); returns ((i*, j*), sim)."""
        C = u.shape[1]
        col_sum = u.sum(axis=0)
        best_pair, best_sim = (0, 1), -np.inf
        for i in range(C):
            for j in range(i + 1, C):
                num = np.minimum(u[:, i], u[:, j]).sum()
                den = min(col_sum[i], col_sum[j])
                sim = num / den if den > 0 else 0.0
                if sim > best_sim:
                    best_sim, best_pair = sim, (i, j)
        return best_pair, best_sim

    def merge_clusters(self, u): # self.previous_max_sim
        """Attempt ONE merge. Returns (u, merged).

        When a merge happens, u loses a column and self.n_clusters is kept
        in sync so the next next_centers()/update_* call uses the new size.
        """
        merged = False
        C = u.shape[1]
        if self.merge and C > 1:
            (i_, j_), max_sim = self.most_similar_pair(u)
            if abs(max_sim - self.prev_max_sim) < self.epsilon_1:
                threshold = 1.0 / (C - 1)
                if max_sim > threshold:
                    u = u.copy()
                    u[:, i_] = u[:, i_] + u[:, j_]   # fold j* into i*
                    u = np.delete(u, j_, axis=1)
                    self.n_clusters = C - 1
                    merged = True
                else:
                    self.on_no_merge()             
            self.prev_max_sim = max_sim
        return u, merged

    def on_no_merge(self):
        """eFCM overrides to enlarge prototypes (beta)"""
        pass
    

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X)
        return np.argmax(u, axis=-1)
    
    def labels(self):
        """Hard labels for the data passed to fit()"""
        if self.u is None:
            raise ValueError("Not fitted yet. Call fit() first.")
        return np.argmax(self.u, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels()

class BaseKanzawa(BaseFuzzyClustering):
    """Shared base for the QFCM / YFCM / eQFCM family (Kanzawa, 2024).

    These algorithms all share Algorithm 1's alternating-optimisation loop
    and differ only in how the cluster-size controller `alpha` (Step 3) and
    the membership `U` (Step 4) are updated. Subclasses implement those two
    methods; everything else lives here.

    Parameter mapping to the paper:
        n_clusters -> C
        m          -> B-parameter / fuzzifier (> 1)
        lambda_    -> regularizing parameter lambda (> 0)
        error      -> stopping threshold on ||U_new - U_old||

    Note: `alpha` is a *learned* attribute set during fit(), not a
    constructor argument.
    """

    def __init__(self, n_clusters=10, m=2, lambda_=1.0, max_iter=150,
                 error=1e-5, random_state=42, merge=False, epsilon_1=0.01):
        super().__init__(n_clusters=n_clusters, m=m, max_iter=max_iter,
                         error=error, random_state=random_state, merge=merge, epsilon_1=epsilon_1)
        self.lambda_ = lambda_
        self.alpha = None          # cluster size controllers, shape (C,)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        u = self._init_membership(X.shape[0])                
        self.alpha = np.full(self.n_clusters, 1.0 / self.n_clusters)
        for _ in range(self.max_iter):
            u_old = u
            self.centers = self.next_centers(X, u)            # Step 2
            D = self.distance_squared(X)  # d_ik
            self.alpha = self.update_alpha(u, D)              # Step 3
            u = self.update_membership(D)                     # Step 4
            u, merged = self.merge_clusters(u)               # optional merge
            if not merged and u.shape == u_old.shape:         # Step 5
                if np.linalg.norm(u - u_old) < self.error:
                    break
        self.u = u
        return self

    # Subclasses must implement these two.
    def update_alpha(self, u, D):
        raise NotImplementedError

    def update_membership(self, D):
        raise NotImplementedError
    

def plot_2(result_fuzzy, X, show_centers=True):
    """Fit on X (must have exactly 2 features) and scatter the points
    coloured by their assigned cluster."""
    if result_fuzzy.centers.shape[1] != 2:
        raise ValueError(
            f"plot() needs 2-D data of shape (n_samples, 2); got {X.shape}."
        )
     
    if result_fuzzy.centers is None:
        raise ValueError("Not fitted yet. Call fit() first.")

    labels = result_fuzzy.labels()        
    if len(labels) != X.shape[0]:
        raise ValueError("Number of samples in X does not match number labels.")

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15, alpha=0.7)

    if show_centers:
        ax.scatter(result_fuzzy.centers[:, 0], result_fuzzy.centers[:, 1],
                    c="red", marker="X", s=200, edgecolors="black",
                    label="centers")
        ax.legend()

    ax.set_title(f"{type(result_fuzzy).__name__} — {result_fuzzy.n_clusters} clusters")
    plt.show()
    return ax