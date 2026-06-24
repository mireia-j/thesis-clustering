import numpy as np
from sklearn.metrics import adjusted_rand_score

from fuzzyclust.base import BaseFuzzyClustering, plot_2

class FCM(BaseFuzzyClustering):
    """Fuzzy C-Means."""
    def fit(self, X):
        u = self._init_membership(X.shape[0])
        for _ in range(self.max_iter):
            u_old = u.copy()
            self.centers = self.next_centers(X, u)   # update centers from u
            u = self.next_u(X)                  # update u from centers
            if np.linalg.norm(u - u_old) < self.error:
                break
        self.u = u
        return self
    

# ----------------------------------------------------------------------
# Sketch of where agglomerative hierarchical clustering plugs in.
# You implement fit() however the algorithm works (build the merge tree,
# cut it at n_clusters, fill in u). If you also set self.centers as the
# membership-weighted means, then predict()/soft_predict() work for new
# points too, with no extra code.
#
# class FuzzyAgglomerative(BaseFuzzyClustering):
#     def fit(self, X):
#         # 1. start from N (or some initial) clusters
#         # 2. repeatedly merge the closest pair until n_clusters remain
#         # 3. build the membership matrix u from the final assignment
#         self.u = ...
#         # optional, enables predict() on unseen data:
#         self.centers = (X.T @ (self.u ** self.m)
#                         / (self.u ** self.m).sum(axis=0)).T
#         return self
# ----------------------------------------------------------------------


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100, 2) - 4,
                   rng.randn(100, 2),
                   rng.randn(100, 2) + 4])
    
    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)


    fcm = FCM(n_clusters=3)
    fcm.fit(X)
    # print("centers:\n", fcm.centers.shape[1])
    # print("labels (first 10):", fcm.labels())
    # X2 = np.array([[0.0, 0.0]])
    # print("predict new point [0,0]:", fcm.predict(X2))

    # plot_2(fcm, X)

    # result_ari = adjusted_rand_score(true_labels, fcm.labels())
    # print("Adjusted Rand Index:", result_ari)
    