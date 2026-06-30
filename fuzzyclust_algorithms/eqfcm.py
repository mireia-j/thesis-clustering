import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

try:
    from fuzzyclust.base import plot_2, BaseKanzawa
except ImportError:
    from base import plot_2, BaseKanzawa

class eQFCM(BaseKanzawa):
    """Extended QFCM (Kanzawa, 2024). Algorithm 2, four fuzzification params.

    Reduces to QFCM when m1=m2 and lambda1=lambda2, and to YFCM when
    m1=m, lambda1=lambda, m2->1+, lambda2->+inf.
    """

    def __init__(self, n_clusters=10, m1=2, m2=2, lambda1=1.0, lambda2=1.0,
                 max_iter=150, error=1e-5, random_state=42):
        # base `m` (used by next_centers) is the membership exponent m1
        super().__init__(n_clusters=n_clusters, m=m1, lambda_=lambda1,
                         max_iter=max_iter, error=error,
                         random_state=random_state)
        self.m1 = m1
        self.m2 = m2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def factor1(self, D):
        # 1 - lambda1(1-m2) d_ik  (>= 1 > 0)
        return 1.0 - self.lambda1 * (1.0 - self.m2) * D

    def update_alpha(self, u, D):
        # eq. 36
        S = ((u ** self.m1) * self.factor1(D)).sum(axis=0)
        a = S ** (1.0 / self.m2)
        return a / a.sum()

    def update_membership(self, D):
        # eq. 38
        bracket = (self.alpha[None, :] ** (1.0 - self.m2)) * self.factor1(D) - 1.0
        inner = (self.lambda2 * (self.m1 - 1.0) * bracket
                 + self.lambda1 * (self.m2 - 1.0))
        num = inner ** (1.0 / (1.0 - self.m1))
        return num / num.sum(axis=1, keepdims=True)