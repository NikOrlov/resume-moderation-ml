import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


def to_numpy_array(arr):
    if sparse.issparse(arr):
        return arr.toarray()
    return arr


class ManualRejectEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_number, delegate):
        self.feature_number_ = feature_number
        self.delegate_ = delegate

    def fit(self, X, y, **fit_params):
        self.delegate_ = self.delegate_.fit(X, y, **fit_params)
        return self

    def _not_rejected(self, X):
        return np.where(to_numpy_array(X[:, self.feature_number_]) == 0.0)[0]

    def predict(self, X):
        not_rejected = self._not_rejected(X)

        result = np.zeros(X.shape[0])
        if not_rejected.size > 0:
            result[not_rejected] = self.delegate_.predict(X[not_rejected])
        return result

    def predict_proba(self, X):
        not_rejected = self._not_rejected(X)

        result = np.vstack((np.ones(X.shape[0]), np.zeros(X.shape[0]))).transpose()
        if not_rejected.size > 0:
            result[not_rejected] = self.delegate_.predict_proba(X[not_rejected])
        return result

    @property
    def delegate(self):
        return self.delegate_

    @property
    def threshold(self):
        return self.delegate.threshold

    @threshold.setter
    def threshold(self, threshold):
        self.delegate.threshold = threshold

    @property
    def eval_results(self):
        return self.delegate.eval_results
