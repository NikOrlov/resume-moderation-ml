import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from resume_moderation_ml.model.train.utils.json_extraction import search_strings_in_json


class NoFitMixin(object):
    def fit(self, X, y=None):
        return self


class JsonTextExtractor(BaseEstimator, TransformerMixin, NoFitMixin):
    def __init__(self, query):
        self.query = query

    def transform(self, X):
        return [u' '.join(search_strings_in_json(data_object, self.query)) for data_object in X]


class ValueExtractor(TransformerMixin, NoFitMixin):
    def __init__(self, extract_value, dtype=None):
        self.extract_value = extract_value
        self.dtype = dtype

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = (x[1] for x in X.iterrows())

        transformed = np.array(list(map(self.extract_value, X)))
        if transformed.ndim == 1:
            transformed = transformed[:, np.newaxis]
        if self.dtype is not None:
            transformed = transformed.astype(self.dtype)
        return transformed


class LogTransformer(TransformerMixin, NoFitMixin):
    def transform(self, X):
        return np.log(1.0 + X)


class InputLength(BaseEstimator, TransformerMixin, NoFitMixin):
    def transform(self, X):
        return np.array([len(obj) for obj in X], dtype=np.float64)[:, np.newaxis]

    def get_feature_names(self):
        return ['len']


# compared to LabelEncoder from sklearn this one doesn't fail if it encounters unknown category
class CategoricalEncoder(TransformerMixin):
    def __init__(self):
        self.categories_ = None
        self.values_ = None

    def fit(self, X, y=None):
        values = sorted(set(x[0] for x in X))
        self.categories_ = {value: idx for idx, value in enumerate(values)}

        self.values_ = values
        self.values_.append('__unknown')
        return self

    def _check_fitted(self):
        if self.categories_ is None:
            raise ValueError('CategoricalEncoder was not fitted yet')

    def transform(self, X):
        self._check_fitted()
        unknown = len(self.categories_)
        transformed = np.array([self.categories_.get(x[0], unknown) for x in X])[:, np.newaxis]
        return transformed

    def inverse_transform(self, X):
        self._check_fitted()

        return np.array([self.values_[int(x[0])] for x in X])[:, np.newaxis]


class ClassifierTransformer(TransformerMixin, NoFitMixin):

    def __init__(self, clf):
        self.clf = clf

    def transform(self, X):
        result = self.clf.predict_proba(X)
        if result.shape[1] == 2:
            return result[:, [1]]
        return result
