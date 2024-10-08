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
