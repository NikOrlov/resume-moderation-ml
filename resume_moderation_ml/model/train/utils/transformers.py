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


