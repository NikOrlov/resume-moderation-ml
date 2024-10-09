import re

import numpy as np
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import strip_accents_unicode

from resume_moderation_ml.model.train.utils.transformers import NoFitMixin


class OutOfVocabularyTokens(BaseEstimator, TransformerMixin, NoFitMixin):
    def __init__(self, vocabulary, stop_words=None):
        self.vocabulary = set(vocabulary)
        self.stop_words = stop_words

    def _transform_single(self, tokens):
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        if not tokens:
            return -1.0
        return sum(1.0 for token in tokens if token not in self.vocabulary) / len(tokens)

    def transform(self, X):
        return np.asarray(list(map(self._transform_single, X)))[:, np.newaxis]

    def get_feature_names(self):
        return ['oov']


class TokenLengthStatistics(BaseEstimator, TransformerMixin, NoFitMixin):
    def __init__(self, percentiles):
        self.percentiles = np.asarray(percentiles)
        self.feature_count = self.percentiles.size + 2  # percentiles, mean and variance

    def _transform_single(self, tokens):
        lengths = list(map(len, tokens))
        if not lengths:
            return np.zeros(self.feature_count)

        percentiles = np.array([np.percentile(lengths, p) for p in self.percentiles])
        return np.concatenate([percentiles, [np.mean(lengths), np.std(lengths)]])

    def transform(self, X):
        result = np.zeros((len(X), self.feature_count))
        for idx in range(len(X)):
            tokens = X[idx]
            if not tokens:
                continue
            lengths = list(map(len, tokens))
            percentiles = np.array([np.percentile(lengths, p) for p in self.percentiles])
            result[idx] = np.concatenate([percentiles, [np.mean(lengths), np.std(lengths)]])

        return result

    def get_feature_names(self):
        return ['perc_{:.1f}'.format(perc) for perc in self.percentiles] + ['mean', 'std']


_STEMMERS_CHAIN = [SnowballStemmer('russian'), PorterStemmer()]
_TOKENIZER_PATTERN = re.compile(r"(?u)\b\w\w+\b")


_STEMMING_CACHE = {}


def _stem(word):
    for stemmer in _STEMMERS_CHAIN:
        stemmed_word = stemmer.stem(word)
        if stemmed_word != word:
            return stemmed_word
    return word


def stem(word):
    if word not in _STEMMING_CACHE:
        _STEMMING_CACHE[word] = _stem(word)
    return _STEMMING_CACHE[word]


def _tokenize(text):
    return _TOKENIZER_PATTERN.findall(text)


class Analyzer(BaseEstimator, TransformerMixin, NoFitMixin):

    def __init__(self, use_stemming=True):
        self.use_stemming = use_stemming

    def analyze(self, text):
        tokens = _tokenize(strip_accents_unicode(text.lower()))
        return list(map(stem, tokens)) if self.use_stemming else tokens

    def transform(self, X):
        return list(map(self.analyze, X))
