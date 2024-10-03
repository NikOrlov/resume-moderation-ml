import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import strip_accents_unicode
from nltk.stem import PorterStemmer, SnowballStemmer

from resume_moderation_ml.model.train.utils.transformers import NoFitMixin

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
