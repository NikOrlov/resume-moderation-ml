from sklearn.feature_extraction.text import TfidfVectorizer

from resume_moderation_ml.model.stopwords import stop_words
from resume_moderation_ml.model.train.utils.transformers import JsonTextExtractor
from resume_moderation_ml.model.train.utils.text import Analyzer
from resume_moderation_ml.model.train.utils import identity_function, is_string

FIELDS_TO_ANALYZE = ['title', 'experience', 'education', 'university', 'skill', 'recommendation', 'name']

COMMON_CONFIG = {
    'analyzer_opts': {
        'use_stemming': True
    },
    'vectorizer_opts': {
        'preprocessor': identity_function,
        'tokenizer': identity_function,
        'decode_error': 'strict',
        'strip_accents': 'unicode',
        'lowercase': True,
        'binary': False
    },
    'tf_idf_opts': {
        'norm': 'l2',
        'use_idf': False,
        'sublinear_tf': True
    }
}

TEXT_FIELDS = {
    'title': {
        'extractor_opts': {
            'query': 'title'
        },
        'analyzer_opts': {
            'use_stemming': False
        },
        'vectorizer_opts': {
            'max_df': 0.9995,
            'min_df': 0.0005,
            'max_features': 1000,
        },
        'tf_idf_opts': {
            'sublinear_tf': False
        }
    },
    'experience': {
        'extractor_opts': {
            'query': {'experience': ['companyName', 'description', 'position']}
        },
        'vectorizer_opts': {
            'stop_words': stop_words,
            'max_df': 0.995,
            'min_df': 0.005,
            'max_features': 2000
        }
    },
    'education': {
        'extractor_opts': {
            'query': {
                'attestationEducation': ['name', 'organization'],
                'additionalEducation': ['name', 'organization'],
                'primaryEducation': ['name', 'organization'],
                'elementaryEducation': 'name'
            }
        },
        'vectorizer_opts': {
            'max_df': 0.9975,
            'min_df': 0.0025,
            'max_features': 1000
        }
    },
    'university': {
        'extractor_opts': {
            'query': {'primaryEducation': 'name'}
        },
        'vectorizer_opts': {
            'max_df': 0.9995,
            'min_df': 0.0005,
            'max_features': 1000
        }
    },
    'skill': {
        'extractor_opts': {
            'query': ['skills', 'keySkills']
        },
        'vectorizer_opts': {
            'max_df': 0.999,
            'min_df': 0.001,
            'max_features': 3000
        }
    },
    'recommendation': {
        'extractor_opts': {
            'query': {'recommendation': ['organization', 'position']}
        },
        'vectorizer_opts': {
            'max_df': 0.999,
            'min_df': 0.001,
            'max_features': 300,
        }
    },
    'name': {
        'extractor_opts': {
            'query': ['firstName', 'middleName', 'lastName']
        },
        'analyzer_opts': {
            'use_stemming': False
        },
        'vectorizer_opts': {
            'max_df': 0.999,
            'min_df': 0.001,
            'max_features': 300,
        },
        'tf_idf_opts': {
            'sublinear_tf': False
        }
    },
    'full_text': {
        'extractor_opts': {
            'query': {
                'title': None,
                'experience': ['companyName', 'description', 'position'],
                'attestationEducation': ['name', 'organization'],
                'additionalEducation': ['name', 'organization'],
                'primaryEducation': ['name', 'organization'],
                'elementaryEducation': 'name',
                'skills': None,
                'keySkills': None,
                'recommendation': ['organization', 'position'],
                'firstName': None,
                'middleName': None,
                'lastName': None
            }
        }
    }
}


def _collect_opts(field_name, keys):
    opts = {}
    if is_string(keys):
        keys = [keys]

    for config in (COMMON_CONFIG, TEXT_FIELDS[field_name]):
        for key in keys:
            opts.update(config.get(key, {}))
    return opts


def create_extractor(field_name):
    return JsonTextExtractor(**_collect_opts(field_name, 'extractor_opts'))


def create_analyzer(field_name):
    return Analyzer(**_collect_opts(field_name, 'analyzer_opts'))


def create_vectorizer(field_name, vocabulary=None):
    opts = _collect_opts(field_name, ['vectorizer_opts', 'tf_idf_opts'])
    if vocabulary is not None:
        opts['vocabulary'] = vocabulary
    return TfidfVectorizer(**opts)
