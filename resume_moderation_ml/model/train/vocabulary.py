import logging
import sys

from sklearn.pipeline import Pipeline

from resume_moderation_ml.model import text_fields
from resume_moderation_ml.model.train import source
from resume_moderation_ml.model.train.environment import init_train_env
from resume_moderation_ml.model.train.utils.cache import cache
from resume_moderation_ml.model.train import cache_obj
from resume_moderation_ml.model.train.logger import setup_logger

logger = setup_logger(__name__)


def _train_vectorizer(raw_resumes, field_name):
    vectorizer_pipe = Pipeline([
        ('extractor', text_fields.create_extractor(field_name)),
        ('analyzer', text_fields.create_analyzer(field_name)),
        ('vectorizer', text_fields.create_vectorizer(field_name))
    ])

    logger.info('fitting vectorizer for text field "%s" started', field_name)
    vectorizer_pipe.fit(raw_resumes)
    term_count = len(vectorizer_pipe.named_steps['vectorizer'].vocabulary_)
    logger.info('fitting vectorizer for text field "%s" finished, got %d terms', field_name, term_count)

    return vectorizer_pipe.named_steps['vectorizer']


def get_vocabularies(field_names):
    raw_resumes = source.get_raw_resumes()
    vocabularies = {}

    for field_name in field_names:
        @cache('moderation/resume/vocabulary/' + field_name, cache_cls=cache_obj)
        def _train_single():
            return _train_vectorizer(raw_resumes, field_name).vocabulary_
        vocabularies[field_name] = _train_single()

    return vocabularies


if __name__ == '__main__':
    init_train_env()
    fields = sys.argv[1:]
    if not fields:
        fields = text_fields.FIELDS_TO_ANALYZE
    get_vocabularies(fields)
