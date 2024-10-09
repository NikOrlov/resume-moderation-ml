import logging
import numbers

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_random_state

from resume_moderation_ml.model.train import config, cache_obj
from resume_moderation_ml.model.train.source import get_source_csv_lines_from_hive, iterate_raw_source_csv
from resume_moderation_ml.model.train.utils.cache import cache
from resume_moderation_ml.model.train.utils.transformers import JsonTextExtractor
from resume_moderation_ml.model.train.utils.text import Analyzer
from resume_moderation_ml.model.train.utils import identity_function
from resume_moderation_ml.model.train.logger import setup_logger

logger = setup_logger(__name__)

_DATA_KEY = 'resume_moderation_ml/model/train/bad_positions/data'
_MODEL_KEY = 'resume_moderation_ml/model/train/bad_positions/model'


def extract_moderator_name(resume):
    moder_full_name = resume['moderFullName']
    if moder_full_name is None:
        return None
    return (moder_full_name['lastName'] + u' ' + moder_full_name['firstName']).lower()


@cache(_DATA_KEY, cache_cls=cache_obj)
def get_raw_data():
    logger.info('start extracting data for "bad titles" model')

    approved_titles = []
    blocked_titles = []
    csv_lines_iterator = get_source_csv_lines_from_hive['vectorizer']
    for id_, status, source_version, moderated_version, complete_status, creation_time \
            in iterate_raw_source_csv(csv_lines_iterator()):
        moderator_name = extract_moderator_name(moderated_version)
        if moderator_name is None or not source_version['title']:
            continue

        if status == 'approved' and 'bad_description' not in moderated_version['moderationFlags']:
            approved_titles.append(source_version['title'].lower().strip())
        elif status in ('blocked', 'deleted') and 'bad_description' in moderated_version['moderationFlags']:
            blocked_titles.append(source_version['title'].lower().strip())

    titles = approved_titles + blocked_titles
    targets = np.hstack([np.zeros(len(approved_titles)), np.ones(len(blocked_titles))])

    rng = check_random_state(config.bad_positions['seed'])
    permutation = rng.permutation(len(titles))
    titles = [titles[i] for i in permutation]
    targets = targets[permutation]

    if isinstance(config.bad_positions['subset_size'], numbers.Real):
        subset_size = int(round(len(titles) * config.bad_positions['subset_size']))
    else:
        subset_size = config.bad_positions['subset_size']

    titles = titles[:subset_size]
    targets = targets[:subset_size]

    logger.info('extraction finished, chose %d titles with %d unique', subset_size, len(titles))
    return {
        'titles': titles,
        'targets': targets
    }


@cache(_MODEL_KEY, cache_cls=cache_obj)
def get_model():
    data = get_raw_data()
    logger.info('start training "bad titles" classifier with %d points', len(data['titles']))

    analyzer = Analyzer(use_stemming=False)
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        preprocessor=identity_function,
        tokenizer=identity_function,
        decode_error='strict',
        ngram_range=(1, 2),
        lowercase=True,
        norm='l2',
        binary=True, use_idf=True
    )
    log_reg = LogisticRegression(random_state=config.bad_positions['seed'], **config.bad_positions['log_reg'])
    make_pipeline(analyzer, vectorizer, log_reg).fit(data['titles'], data['targets'])
    extractor = JsonTextExtractor('title')

    logger.info('training finished')
    return make_pipeline(extractor, analyzer, vectorizer, log_reg)


if __name__ == '__main__':
    get_model()
