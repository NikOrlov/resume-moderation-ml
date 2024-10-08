import csv
import datetime
import functools
import itertools
import json
import logging
import os
import sys
from collections import Counter

from ciso8601 import parse_datetime
from sklearn.model_selection import train_test_split

from resume_moderation_ml.model.train.config import ModerationConfig
from resume_moderation_ml.model.train.environment import init_train_env
from resume_moderation_ml.model.train.targets import ModerationTargets
from resume_moderation_ml.model.train.utils.cache import Cache
from resume_moderation_ml.model.train import iterate_file_lines
from ml_tools.kardinal_tools.state import State
from resume_moderation_ml.model.train import cache_obj, config, state

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

# config = ModerationConfig()
# state = State(config)
# cache = Cache(state)

_SOURCE_RESUME_CSV_FILES_FROM_HIVE = {
    'all': 'resume_moderation_ml/model/train/data/resume_from_hive.csv',
    'main': 'resume_moderation_ml/model/train/data/resume_from_hive_main.csv',
    'vectorizer': 'resume_moderation_ml/model/train/data/resume_from_hive_vectorizer.csv'
}

_SOURCE_DELETED_RESUME_CSV_FILE_FROM_DB = 'resume_moderation_ml/model/train/data/resume_deleted_from_db.csv'
_RAW_RESUMES_KEYS = {'main': 'resume_moderation_ml/model/train/raw_resumes',
                     'vectorizer': 'resume_moderation_ml/model/train/vectorizer_resumes'}
_TARGETS_KEYS = {'main': 'resume_moderation_ml/model/train/targets',
                 'vectorizer': 'resume_moderation_ml/model/train/vectorizer_targets'}


# This function is a hack for obtaining a lines of source moderation records csv file, it should be replaced with hdfs
# access in the future. You should reassign this function for providing alternative sources in unit tests.
get_source_csv_lines_from_hive = {
    'all': functools.partial(iterate_file_lines, _SOURCE_RESUME_CSV_FILES_FROM_HIVE['all']),
    'main': functools.partial(iterate_file_lines, _SOURCE_RESUME_CSV_FILES_FROM_HIVE['main']),
    'vectorizer': functools.partial(iterate_file_lines, _SOURCE_RESUME_CSV_FILES_FROM_HIVE['vectorizer'])
}
get_source_csv_lines_from_db = functools.partial(iterate_file_lines, _SOURCE_DELETED_RESUME_CSV_FILE_FROM_DB)
DELIMITER = '\x01'
QUOTE_CHAR = '`'


_ALLOWED_STATUSES = ['approved', 'blocked', 'deleted']

_CREATION_TIME_THRESHOLD = parse_datetime(config.creation_time_threshold)


def iterate_raw_source_csv(input_file):
    reader = csv.reader(input_file, delimiter=DELIMITER, quotechar=QUOTE_CHAR)

    for row in reader:
        log_create_time, id_, moderator_name, status, creation_time, source_version, moderated_version = row

        if not moderator_name.strip():
            logger.debug('skip record with id %s, moderator name is absent', id_)
            continue

        if status not in _ALLOWED_STATUSES:
            logger.debug('skip record with id %s, unknown record type', id_)
            continue

        if not creation_time:
            logger.debug('skip record with id %s, creation time is absent', id_)
            continue
        creation_time = datetime.datetime.fromtimestamp(float(creation_time) / 1000)
        if status != 'deleted' and creation_time < _CREATION_TIME_THRESHOLD:
            logger.debug('skip record with id %s, creation time is before threshold: %s', id_,
                         creation_time.strftime('%Y-%m-%d %H:%M:%S'))
            continue

        try:
            source_version = json.loads(source_version)
            moderated_version = json.loads(moderated_version)
        except ValueError:
            logger.debug('skip record with id %s, json format error', id_)
            continue

        if not source_version or not moderated_version:
            logger.debug('skip record with id %s, resume is absent', id_)
            continue
        if 'validationSchema' in source_version and source_version['validationSchema'] == 'incomplete':
            complete_status = 'incomplete'
        else:
            complete_status = 'complete'
        yield id_, status, source_version, moderated_version, complete_status, creation_time


def _extract_raw_data(dataset='main'):
    csv_lines_iterator = get_source_csv_lines_from_hive[dataset]
    raw_resumes = []
    targets = ModerationTargets()

    status_counter = Counter()

    for id_, status, source_version, moderated_version, complete_status, creation_time in itertools.chain(
            iterate_raw_source_csv(csv_lines_iterator()),
            iterate_raw_source_csv(get_source_csv_lines_from_db())
    ):
        raw_resumes.append(source_version)
        targets.add(status, complete_status, moderated_version['moderationFlags'])

        status_counter[status] += 1

    logger.info('created %d data objects eligible for learning, %s', targets.size,
                '; '.join(status + ': ' + str(count) for status, count in status_counter.items()))
    return raw_resumes, targets


def get_raw_resumes_and_targets(dataset='main'):
    if cache_obj.contains_all(_RAW_RESUMES_KEYS[dataset], _TARGETS_KEYS[dataset]):
        return cache_obj.load(_RAW_RESUMES_KEYS[dataset]), cache_obj.load(_TARGETS_KEYS[dataset])

    if not os.path.exists(_SOURCE_RESUME_CSV_FILES_FROM_HIVE[dataset]):
        split_main_vectorizer()

    logger.info('not all raw data is present in storage, start extracting')
    raw_resumes, targets = _extract_raw_data(dataset=dataset)

    cache_obj.save(raw_resumes, _RAW_RESUMES_KEYS[dataset])
    cache_obj.save(targets, _TARGETS_KEYS[dataset])

    return raw_resumes, targets


def get_raw_resumes():
    return get_raw_resumes_and_targets()[0]


def get_targets():
    return get_raw_resumes_and_targets()[1]


def get_vectorizer_resumes():
    return get_raw_resumes_and_targets(dataset='vectorizer')[0]


def get_vectorizer_targets():
    return get_raw_resumes_and_targets(dataset='vectorizer')[1]


def split_main_vectorizer():
    if (os.path.exists(_SOURCE_RESUME_CSV_FILES_FROM_HIVE['main']) and
       os.path.exists(_SOURCE_RESUME_CSV_FILES_FROM_HIVE['vectorizer'])):
        return
    csv_lines_iterator = get_source_csv_lines_from_hive['all']
    allowed_ids = []
    for id_, status, source_version, moderated_version, complete_status, creation_time \
            in iterate_raw_source_csv(csv_lines_iterator()):
        id_ = int(id_)
        if not config.moderated_by_human(id_, creation_time):
            continue
        allowed_ids.append(id_)
    main_resume_ids, vectorizer_resume_ids = map(set, train_test_split(allowed_ids, test_size=config.vectorizer_size,
                                                                       random_state=config.cv_seed))

    with open(_SOURCE_RESUME_CSV_FILES_FROM_HIVE['all'], 'r') as in_file, \
            open(_SOURCE_RESUME_CSV_FILES_FROM_HIVE['main'], 'w') as main_file, \
            open(_SOURCE_RESUME_CSV_FILES_FROM_HIVE['vectorizer'], 'w') as vectorizer_file:
        for line in in_file:
            try:
                resume_id = int(line.strip().split(DELIMITER)[1])
            except ValueError:
                continue
            if resume_id in main_resume_ids:
                main_file.write(line)
            elif resume_id in vectorizer_resume_ids:
                vectorizer_file.write(line)


if __name__ == '__main__':
    split_main_vectorizer()
    get_raw_resumes_and_targets(dataset='main')
    get_raw_resumes_and_targets(dataset='vectorizer')
