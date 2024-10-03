import functools
import logging
import os.path

from hhkardinal.server import lfs
from hhkardinal.utils.storage import PICKLE_FORMAT, FSStorage, LoggingStorage

TARGET_FLAGS = ['careless_key_skill_information', 'careless_additional_information',
                'bad_function', 'bad_education']

TASKS = ['approve_incomplete', 'approve_complete', 'block'] + TARGET_FLAGS

_TRAIN_STORAGE = LoggingStorage(
    FSStorage(os.path.dirname(__file__), PICKLE_FORMAT, 'resume_moderation_classifier'),
    logging.getLogger('model')
)


def save(data, key):
    _TRAIN_STORAGE.save(data, key)


def load(key):
    return _TRAIN_STORAGE.load(key)


# TODO: unify decorators with storage registry
def store(filename):
    def _inner(func):
        @functools.wraps(func)
        def _process():
            if filename in _TRAIN_STORAGE:
                return load(filename)

            result = func()
            save(result, filename)
            return result

        return _process

    return _inner


STORAGE = lfs.create_storage_variants(
    'resume_moderation', lfs.LFS_PICKLE_FORMAT,
    name_prefix='resume_moderation_classifier', logger=logging.getLogger('model')
)
