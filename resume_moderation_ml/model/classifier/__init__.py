import functools
import os.path

from ml_tools.kardinal_tools.utils.storage import PICKLE_FORMAT, FSStorage

TARGET_FLAGS = ["careless_key_skill_information", "careless_additional_information", "bad_function", "bad_education"]

TASKS = ["approve_incomplete", "approve_complete", "block"] + TARGET_FLAGS

_TRAIN_STORAGE = FSStorage(os.path.dirname(__file__), PICKLE_FORMAT, "resume_moderation_classifier")


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
