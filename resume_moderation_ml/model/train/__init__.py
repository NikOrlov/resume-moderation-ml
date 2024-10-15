from typing import Iterator

from ml_tools.kardinal_tools.state import State

from resume_moderation_ml.model.train.config import resume_moderation_config
from resume_moderation_ml.model.train.utils.cache import Cache

TARGET_FLAGS = ["careless_key_skill_information", "careless_additional_information", "bad_function", "bad_education"]

TASKS = ["approve_incomplete", "approve_complete", "block"] + TARGET_FLAGS


def iterate_file_lines(filename: str) -> Iterator[str]:
    with open(filename) as input_file:
        for line in input_file:
            yield line


state = State(resume_moderation_config)
cache_obj = Cache(state)
