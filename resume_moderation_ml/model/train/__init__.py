from typing import Iterator
from resume_moderation_ml.model.train.config import ModerationConfig
from resume_moderation_ml.model.train.utils.cache import Cache
from ml_tools.kardinal_tools.state import State


def iterate_file_lines(filename: str) -> Iterator[str]:
    with open(filename) as input_file:
        for line in input_file:
            yield line


config = ModerationConfig()
state = State(config)
cache_obj = Cache(state)
