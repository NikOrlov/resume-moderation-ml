import os


def _load(path):
    with open(path) as fp:
        return {line.strip().lower() for line in fp.readlines()}


_path = os.path.dirname(os.path.abspath(__file__))

stop_positions = _load(os.path.join(_path, 'stop_positions.txt'))
stop_words = _load(os.path.join(_path, 'stop_words.txt'))
