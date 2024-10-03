from typing import Iterator


def iterate_file_lines(filename: str) -> Iterator[str]:
    with open(filename) as input_file:
        for line in input_file:
            yield line
