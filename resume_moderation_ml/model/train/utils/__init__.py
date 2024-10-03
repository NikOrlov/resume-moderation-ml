import calendar
import dateutil.parser
from datetime import datetime

from typing import Any, Iterable, Mapping, TypeVar, Union


T = TypeVar('T')  # Any type
MaybeIterable = Union[T, Iterable[T]]
StorageKey = MaybeIterable[str]


def identity_function(argument: T) -> T:
    return argument

def is_integer(variable: Any) -> bool:
    return isinstance(variable, int)


def is_string(variable: Any) -> bool:
    return isinstance(variable, str)


def is_mapping(variable: Any) -> bool:
    return isinstance(variable, Mapping)


def is_iterable(variable: Any) -> bool:
    return isinstance(variable, Iterable) and not (is_string(variable) or is_mapping(variable))


def timestamp_from_string(string):
    return calendar.timegm(dateutil.parser.parse(string).utctimetuple()) * 1000


def timestamp_from_year(year):
    return calendar.timegm(datetime(year, 1, 1, 0, 0).utctimetuple()) * 1000
