import calendar
from datetime import datetime
from typing import Any, Iterable, Mapping, TypeVar, Union

import dateutil.parser

T = TypeVar("T")  # Any type
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


def alpha_quotient(text: str) -> float:
    if not text:
        return 0.0
    return sum(1.0 for ch in text if ch.isalpha()) / len(text)


def caps_quotient(text: str) -> float:
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1.0 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)


def load_currency_rates() -> dict:
    # data = requests.get('https://api.hh.ru/dictionaries').json()
    # return {currency['code']: currency['rate'] for currency in data['currency']}
    return {
        "AZN": 0.017696,
        "BYR": 0.034052,
        "EUR": 0.009496,
        "GEL": 0.028475,
        "KGS": 0.881694,
        "KZT": 5.015775,
        "RUR": 1.0,
        "UAH": 0.42873,
        "USD": 0.01041,
        "UZS": 132.983144,
    }
