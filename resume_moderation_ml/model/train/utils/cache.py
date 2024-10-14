import functools
from typing import Any, Callable, Iterable, TypeVar, Union, cast

from ml_tools.kardinal_tools.state import State

T = TypeVar("T")  # Any type
MaybeIterable = Union[T, Iterable[T]]

StorageKey = MaybeIterable[str]


class Cache:
    def __init__(self, state: State):
        self.state = state

    def save(self, data: Any, key: StorageKey) -> None:
        self.state.cache_storage.save(data, key)

    def load(self, key: StorageKey) -> Any:
        return self.state.cache_storage.load(key)

    def contains(self, key: StorageKey) -> bool:
        return key in self.state.cache_storage

    def contains_all(self, *keys: StorageKey) -> bool:
        return all(self.contains(key) for key in keys)


def cache(filename: StorageKey, cache_cls: Cache) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def _inner(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def _process(*args: Any, **kwargs: Any) -> T:
            if filename in cache_cls.state.cache_storage:
                return cast(T, cache_cls.load(filename))

            result = func(*args, **kwargs)
            cache_cls.save(result, filename)
            return result

        return _process

    return _inner
