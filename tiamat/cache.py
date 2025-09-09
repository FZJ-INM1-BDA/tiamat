import weakref
from functools import wraps
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class instance_cache(Generic[T, R]):
    def __init__(self, func: Callable[..., R]):
        self.func = func
        self._caches = weakref.WeakKeyDictionary()

    def __get__(self, instance: T, owner) -> Callable[..., R]:
        if instance is None:
            return self

        @wraps(self.func)
        def wrapper(*args, **kwargs):
            cache = self._caches.setdefault(instance, {})
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = self.func(instance, *args, **kwargs)
            return cache[key]

        return wrapper


class instance_cached_property(property, Generic[T, R]):
    def __init__(self, func: Callable[[T], R]):
        super().__init__(func)
        self.func = func
        self._values = weakref.WeakKeyDictionary()

    def __get__(self, instance: T, owner) -> R:
        if instance is None:
            return self
        if instance not in self._values:
            self._values[instance] = self.func(instance)
        return self._values[instance]

    def __set__(self, instance: T, value: R):
        self._values[instance] = value

    def __delete__(self, instance: T):
        self._values.pop(instance, None)
