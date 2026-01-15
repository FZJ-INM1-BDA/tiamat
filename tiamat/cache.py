import weakref
from functools import wraps
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def make_hashable(obj: object):
    """
    Function to make lists, dictionary, sets or nested variants of these hashable.

    This works by transforming each collection into a fixed size tuple.

    Parameters
    ----------
    obj : object
        Any python object to return a hashable version of.

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((make_hashable(k), make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(x) for x in obj)
    return obj


class instance_cache(Generic[T, R]):
    """
    Decorator to cache method results per instance and argument set.

    Uses a weak reference dictionary to avoid memory leaks by not preventing
    garbage collection of instances.

    Example:
        @instance_cache
        def expensive_method(self, x, y):
            # expensive computation
            return result
    """

    def __init__(self, func: Callable[..., R]):
        """
        Initialize the instance_cache decorator.

        Args:
            func: The method to decorate. The results of this method
                  will be cached per instance and per argument set.
        """
        self.func = func
        self._caches = weakref.WeakKeyDictionary()

    def __get__(self, instance: T, owner) -> Callable[..., R]:
        """
        Return a wrapper that caches results for the given instance.

        Args:
            instance: The instance of the class calling the method.
            owner: The owner class of the method.

        Returns:
            Callable: The wrapped method with caching enabled.
        """
        if instance is None:
            return self

        @wraps(self.func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = self._caches.setdefault(instance, {})
            key = (make_hashable(args), make_hashable(kwargs))
            if key not in cache:
                cache[key] = self.func(instance, *args, **kwargs)
            return cache[key]

        return wrapper


class instance_cached_property(property, Generic[T, R]):
    """
    Descriptor to cache a computed property per instance.

    The cached value is stored in a weak reference dictionary to avoid memory leaks.
    Supports getting, setting, and deleting the cached value.

    Example:
        class MyClass:
            @instance_cached_property
            def expensive_property(self):
                # compute expensive value
                return value
    """

    def __init__(self, func: Callable[[T], R]):
        super().__init__(func)
        self.func = func
        self._values = weakref.WeakKeyDictionary()

    def __get__(self, instance: T, owner) -> R:
        """
        Get the cached property value, computing it if necessary.

        Args:
            instance: The instance accessing the property.
            owner: The owner class of the property.

        Returns:
            The cached property value.
        """
        if instance is None:
            return self
        if instance not in self._values:
            self._values[instance] = self.func(instance)
        return self._values[instance]

    def __set__(self, instance: T, value: R):
        """
        Set or override the cached property value.

        Args:
            instance: The instance for which to set the value.
            value: The value to store.
        """
        self._values[instance] = value

    def __delete__(self, instance: T):
        """
        Delete the cached property value for the instance.

        Args:
            instance: The instance for which to delete the value.
        """
        self._values.pop(instance, None)
