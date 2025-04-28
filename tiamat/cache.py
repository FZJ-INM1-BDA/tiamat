import weakref
from functools import wraps


class instance_cache:
    def __init__(self, func):
        self.func = func
        self._caches = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
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


class instance_cached_property:
    def __init__(self, func):
        self.func = func
        self._values = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        if instance not in self._values:
            value = self.func(instance)
            self._values[instance] = value

        return self._values[instance]
    
    def __set__(self, instance, value):
        self._values[instance] = value

    def __delete__(self, instance):
        if instance in self._values:
            del self._values[instance]
