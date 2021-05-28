from typing import Any, Optional


def _factory(cls: Any, name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
    """Create object of class 'name' for parent class 'cls', return None if failed."""
    for subcls in cls.__subclasses__():
        if subcls.__name__ == name:
            return subcls(*args, **kwargs)
        obj = _factory(subcls, name, *args, **kwargs)
        if obj is not None:
            return obj
    return None


def factory(cls: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Create object of class 'name' for parent class 'cls', raise NotImplementedError if failed."""
    obj = _factory(cls, name, *args, **kwargs)
    if obj is not None:
        return obj
    raise NotImplementedError("Class '{}' not found as subclass of '{}'".format(name, cls.__name__))
