"""
Pipeline serialization and instantiation from configuration dictionaries.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from tiamat.readers.protocol import ImageReader

# Example class registry
class_registry: dict[str, type] = {}


def register_class(cls: type) -> type:
    """
    Decorator to register a class for safe instantiation from configuration.

    Args:
        cls (Type): The class to register.

    Returns:
        Type: The registered class.
    """
    class_registry[cls.__name__] = cls
    return cls


def create_instance(class_name: str, args: dict[str, Any]) -> Any:
    """
    Safely instantiate a registered class with arguments from a configuration dictionary.

    Args:
        class_name (str): The name of the class to instantiate.
        args (Dict[str, Any]): Initialization arguments.

    Returns:
        Any: The instantiated object.

    Raises:
        ValueError: If the class is not registered.
    """
    if class_name not in class_registry:
        raise ValueError(f"Class '{class_name}' is not registered.")

    cls = class_registry[class_name]

    if hasattr(cls, "from_json"):
        return cls.from_json(args)
    else:
        # Validate provided arguments
        allowed_args = cls.__init__.__code__.co_varnames[1 : cls.__init__.__code__.co_argcount]
        filtered_args = {k: v for k, v in args.items() if k in allowed_args}

        return cls(**filtered_args)


def make_object_from_config(config_entry: dict) -> Any:
    """
    Instantiate an object from a single configuration entry.

    Args:
        config_entry (dict): Dictionary with "class" and optional "args" keys.

    Returns:
        Any: Instantiated object.

    Raises:
        ValueError: If the config entry is missing the "class" key.
    """
    class_name = config_entry.get("class")
    args = config_entry.get("args", {})

    if class_name is None:
        raise ValueError("Config entry must contain a 'class' key.")
    return create_instance(class_name, args)


def get_reader_from_config(
    config_reader: dict | None, reader_post_creation_hook: Callable | None = None
) -> Callable[..., "ImageReader"]:
    """
    Get a reader constructor from configuration.

    Args:
        config_reader dict | None: Configuration dictionary for the reader.
        reader_post_creation_hook Callable | None: Hook to modify the reader after construction.

    Returns:
        Callable[..., Any]: A reader constructor (possibly a partial).
    """

    if config_reader is None:
        from tiamat.readers.factory import get_reader

        if reader_post_creation_hook is None:
            return get_reader
        else:
            return partial(reader_post_creation_hook, get_reader)

    from tiamat.readers.factory import get_reader_from_registry

    class_name = config_reader.get("class")
    if class_name is None:
        raise ValueError("Reader config must contain a 'class' key.")
    cls = get_reader_from_registry(class_name)
    args = config_reader.get("args", {})

    if hasattr(cls, "from_json"):
        return cls.from_json(args, reader_post_creation_hook=reader_post_creation_hook)
    else:
        # Validate provided arguments
        allowed_args = cls.__init__.__code__.co_varnames[1 : cls.__init__.__code__.co_argcount]
        filtered_args = {k: v for k, v in args.items() if k in allowed_args}

        if reader_post_creation_hook is None:
            return partial(cls, **filtered_args)
        else:
            return partial(reader_post_creation_hook, cls, **filtered_args)


def load_pipeline_from_config(
    config: dict, auto_register_default_readers: bool = True, reader_post_creation_hook: Callable | None = None
) -> Any:
    """
    Construct a pipeline from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary defining transformers and reader.
        auto_register_default_readers (bool): If True, registers built-in readers.
        reader_post_creation_hook (Callable | None): Optional hook for customizing the reader.

    Returns:
        Pipeline: The constructed pipeline.
    """
    from .pipeline import Pipeline

    if auto_register_default_readers:
        from tiamat.readers import register_all_readers

        register_all_readers()

    return Pipeline(
        transformers=[make_object_from_config(item) for item in config.get("transformers", [])],
        access_transformers=[make_object_from_config(item) for item in config.get("access_transformers", [])],
        image_transformers=[make_object_from_config(item) for item in config.get("image_transformers", [])],
        reader_factory=get_reader_from_config(
            config.get("reader", None), reader_post_creation_hook=reader_post_creation_hook
        ),
        auto_register_default_readers=False,
    )
