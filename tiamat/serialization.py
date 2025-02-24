"""
Pipeline serialization.
"""

import json
from typing import Dict, Any, Type

# Example class registry
class_registry: Dict[str, Type] = {}


def register_class(cls: Type):
    """Decorator to register a class for safe instantiation."""
    class_registry[cls.__name__] = cls
    return cls


def create_instance(class_name: str, args: Dict[str, Any]) -> Any:
    """Safely instantiate a class with arguments from JSON."""
    if class_name not in class_registry:
        raise ValueError(f"Class '{class_name}' is not registered.")

    cls = class_registry[class_name]

    # Validate provided arguments
    allowed_args = cls.__init__.__code__.co_varnames[
        1 : cls.__init__.__code__.co_argcount
    ]
    filtered_args = {k: v for k, v in args.items() if k in allowed_args}

    return cls(**filtered_args)


def make_object_from_config(config_entry: dict):
    """Make an object from a single config entry."""
    return create_instance(config_entry["class"], config_entry.get("args", {}))


def load_pipeline_from_config(config: dict):
    """Make a pipeline from a config dict."""
    from .pipeline import Pipeline

    return Pipeline(
        transformers=[
            make_object_from_config(item) for item in config.get("transformers", [])
        ],
        access_transformers=[
            make_object_from_config(item)
            for item in config.get("access_transformers", [])
        ],
        image_transformers=[
            make_object_from_config(item)
            for item in config.get("image_transformers", [])
        ],
    )
