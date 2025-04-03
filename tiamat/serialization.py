"""
Pipeline serialization.
"""

from functools import partial
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

    if hasattr(cls, "from_json"):
        return cls.from_json(args)
    else:
        # Validate provided arguments
        allowed_args = cls.__init__.__code__.co_varnames[
            1 : cls.__init__.__code__.co_argcount
        ]
        filtered_args = {k: v for k, v in args.items() if k in allowed_args}

        return cls(**filtered_args)


def make_object_from_config(config_entry: dict):
    """Make a transformer from a single config entry."""
    return create_instance(config_entry["class"], config_entry.get("args", {}))


def get_reader_from_config(config_reader: dict):
    from tiamat.readers.factory import get_reader_from_registry#

    if config_reader is None:
        return None

    cls = get_reader_from_registry(config_reader["class"])
    args = config_reader.get("args", {})

    if hasattr(cls, "from_json"):
        return cls.from_json(args)
    else:        
        # Validate provided arguments
        allowed_args = cls.__init__.__code__.co_varnames[
            1 : cls.__init__.__code__.co_argcount
        ]
        filtered_args = {k: v for k, v in args.items() if k in allowed_args}
        return partial(cls, **filtered_args)


def load_pipeline_from_config(config: dict, auto_register_default_readers=True):
    """Make a pipeline from a config dict."""
    from .pipeline import Pipeline

    if auto_register_default_readers:
        from tiamat.readers import register_all_readers

        register_all_readers()

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
        reader_factory=get_reader_from_config(config.get("reader", None)),
        auto_register_default_readers=False,
    )


# def connect_pipelines_from_config(config: dict, auto_register_default_readers=True, out_pipeline="out"):
#     """Connect pipelines defined in a config. Each entry in the config is a pipeline"""
#     from tiamat.readers.pipeline import PipelineReader

#     pipeline_config = config[out_pipeline]
#     pipeline_input = pipeline_config.get("input")

#     def _setup_pipeline(name):
#         pipeline = connect_pipelines_from_config(config, out_pipeline=name)
#         return partial(PipelineReader, pipeline=pipeline)

#     if pipeline_input is None:
#         return load_pipeline_from_config(
#             pipeline_config,
#             auto_register_default_readers=auto_register_default_readers,
#         )
#     else:
#         input_pipelines = []
#         if type(pipeline_input) is str:
#             input_pipelines = [_setup_pipeline(pipeline_input)]
#         elif hasattr(pipeline_input, "__iter__"):
#             for name in pipeline_input:
#                 input_pipelines.append(_setup_pipeline(name))

#         return load_pipeline_from_config(
#             pipeline_config,
#             auto_register_default_readers=auto_register_default_readers,
#             input_pipelines=input_pipelines,
#         )
