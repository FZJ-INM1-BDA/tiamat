"""
Pipeline for color mapping.
"""

import json
import os
import shutil

import matplotlib.pyplot as plt

from tiamat.io import ImageAccessor
from tiamat.readers import register_reader
from tiamat.readers.pipeline import PipelineReader
from tiamat.readers.stack import ImageStackReader
from tiamat.serialization import load_pipeline_from_config, register_class
from tiamat.transformers.axes import MirrorTransformer

# register some classes as valid transformers and readers
register_class(MirrorTransformer)
register_reader(ImageStackReader)
register_reader(PipelineReader)


with open("./data/stack_pipeline.json", "r") as f:
    pipeline_config = json.load(f)

# Stack the output of multiple pipelines
file_name = ["./data/Koala.jpg", "./data/Koala.jpg"]
pipeline = load_pipeline_from_config(pipeline_config)

result = pipeline(
    file_name=file_name,
    accessor=ImageAccessor(),
)
print(result.image.shape)
