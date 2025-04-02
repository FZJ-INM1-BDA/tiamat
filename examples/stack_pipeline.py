"""
Using pipeline as readers.
"""
from functools import partial
import shutil
import os

from tiamat.transformers.color import GrayscaleTransformer, GrayscaleToRGBTransformer
from tiamat.pipeline import Pipeline
from tiamat.io import ImageAccessor
from tiamat.readers.pipeline import PipelineReader
from tiamat.readers.stack import StackReader

shutil.copy("./data/Koala.jpg", "./data/Koala_2.jpg")

def _cleanup():
    # Cleanup
    os.remove("./data/Koala_2.jpg")

try:
    pipeline_a = Pipeline(transformers=[])
    result_a = pipeline_a(
        file_name="./data/Koala.jpg", accessor=ImageAccessor()
    )
    print(result_a.image.shape)

    pipeline_b = Pipeline(transformers=[])
    result_b = pipeline_b(
        file_name="./data/Koala_2.jpg", accessor=ImageAccessor()
    )
    print(result_b.image.shape)

    pipelines = [pipeline_a, pipeline_b]

    # Stack single pipeline for single file
    file_name = "./data/Koala.jpg"
    reader_factory = partial(StackReader, reader_factory=partial(PipelineReader, pipeline=pipeline_a))

    pipeline = Pipeline(
        transformers=[],
        reader_factory=reader_factory,
    )
    result = pipeline(file_name=file_name, accessor=ImageAccessor())
    print(result.image.shape)

    # Stack single pipeline from multiple files
    file_name = "./data/Koala*.jpg"
    reader_factory = partial(StackReader, reader_factory=partial(PipelineReader, pipeline=pipeline_a))

    pipeline = Pipeline(
        transformers=[],
        reader_factory=reader_factory,
    )
    result = pipeline(file_name=file_name, accessor=ImageAccessor())
    print(result.image.shape)

    # Stack one pipeline for each file
    file_name = ["./data/Koala.jpg", "./data/Koala_2.jpg"]
    pipeline_dict = dict(
        (fname, pipelines[i % 2]) for i, fname in enumerate(file_name)
    )
    reader_factory = partial(StackReader, reader_factory=lambda filename: PipelineReader(filename, pipeline=pipeline_dict[filename]))

    pipeline = Pipeline(
        transformers=[],
        reader_factory=reader_factory,
    )
    result = pipeline(file_name=file_name, accessor=ImageAccessor())
    print(result.image.shape)

    # Multiple pipelines for single file
    file_name = ["./data/Koala.jpg"] * 2
    reader_factory =  partial(StackReader, reader_factory=[
        partial(PipelineReader, pipeline=pipeline_a),
        partial(PipelineReader, pipeline=pipeline_b)
    ])

    pipeline = Pipeline(
        transformers=[],
        reader_factory=reader_factory,
    )
    result = pipeline(file_name=file_name, accessor=ImageAccessor())
    print(result.image.shape)

    _cleanup()
except:
    _cleanup()
    raise
