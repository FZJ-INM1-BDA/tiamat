"""
Reader for pipelines.
"""

from functools import cached_property, cache, partial
from typing import Any, Dict
from .protocol import ImageReader
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class PipelineReader(ImageReader):
    def __init__(self, fname, pipeline, **reader_kwargs):
        self.fname = fname
        self.pipeline = pipeline
        self.reader_kwargs = reader_kwargs

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        return self.pipeline(
            file_name=self.fname, accessor=accessor, **self.reader_kwargs
        )

    def read_metadata(self) -> ImageMetadata:
        return self.pipeline.read_metadata(file_name=self.fname, **self.reader_kwargs)

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        return False

    @classmethod
    def from_json(cls, args: Dict[str, Any], reader_post_creation_hook=None):
        from tiamat.serialization import load_pipeline_from_config

        pipeline = load_pipeline_from_config(args["pipeline"], reader_post_creation_hook=reader_post_creation_hook)

        #TODO: Handle kwargs
        return partial(
            cls,
            pipeline=pipeline,
        )
