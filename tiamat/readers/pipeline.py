"""
Reader for pipelines.
"""

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
