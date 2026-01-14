"""
Reader that delegates image reading to a processing pipeline.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from tiamat.cache import instance_cache

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import ImageReader


class PipelineReader(ImageReader):
    """
    A reader that delegates image reading and metadata access to a given processing pipeline.

    This allows pipelines to be used as drop-in replacements for standard image readers.
    """

    def __init__(self, fname: str, pipeline: Callable, **reader_kwargs: Any):
        """
        Initializes a PipelineReader.

        Args:
            fname (str): Filename of the image.
            pipeline (Callable): A callable representing the image pipeline.
            **reader_kwargs (Any): Additional keyword arguments passed to the pipeline.
        """
        self.fname = fname
        self.pipeline = pipeline
        self.reader_kwargs = reader_kwargs

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Reads and processes the image using the pipeline.

        Args:
            accessor (ImageAccessor): Accessor specifying what to read.

        Returns:
            np.ndarray: The image array after applying the pipeline.
        """
        return self.pipeline(file_name=self.fname, accessor=accessor, **self.reader_kwargs).image

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        """
        Reads metadata using the pipeline.

        Returns:
            ImageMetadata: Metadata read from the pipeline.
        """
        return self.pipeline.read_metadata(file_name=self.fname, **self.reader_kwargs)

    @classmethod
    def check_file(cls, fname: str) -> bool:
        """
        This implementation always returns False (not implemented).

        Args:
            fname (str): Path to the file.

        Returns:
            bool: Always False.
        """
        return False

    @classmethod
    def from_json(
        cls, args: dict[str, Any], reader_post_creation_hook: Callable | None = None
    ) -> Callable[..., "PipelineReader"]:
        """
        Creates a PipelineReader from a JSON-like configuration.

        Args:
            args (Dict[str, Any]): Configuration dictionary with 'pipeline' key.
            reader_post_creation_hook (Callable, optional): Optional hook to modify the pipeline.

        Returns:
            Callable[..., PipelineReader]: A constructor for PipelineReader (partial with fixed pipeline).

        Note:
            Currently does not support reader_kwargs (TODO).
        """
        from tiamat.serialization import load_pipeline_from_config

        pipeline = load_pipeline_from_config(args["pipeline"], reader_post_creation_hook=reader_post_creation_hook)

        # TODO: Handle kwargs
        return partial(
            cls,
            pipeline=pipeline,
        )
