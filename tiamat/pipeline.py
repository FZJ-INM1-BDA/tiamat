"""
Helper functions for running a tiamat processing pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .io import ImageAccessor, ImageResult
from .metadata import ImageMetadata
from .readers.factory import get_reader
from .readers.protocol import ImageReader
from .transformers.protocol import Transformer


class Pipeline:
    """
    A processing pipeline to read and transform images.

    The pipeline first transforms the image accessor (e.g. coordinates),
    reads image data using the appropriate reader, and then applies image transformations.
    """

    def __init__(
        self,
        transformers: Iterable[Transformer] | None = None,
        access_transformers: Iterable[Transformer] | None = None,
        image_transformers: Iterable[Transformer] | None = None,
        reader_factory: Callable[[str], ImageReader] | None = None,
        auto_register_default_readers: bool = True,
    ):
        """
        Initializes the Pipeline.

        Args:
            transformers (Iterable[Transformer] | None): Transformers applied to both access and image.
            access_transformers (Iterable[Transformer] | None): Transformers for modifying image access (coordinates).
                Cannot be used together with `transformers`.
            image_transformers (Iterable[Transformer] | None): Transformers for modifying image data.
                Cannot be used together with `transformers`.
            reader_factory (Callable[[str], ImageReader] | None): Function returning a reader for a given file name.
            auto_register_default_readers (bool): Whether to auto-register default readers on pipeline call.
        """
        if transformers:
            assert (
                not access_transformers and not image_transformers
            ), "access_transformers and image_transformers may not be used together with transformers argument."
        self.transformers = list(transformers or [])

        # For convenience, access transformers and image transformers can be specified separately.
        # This saves users from thinking about the (maybe) unintuitive order of coordinate transformers.
        if image_transformers:
            self.transformers.extend(image_transformers)
        if access_transformers:
            # Access transformers are applied in reverse order. Make sure that access_transformers is reversable first.
            access_transformers = list(access_transformers)
            self.transformers.extend(access_transformers[::-1])

        self.reader_factory = reader_factory or get_reader
        self.auto_register_default_readers = auto_register_default_readers

    def __call__(
        self,
        file_name: str,
        accessor: ImageAccessor,
        **reader_kwargs: Any,
    ) -> ImageResult:
        """
        Runs the pipeline on a file and returns the processed result.

        Args:
            file_name (str): Path to the image file.
            accessor (ImageAccessor): Image accessor specifying region and resolution.
            read_metadata (bool): Whether to read metadata if not already set.
            **reader_kwargs: Additional keyword arguments passed to the reader.

        Returns:
            ImageResult: The final image result after all transformations.
        """
        from dataclasses import replace

        if self.auto_register_default_readers:
            from tiamat.readers import register_all_readers

            register_all_readers()
        reader = self.reader_factory(file_name, **reader_kwargs)

        # Forward rollout of metadata through transformers
        metadata = [reader.read_metadata()]

        for transformer in self.transformers:
            # Check for transformers that do not implement transform_metadata
            if hasattr(transformer, "transform_metadata"):
                metadata.append(
                    transformer.transform_metadata(metadata=replace(metadata[-1])),
                )
            else:
                metadata.append(replace(metadata[-1]))

        # Backwards rollout of accessor through transformers and metadata
        accessors = [accessor]
        for transformer, meta in zip(self.transformers[::-1], metadata[::-1][:-1]):
            accessor = transformer.transform_access(accessor=replace(accessors[-1]), metadata=meta)
            accessors.append(accessor)

        # Read image data
        image = reader.read_image(accessor=accessors[-1])

        # Forward pass through the transformers to get final image result
        for transformer, meta, acc in zip(self.transformers, metadata[:-1], accessors[::-1][1:]):
            image = transformer.transform_image(
                image=image,
                metadata=meta,
                accessor=acc,
            )

        # Associate final output image with output metadata
        return ImageResult(
            image=image,
            metadata=metadata[-1],
        )

    def read_metadata(self, file_name: str, **reader_kwargs: Any) -> ImageMetadata:
        """
        Reads and transforms image metadata.

        Args:
            file_name (str): Path to the image file.
            **reader_kwargs: Additional keyword arguments passed to the reader.

        Returns:
            ImageMetadata: The (possibly transformed) metadata.
        """
        from dataclasses import replace

        reader = self.reader_factory(file_name, **reader_kwargs)
        metadata = reader.read_metadata()
        # backwards pass through the transformers to transform the accessor
        for transformer in self.transformers:
            if hasattr(transformer, "transform_metadata"):
                # check for transformers that do not implement transform_metadata
                metadata = transformer.transform_metadata(metadata=replace(metadata))
        return metadata
