"""
Helper functions for running a tiamat processing pipeline.
"""

from typing import Iterable, Callable
from .transformers.protocol import Transformer
from .readers.protocol import ImageReader
from .readers.factory import get_reader
from .io import ImageAccessor, ImageResult
from .metadata import ImageMetadata


class Pipeline:
    """
    Processing pipeline to read and transform images.
    A pipeline can be used to read an image and transform it.
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
        Args:
            transformers (iterable of Transformer): A list of transformers to apply to each image.
            access_transformers(iterable of Transformer): A list of transformers to modify the access, defined from coordinates to reading the image.
                                                          May not be used with transformers argument.
            image_transformers(iterable of Transformer): A list of transformers to modify images, defined from read image to result.
                                                         May not be used with transformers argument.
            reader_factory (callable): A function returning a reader for a given file name.
            register_default_readers (bool): Register default readers on pipeline call.
        """
        if transformers:
            assert (
                not access_transformers and not image_transformers
            ), "access_transformers and image_transformers may ne be used together with transformers argument."
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
        self, file_name, accessor: ImageAccessor, **reader_kwargs
    ) -> ImageResult:
        from dataclasses import replace

        # Disable user defined metadata for now
        assert accessor.metadata is None

        if self.auto_register_default_readers:
            from tiamat.readers import register_all_readers

            register_all_readers()
        reader = self.reader_factory(file_name, **reader_kwargs)

        # Forward rollout of metadata through transformers
        metadata = [reader.read_metadata()]

        for transformer in self.transformers:
            # Check for transformers that do not implement transform_metadata
            if hasattr(transformer, "transform_metadata"):
                metadata.append(transformer.transform_metadata(
                    metadata=replace(metadata[-1])),
                )
            else:
                metadata.append(metadata=replace(metadata[-1]))

        # Set the first accessor metadata to output metadata
        accessor.metadata = metadata[-1]

        # Backwards rollout of accessor through transformers and metadata
        accessors = [accessor]
        for transformer, meta in zip(self.transformers[::-1], metadata[::-1][1:]):
            accessor = transformer.transform_access(
                accessor=replace(accessors[-1]),
            )
            accessor.metadata = meta
            accessors.append(accessor)

        # Read image data
        image_result = reader.read_image(accessor=accessors[-1])

        # Forward pass through the transformers to get final image result
        for transformer, meta, acc in zip(self.transformers, metadata[:-1], accessors[::-1][1:]):
            image_result.metadata = meta
            image_result = transformer.transform_image(
                image_result=image_result,
                accessor=acc,
            )

        # Associate final output image with output metadata
        image_result.metadata = metadata[-1]

        return image_result

    def read_metadata(self, file_name, **reader_kwargs) -> ImageMetadata:
        from dataclasses import replace

        reader = self.reader_factory(file_name, **reader_kwargs)
        metadata = reader.read_metadata()
        # backwards pass through the transformers to transform the accessor
        for transformer in self.transformers:
            if hasattr(transformer, "transform_metadata"):
                # check for transformers that do not implement transform_metadata
                metadata = transformer.transform_metadata(
                    metadata=replace(metadata)
                )
        return metadata
