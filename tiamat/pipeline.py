"""
Helper functions for running a tiamat processing pipeline.
"""

from typing import Iterable, Callable, Optional, Any
from .transformers.protocol import Transformer
from .readers.protocol import ImageReader
from .readers.factory import get_reader
from .io import ImageAccessor, ImageResult
from .metadata import ImageMetadata


class Pipeline:
    """
    A processing pipeline to read and transform images.

    The pipeline first transforms the image accessor (e.g. coordinates),
    reads image data using the appropriate reader, and then applies image transformations.
    """

    def __init__(
            self,
            transformers: Optional[Iterable[Transformer]] = None,
            access_transformers: Optional[Iterable[Transformer]] = None,
            image_transformers: Optional[Iterable[Transformer]] = None,
            reader_factory: Optional[Callable[[str], ImageReader]] = None,
            auto_register_default_readers: bool = True,
    ):
        """
        Initializes the Pipeline.

        Args:
            transformers (Optional[Iterable[Transformer]]): Transformers to apply to both access and image.
            access_transformers (Optional[Iterable[Transformer]]): Transformers for modifying image access (e.g., coordinates).
                Cannot be used together with `transformers`.
            image_transformers (Optional[Iterable[Transformer]]): Transformers for modifying image data.
                Cannot be used together with `transformers`.
            reader_factory (Optional[Callable[[str], ImageReader]]): A function returning a reader for a given file name.
            auto_register_default_readers (bool): Whether to auto-register default readers on pipeline call.
        """
        if transformers:
            assert (
                    not access_transformers and not image_transformers
            ), "access_transformers and image_transformers may not be used together with transformers argument."
        self.transformers = list(transformers or [])

        # For convenience, access transformers and image transformers can be specified separately.
        # This saves users from thinking about the (maybe) unintuitive order of coordinate transformers.
        if access_transformers:
            # Access transformers are applied first, but in reverse order. Make sure that access_transformers is reversable first.
            access_transformers = list(access_transformers)
            self.transformers.extend(access_transformers[::-1])
        if image_transformers:
            self.transformers.extend(image_transformers)

        self.reader_factory = reader_factory or get_reader
        self.auto_register_default_readers = auto_register_default_readers

    def __call__(
            self,
            file_name: str,
            accessor: ImageAccessor,
            read_metadata: bool = True,
            **reader_kwargs: Any
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
        if self.auto_register_default_readers:
            from tiamat.readers import register_all_readers

            register_all_readers()
        reader = self.reader_factory(file_name, **reader_kwargs)
        if not accessor.metadata and read_metadata:
            accessor.metadata = reader.read_metadata()

        # backwards pass through the transformers to transform the accessor
        for transformer in self.transformers[::-1]:
            accessor = transformer.transform_access(accessor=accessor)

        # Read image data
        image_result = reader.read_image(accessor=accessor)

        # forward pass through the transformers
        for transformer in self.transformers:
            image_result = transformer.transform_image(image_result=image_result)
            if hasattr(transformer, "transform_metadata") and image_result.metadata is not None:
                image_result.metadata = transformer.transform_metadata(image_result.metadata)

        return image_result

    def read_metadata(self, file_name:str, **reader_kwargs:Any) -> ImageMetadata:
        """
        Reads and transforms image metadata.

        Args:
            file_name (str): Path to the image file.
            **reader_kwargs: Additional keyword arguments passed to the reader.

        Returns:
            ImageMetadata: The (possibly transformed) metadata.
        """
        reader = self.reader_factory(file_name, **reader_kwargs)
        metadata = reader.read_metadata()
        # backwards pass through the transformers to transform the accessor
        for transformer in self.transformers:
            if hasattr(transformer, "transform_metadata"):
                # check for transformers that do not implement transform_metadata
                metadata = transformer.transform_metadata(metadata=metadata)
        return metadata
