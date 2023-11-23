"""
Helper functions for running a tiamat processing pipeline.
"""
from typing import Iterable, Callable
from .transformers.protocol import Transformer
from .readers.protocol import ImageReader
from .readers.factory import get_reader
from .io import ImageAccessor, ImageResult


class Pipeline:
    """
    Processing pipeline to read and transform images.
    A pipeline can be used to read an image and transform it.
    """

    def __init__(self,
                 transformers: Iterable[Transformer] = None,
                 access_transformers: Iterable[Transformer] = None,
                 image_transformers: Iterable[Transformer] = None,
                 reader_factory: Callable[[str], ImageReader] = None):
        """
        Args:
            transformers (iterable of Transformer): A list of transformers to apply to each image.
            access_transformers(iterable of Transformer): A list of transformers to modify the access, defined from coordinates to reading the image.
                                                          May not be used with transformers argument.
            image_transformers(iterable of Transformer): A list of transformers to modify images, defined from read image to result.
                                                         May not be used with transformers argument.
            reader_factory (callable): A function returning a reader for a given file name.
        """
        if transformers:
            assert not access_transformers and not image_transformers, f"access_transformers and image_transformers may ne be used together with transformers argument."
        self.transformers = transformers or []

        # For convenience, access transformers and image transformers can be specified separately.
        # This saves users from thinking about the (maybe) unintuitive order of coordinate transformers.
        if access_transformers:
            # Access transformers are applied first, but in reverse order.
            self.transformers.extend(access_transformers[::-1])
        if image_transformers:
            self.transformers.extend(image_transformers)

        self.reader_factory = reader_factory or get_reader

    def __call__(self, accessor: ImageAccessor, read_metadata=True, **reader_kwargs) -> ImageResult:
        reader = self.reader_factory(accessor.file_name, **reader_kwargs)
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

        return image_result

