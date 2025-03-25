"""
Reader for in-memory arrays.
"""

from .protocol import ImageReader
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class MemoryReader(ImageReader):
    def __init__(self, fname, **metadata_kwargs):
        self.image = fname
        self._cached_image = None
        self.metadata_kwargs = metadata_kwargs

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from .processing import access_and_rescale_image

        # Read, crop, and rescale.
        image = access_and_rescale_image(image=self.image, accessor=accessor)

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        return md.ImageMetadata(
            image_type=self.metadata_kwargs.get("image_type", md.IMAGE_TYPE_IMAGE),
            shape=self.image.shape,
            dtype=self.image.dtype,
            value_range=self.metadata_kwargs.get("value_range", (0, 255)),
            spacing=self.metadata_kwargs.get("spacing", None),
            channel_dimension=self.metadata_kwargs.get(
                "channel_dimension", None if len(self.image.shape) == 2 else 2
            ),
            channel_interpretation=self.metadata_kwargs.get(
                "channel_interpretation", md.CHANNEL_INTERPRETATION_COLOR
            ),
        )

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        # Checking for the generic __array__ attribute makes sure we can not only handle numpy arrays, but also other array-like objects, like HDF5 data
        return hasattr(fname, "__array__")
