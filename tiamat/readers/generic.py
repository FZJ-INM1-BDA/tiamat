"""
Reader for generic image formats.
"""
from functools import cache
from .protocol import ImageReader
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class GenericReader(ImageReader):
    def __init__(self, fname, cache_image=False, image_spacing=None):
        self.fname = fname
        self.cache_image = cache_image
        self._cached_image = None
        self.image_spacing = image_spacing

    def _read_image(self):
        from imageio.v3 import imread

        image = self._cached_image if self._cached_image is not None else imread(self.fname)
        if self.cache_image:
            self._cached_image = image

        return image

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from .processing import access_and_rescale_image

        # Read, crop, and rescale.
        image = self._read_image()
        image = access_and_rescale_image(image=image, accessor=accessor)

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        # For generic images, we have to assume a lot and cannot derive much, even with reading the data.
        image = self._read_image()

        return md.ImageMetadata(image_type=md.IMAGE_TYPE_IMAGE,
                                shape=image.shape,
                                dtype=image.dtype,
                                file_path=self.fname,
                                value_range=(0, 255),
                                spacing=self.image_spacing,
                                channel_interpretation=md.CHANNEL_INTERPRETATION_COLOR)
