"""
Reader for generic image formats.
"""
from .base import ImageReader


class GenericReader(ImageReader):
    def __init__(self, fname, cache_image=False):
        self.fname = fname
        self.cache_image = cache_image
        self._cached_image = None

    def _read_image(self):
        from imageio.v3 import imread

        image = self._cached_image if self._cached_image is not None else imread(self.fname)
        if self.cache_image:
            self._cached_image = image

        return image

    def get_crop(self, x, y, scale, z=None, c=None):
        from ._processing import access_image, rescale

        # Read
        image = self._read_image()
        # Access
        image = access_image(image, x=x, y=y, z=z, c=c)
        # TODO Derive interpolation for rescaling, perhabs from metadata.
        # Rescale
        image = rescale(image, scale=scale)
        return image

    def get_metadata(self):
        from tiamat import metadata as md

        # For generic images, we have to assume a lot and cannot derive much, even with reading the data.
        image = self._read_image()

        return md.ImageMetadata(image_type=md.IMAGE_TYPE_IMAGE,
                                shape=image.shape,
                                dtype=image.dtype,
                                value_range=(0, 255),
                                channel_interpretation=md.CHANNEL_INTERPRETATION_COLOR)
