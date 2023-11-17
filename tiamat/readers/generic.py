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
        from imageio import imread

        image = self._cached_image if self._cached_image is not None else imread(self.fname)
        if self.cache_image:
            self._cached_image = image

        return image

    def get_crop(self, x, y, scale, z=None, c=None):
        from ..helpers import access_image

        image = self._read_image()
        # TODO Scale image
        return access_image(image, x=x, y=y, z=z, c=c)

    def get_metadata(self):
        image = self._read_image()

        # TODO Define metadata
        return {"shape": image.shape, }
