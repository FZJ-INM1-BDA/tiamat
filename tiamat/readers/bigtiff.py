"""
Reader for BigTiff.
"""
from .base import ImageReader


class BigTiffReader(ImageReader):
    def __init__(self, fname):
        self.fname = fname

    def get_crop(self, x, y, z, c, scale):
        ...

    def get_metadata(self):
        ...
