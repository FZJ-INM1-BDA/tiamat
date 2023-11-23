"""
Reader for BigTiff.
"""
from .protocol import ImageReader


class BigTiffReader(ImageReader):
    def __init__(self, fname):
        self.fname = fname

    def read_image(self, x, y, z, c, scale):
        ...

    def read_metadata(self):
        ...
