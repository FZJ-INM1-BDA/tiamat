from functools import cached_property

import numpy as np

from tiamat.cache import instance_cache
from tiamat.io import ImageAccessor

from .protocol import ImageReader


class NiftiReader(ImageReader):

    def __init__(self, fname: str):
        self.fname = fname

    @cached_property
    def handle(self):
        import nibabel as nib

        return nib.load(self.fname)

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        from ..io import ImageResult
        from .processing import access_and_rescale_image

        data = self.handle.get_fdata()

        image = access_and_rescale_image(image=data, metadata=self.read_metadata(), accessor=accessor)

        return image

    @instance_cache
    def read_metadata(self):
        from tiamat import metadata as md

        dtype = self.handle.get_data_dtype()
        value_range = md.get_dtype_limits(dtype)

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=self.handle.shape,
            value_range=value_range,
            dtype=dtype,
            file_path=self.fname,
        )

    @classmethod
    def check_file(cls, fname: str):
        return fname.endswith(".nii") or fname.endswith(".nii.gz")
