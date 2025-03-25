from functools import cached_property
from .protocol import ImageReader


class NiftiReader(ImageReader):

    def __init__(self, fname: str):
        self.fname = fname

    @cached_property
    def handle(self):
        import nibabel as nib

        return nib.load(self.fname)

    def read_image(self, accessor):
        from ._processing import access_and_rescale_image
        from ..io import ImageResult

        data = self.handle.get_fdata()

        image = access_and_rescale_image(data, accessor)

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

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
