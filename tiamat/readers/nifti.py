from .protocol import ImageReader

DTYPE_MINMAX_DICT = {
    "int8": (-128, 127),
    "int16": (-32_768, 32_767),
    "int32": (-2_147_483_648, 2_147_483_647),

    "uint8": (0, 255),
    "uint16": (0, 65_525),
    "uint32": (0, 4_294_967_295),
}

class NiftiReader(ImageReader):

    def __init__(self, fname: str):
        self.fname = fname

    def read_image(self, accessor):
        import nibabel as nib
        
        from ._processing import access_and_rescale_image
        from ..io import ImageResult
        
        
        nii: nib.Nifti1Image = nib.load(self.fname)
        data = nii.get_fdata()
        
        image = access_and_rescale_image(data, accessor)
        
        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    def read_metadata(self):
        from tiamat import metadata as md
        import nibabel as nib

        nii: nib.Nifti1Image = nib.load(self.fname)
        assert len(nii.shape) == 2

        dtype = nii.get_data_dtype()
        value_range = DTYPE_MINMAX_DICT.get(str(dtype), (None, None))

        return md.ImageMetadata(image_type=md.IMAGE_TYPE_IMAGE,
                                shape=nii.shape,
                                value_range=value_range,
                                dtype=dtype,
                                file_path=self.fname)

    @classmethod
    def check_file(cls, fname: str):
        return fname.endswith(".nii") or fname.endswith(".nii.gz")
