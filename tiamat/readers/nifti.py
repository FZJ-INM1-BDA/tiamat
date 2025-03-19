from .protocol import ImageReader

class NiftiReader(ImageReader):

    def __init__(self, fname: str):
        self.fname = fname

    def read_image(self, accessor):
        import nibabel as nib
        
        from ._processing import access_and_rescale_image
        from ..io import ImageResult
        
        
        nii: nib.Nifti1Image = nib.load(self.fname)
        data = nii.get_fdata()

        # assert np.max(data) < 257

        # new_data = np.astype(data, np.uint8)
        
        image = access_and_rescale_image(data, accessor)
        print(accessor.metadata)
        
        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    def read_metadata(self):
        from tiamat import metadata as md
        import nibabel as nib
        import numpy as np

        nii: nib.Nifti1Image = nib.load(self.fname)
        assert len(nii.shape) == 2
        data = nii.get_fdata()

        return md.ImageMetadata(image_type=md.IMAGE_TYPE_IMAGE,
                                shape=nii.shape,
                                value_range=(np.min(data), np.max(data)),
                                dtype=nii.get_data_dtype(),
                                file_path=self.fname)

    @classmethod
    def check_file(cls, fname: str):
        return fname.endswith(".nii") or fname.endswith(".nii.gz")
