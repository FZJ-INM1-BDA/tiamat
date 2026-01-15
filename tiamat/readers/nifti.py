from functools import cached_property
from typing import Any

import numpy as np

from tiamat.cache import instance_cache
from tiamat.io import ImageAccessor

from .protocol import ImageReader


class NiftiReader(ImageReader):
    """
    Reader for NIfTI image files (.nii / .nii.gz).

    Uses nibabel to load image data and metadata.
    """

    def __init__(self, fname: str) -> None:
        """
        Initialize the NiftiReader.

        Args:
            fname (str): Path to the NIfTI file.
        """
        self.fname = fname

    @cached_property
    def handle(self) -> Any:
        """
        Initialize the NiftiReader.

        Args:
            fname (str): Path to the NIfTI file.
        """
        import nibabel as nib

        return nib.load(self.fname)

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Read image data from the NIfTI file.

        Args:
            accessor (ImageAccessor): Provides metadata and access specifications.

        Returns:
            Any ImageResult with image data and metadata.
        """
        from .processing import access_and_rescale_image

        data = self.handle.get_fdata()

        image = access_and_rescale_image(image=data, metadata=self.read_metadata(), accessor=accessor)

        return image

    @instance_cache
    def read_metadata(self) -> Any:
        """
        Read metadata from the NIfTI file.

        Returns:
            Any: ImageMetadata object containing metadata such as dtype, shape, and file path.
        """
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
    def check_file(cls, fname: str) -> bool:
        """
        For parity with MemoryReader: declare support for array-like inputs.

        Args:
            fname: Object to test.

        Returns:
            True iff `fname` exposes a numpy-compatible array protocol.

        Note:
            In normal file-based selection this will evaluate to False (strings lack __array__),
            so ConstantReader won't be selected by accident in the file reader factory.
        """
        return fname.endswith(".nii") or fname.endswith(".nii.gz")
