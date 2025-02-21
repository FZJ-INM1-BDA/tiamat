"""
Reader for HDF5 pyramid files.
"""

from functools import cached_property, cache
from .protocol import ImageReader
import pytiff

from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class HDF5Reader(ImageReader):
    def __init__(self, fname, prefix="/"):
        self.fname = fname
        self.prefix = prefix

    @cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=self.shape,
            dtype=self.dtype,
            file_path=self.fname,
            value_range=self.value_range,
            spacing=self.image_spacing,
            channel_interpretation=md.CHANNEL_INTERPRETATION_COLOR,
            additional_metadata=self.attributes,
        )

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from .processing import access_and_rescale_image

        # Read, crop, and rescale.
        target_scale = accessor.scale
        available_scale, available_scale_index = self._find_scale(target_scale)
        print(target_scale, available_scale)

        pyramid_level = self.pyramid_datasets[available_scale_index]
        image = access_and_rescale_image(
            image=pyramid_level, accessor=accessor, image_scale=available_scale
        )

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cache
    def _find_scale(self, target_scale: float) -> tuple[float, int]:
        scales = [
            (scale, index)
            for index, scale in enumerate(self.scales)
            if scale >= target_scale
        ]
        # return either smallest, or smallest suitable scale
        if scales:
            return scales[-1]
        else:
            return (self.scales[-1], len(self.scales) - 1)

    @cached_property
    def file_handle(self) -> pytiff.Tiff:
        import h5py

        return h5py.File(self.fname)

    @cached_property
    def pyramid_root_dataset(self):
        import os

        return self.file_handle[os.path.join(self.prefix, "pyramid")]

    @cached_property
    def pyramid_datasets(self):
        keys = sorted(self.pyramid_root_dataset.keys())
        return [self.pyramid_root_dataset[key] for key in keys]

    @cached_property
    def pyramid_shapes(self) -> list[tuple[int, int]]:
        return [pyramid_level.shape for pyramid_level in self.pyramid_datasets]

    @cached_property
    def scales(self) -> list[float]:
        return [scale for scale in self.attributes["scales"]]

    @cached_property
    def shape(self) -> tuple:
        return self.pyramid_shapes[0]

    @cached_property
    def dtype(self):
        return self.pyramid_datasets[0].dtype

    @cached_property
    def image_spacing(self) -> tuple[float, float]:
        return self.attributes["spacing"]

    @cached_property
    def attributes(self) -> dict:
        return dict(self.pyramid_root_dataset.attrs)

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        import numpy as np

        dtype = self.dtype
        if np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
        elif np.issubdtype(dtype, np.floating):
            dtype_info = np.finfo(dtype)
        else:
            raise TypeError(f"Cannot determin value range for dtype {dtype}")

        return dtype_info.min, dtype_info.max

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        import os

        _, ext = os.path.splitext(fname)
        # return higher priority than generic reader, but lower than tiamat-pli reader
        return 5 if ext.lower() in (".hdf5", ".h5") else False
