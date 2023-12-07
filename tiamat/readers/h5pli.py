"""
Reader for h5pli corresponding to the PLI.json.
"""
# pylint: disable=no-member

import pathlib
from functools import cache, cached_property

import h5py
import numpy as np

from .h5pyramid import H5Pyramid

CHECK_ATTRIBUTES = [
    "channels",
    "stitched",
    "image_modality",
    "pixel_width",
    "measurement_time",
]

MODALITIES_META = {
    "Raw": {
        "channels": 1,
        "dtype": np.uint8,
        "min_intensity": 0,
        "max_intensity": 2**16 - 1,
    },
    "RawPreview": {
        "channels": 3,
        "dtype": np.uint8,
        "min_intensity": 0,
        "max_intensity": 255,
    },
    "Calibrated": {
        "channels": 1,
        "dtype": np.float32,
        "min_intensity": 0.0,
        "max_intensity": 2**16 - 1,
    },
    "Transmittance": {
        "channels": 1,
        "dtype": np.uint16,
        "min_intensity": 0,
        "max_intensity": 2**16 - 1,
    },
    "NTransmittance": {
        "channels": 1,
        "dtype": np.float32,
        "min_intensity": 0.0,
        "max_intensity": 2.0,
    },
    "Direction": {
        "channels": 1,
        "dtype": np.float32,
        "min_intensity": 0.0,
        "max_intensity": 180.0,
    },
    "Retardation": {
        "channels": 1,
        "dtype": np.float32,
        "min_intensity": 0.0,
        "max_intensity": 1.0,
    },
    "Inclination": {
        "channels": 1,
        "dtype": np.float32,
        "min_intensity": -90.0,
        "max_intensity": 90.0,
    },
    "Trel": {
        "channels": 1,
        "dtype": np.float32,
        "min_intensity": 0.0,
        "max_intensity": 1.0,
    },
    "FOM": {
        "channels": 3,
        "dtype": np.uint8,
        "min_intensity": 0,
        "max_intensity": 255,
    },
    "Mask": {
        "channels": 1,
        "dtype": np.uint8,
        "min_intensity": 0,
        "max_intensity": 255,
    },
}


class H5Pli(H5Pyramid):
    def __init__(self, fname: pathlib.Path, **h5py_kwargs):
        fname = pathlib.Path(fname)

        if not fname.is_file():
            raise FileNotFoundError(f"File {fname} does not exist.")
        if not self.valid_file(fname):
            raise ValueError(f"File {fname} is not a valid h5pli file.")

        self.fname = fname
        metadata, dtype = self._init_attributes(fname)
        metadata["resolution_x"] = metadata["pixel_width"]
        metadata["resolution_y"] = metadata["pixel_height"]
        metadata["value_range"] = self._value_range(metadata, dtype)

        super().__init__(
            fname=fname,
            pyramid_path="/pyramid",
            metadata=metadata,
            **h5py_kwargs,
        )

    @staticmethod
    def valid_file(fname: str) -> bool:
        try:
            with h5py.File(fname, "r") as h5file:
                if "Image" not in h5file:
                    return False, '"Image" not in h5file'
                elif "pyramid" not in h5file:
                    return False, '"pyramid" not in h5file'

                for attr in CHECK_ATTRIBUTES:
                    if attr not in h5file["Image"].attrs:
                        return False, f'"{attr}" not in h5file["Image"].attrs'
                return True

        except OSError:
            raise ValueError(f"File {fname} is not a valid hdf5 file.")

    @staticmethod
    def _value_range(metadata, dtype):
        if "image_modality" in metadata:
            return (
                MODALITIES_META[metadata["image_modality"]]["min_intensity"],
                MODALITIES_META[metadata["image_modality"]]["max_intensity"],
            )
        else:
            if "min_intensity" in metadata and "max_intensity" in metadata:
                return metadata["min_intensity"], metadata["max_intensity"]
            elif "max_intensity" in metadata:  # older standard
                return 0.0, metadata["max_intensity"]
            else:
                if np.issubdtype(dtype, np.integer):
                    dtype_info = np.iinfo(dtype)
                elif np.issubdtype(dtype, np.floating):
                    dtype_info = np.finfo(dtype)
                else:
                    raise TypeError(f"Cannot determin value range for dtype {dtype}")

                return dtype_info.min, dtype_info.max

    @staticmethod
    def _init_attributes(fname):
        with h5py.File(fname, "r") as h5_file:
            attrs = dict(h5_file["Image"].attrs)
            dtype = h5_file["Image"].dtype

        for key, value in attrs.items():
            if isinstance(value, bytes):
                attrs[key] = value.decode("utf-8")
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, bytes):
                    attrs[key] = value.astype(str)
        return attrs, dtype

    @cached_property
    def image_spacing(self) -> tuple[float, float]:
        return (
            self.metadata["pixel_width"],
            self.metadata["pixel_height"],
        )

    @cached_property
    def is_tiled(self) -> bool:
        return not self.metadata["stitched"]
