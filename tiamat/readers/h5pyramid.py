"""
Reader for Hdf5 files with a group of datasets interpreted as pyramid.
"""
import pathlib

import h5py

from .pyramid import Pyramid


class H5Pyramid(Pyramid):
    def __init__(
        self,
        fname: pathlib.Path,
        pyramid_path: str,
        metadata: dict | None,
        **h5py_kwargs: dict | None,
    ):
        fname = pathlib.Path(fname)
        if not fname.exists():
            raise FileNotFoundError(f"File {fname} does not exist.")

        self.fname = fname
        self.h5_file = h5py.File(fname, "r", **h5py_kwargs)
        super().__init__(
            # TODO: discuss
            pyramid=[
                self.h5_file[pyramid_path][key]
                for key in self.h5_file[pyramid_path].keys()
            ],
            metadata=metadata,
        )
