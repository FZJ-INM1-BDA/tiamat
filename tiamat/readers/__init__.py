from .. import constants as const
from .factory import get_reader, register_reader

# Generic image formats
from .generic import GenericReader

register_reader(GenericReader)

# BigTiff
from .bigtiff import BigTiffReader

register_reader(BigTiffReader)

# HDF5
from .hdf5 import HDF5Reader

register_reader(HDF5Reader)

# nifti
from .nifti import NiftiReader

register_reader(NiftiReader)
