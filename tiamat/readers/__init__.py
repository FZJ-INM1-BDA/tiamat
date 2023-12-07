from .. import constants as const
from .factory import get_reader, register_reader

# Generic image formats
from .generic import GenericReader
register_reader(const.FILE_TYPES_GENERIC, GenericReader)

# BigTiff
from .bigtiff import BigTiffReader
register_reader(const.FILE_TYPES_BIGTIFF, BigTiffReader)

# H5Pli
from .h5pli import H5Pli
register_reader(const.FILE_TYPES_HDF5, H5Pli)
