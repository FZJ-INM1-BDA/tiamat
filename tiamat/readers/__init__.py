from .factory import get_reader, register_reader
from .. import constants as const

# Generic image formats
from .generic import GenericReader
register_reader(const.FILE_TYPES_GENERIC, GenericReader)

# BigTiff
from .bigtiff import BigTiffReader
register_reader(const.FILE_TYPES_BIGTIFF, BigTiffReader)
