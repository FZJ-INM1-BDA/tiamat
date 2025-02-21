from .. import constants as const
from .factory import get_reader, register_reader

# Generic image formats
from .generic import GenericReader

register_reader(GenericReader)

# BigTiff
from .bigtiff import BigTiffReader

register_reader(BigTiffReader)
