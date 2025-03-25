from .factory import register_reader

# Generic image formats
from .generic import GenericReader

register_reader(GenericReader)

# nifti
from .nifti import NiftiReader

register_reader(NiftiReader)
