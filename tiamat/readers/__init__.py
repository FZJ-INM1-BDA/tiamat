from .factory import get_reader, register_reader  # noqa: F401


def register_all_readers():
    # Generic image formats
    from .generic import GenericReader

    register_reader(GenericReader)

    # NIFTI
    from .nifti import NiftiReader

    register_reader(NiftiReader)

    # In-memory arrays
    from .memory import MemoryReader

    register_reader(MemoryReader)

    from .stack import ImageStackReader, VolumeStackReader

    register_reader(ImageStackReader)
    register_reader(VolumeStackReader)

    from .pipeline import PipelineReader

    register_reader(PipelineReader)

    from .zarr import OmeZarrReader

    register_reader(OmeZarrReader)
