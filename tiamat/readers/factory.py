"""
Factory for readers.
"""
from ..readers.protocol import ImageReader

_READER_REGISTRY = {}


def register_reader(reader_class):
    if reader_class not in _READER_REGISTRY:
        _READER_REGISTRY.append(reader_class)


def get_reader_for_file_type(file_type):
    from tiamat.errors import UnknownFileError
    if file_type in _READER_REGISTRY:
        return _READER_REGISTRY[file_type]
    else:
        raise UnknownFileError(f"Could not find reader for file type {file_type}")


def get_reader(fname: str, **kwargs) -> ImageReader:
    import os

    _, ext = os.path.splitext(fname)
    # remove the leading dot
    ext = ext[1:]

    reader_cls = get_reader_for_file_type(ext)
    return reader_cls(fname, **kwargs)
