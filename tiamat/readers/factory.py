"""
Factory for readers.
"""
from ..readers.protocol import ImageReader

_READER_REGISTRY = {}


def register_reader(file_types, reader_class, overwrite=False):
    from tiamat.errors import ReaderExistsError

    for file_type in file_types:
        if file_type in _READER_REGISTRY and not overwrite:
            raise ReaderExistsError(f"Error while registering {reader_class} for file type {file_type}:"
                                    f"Reader already exists (registered: {_READER_REGISTRY[file_type]})")
        _READER_REGISTRY[file_type] = reader_class


def get_reader_for_file_type(file_type):
    from tiamat.errors import UnknownFileError
    if file_type in _READER_REGISTRY:
        return _READER_REGISTRY[file_type]
    else:
        raise UnknownFileError(f"Could not find reader for file type {file_type}")


def get_reader(fname: str, **kwargs) -> ImageReader:
    import os
    from tiamat.errors import UnknownFileError

    filename = os.path.basename(fname)
    fragments = filename.split(".")

    # start from the most specific (contains all fragments)
    # to least specific (contains only the ext)
    while len(fragments) > 0:
        try:
            reader_cls = get_reader_for_file_type(".".join(fragments))
            return reader_cls(fname, **kwargs)
        except UnknownFileError:
            fragments.pop(0)
    raise UnknownFileError(f"Could not find reader for {fname}")
