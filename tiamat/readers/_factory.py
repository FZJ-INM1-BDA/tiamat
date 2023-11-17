"""
Factory for readers.
"""

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


def get_reader(fname, **kwargs):
    import os

    _, ext = os.path.splitext(fname)
    # remove the leading dot
    ext = ext[1:]

    reader_cls = get_reader_for_file_type(ext)
    return reader_cls(fname, **kwargs)
