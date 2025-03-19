"""
Factory for readers.
"""

from typing import List
from ..readers.protocol import ImageReader

_READER_REGISTRY: List[ImageReader] = []


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
    from tiamat.errors import UnknownFileError

    reader_by_priority = []
    for reader in _READER_REGISTRY:
        reader_priority = reader.check_file(fname)
        if isinstance(reader_priority, bool):
            if reader_priority:
                reader_priority = 0
            else:
                # reader does not support this file
                continue
        elif isinstance(reader_priority, (float, int)):
            if reader_priority < 0:
                # reader does not support this file
                continue
        else:
            raise RuntimeError("Reader must return bool or int from check_file")
        reader_by_priority.append((reader, reader_priority))
    if not reader_by_priority:
        raise UnknownFileError(f"Could not find reader for file {fname}")

    # sort by priority (descending)
    reader_by_priority.sort(key=lambda x: -x[1])
    top_priority = reader_by_priority[0][1]
    second_priority = reader_by_priority[1][1] if len(reader_by_priority) > 1 else None
    if second_priority is not None and top_priority == second_priority:
        raise UnknownFileError(f"Multiple readers with same priority for file {fname}")
    selected_reader = reader_by_priority[0][0]
    return selected_reader(fname, **kwargs)
