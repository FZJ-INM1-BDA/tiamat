"""
Factory for managing and instantiating ImageReader classes.
"""

from ..readers.protocol import ImageReader

# Registry of registered ImageReader classes
_READER_REGISTRY: list[type[ImageReader]] = []


def register_reader(*reader_classes: type[ImageReader]) -> None:
    """
    Register one or more ImageReader classes to the global registry.

    Args:
        *reader_classes: One or more ImageReader subclasses to register.
    """

    for reader_class in reader_classes:
        if reader_class not in _READER_REGISTRY:
            _READER_REGISTRY.append(reader_class)


def get_reader_for_file_type(file_type):
    from tiamat.errors import UnknownFileError

    if file_type in _READER_REGISTRY:
        return _READER_REGISTRY[file_type]
    else:
        raise UnknownFileError(f"Could not find reader for file type {file_type}")


def get_reader_from_registry(reader_name: str) -> type[ImageReader]:
    """
    Get a registered ImageReader class by its class name.

    Args:
        reader_name: The name of the reader class.

    Returns:
        The ImageReader class matching the name.

    Raises:
        KeyError: If no reader with the given name is found.
    """
    for cls in _READER_REGISTRY:
        if cls.__name__ == reader_name:
            return cls
    raise KeyError(f"No reader found with name {reader_name}")


def get_reader(fname: str, auto_register_default_readers: bool = True, **kwargs) -> ImageReader:
    """
    Select and instantiate the best matching reader for a given file.

    Args:
        fname: Path to the image file.
        auto_register_default_readers: If True, automatically register default readers before selection.
        **kwargs: Additional keyword arguments passed to the reader constructor.

    Returns:
        An instantiated ImageReader object suitable for reading the file.

    Raises:
        UnknownFileError: If no reader can handle the file or multiple readers have the same priority.
    """
    from tiamat.errors import UnknownFileError

    if auto_register_default_readers:
        from . import register_all_readers

        register_all_readers()

    reader_by_priority = []
    reader_errors = []
    for reader in _READER_REGISTRY:
        try:
            reader_priority = reader.check_file(fname)
        except Exception as ex:
            # Ignore errors in check_file, but keep track of them for diagnostics
            reader_errors.append((reader, ex))
            continue
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
        raise UnknownFileError(f"Could not find reader for file {fname}. Reader errors: {reader_errors}")

    # sort by priority (descending)
    reader_by_priority.sort(key=lambda x: -x[1])
    top_priority = reader_by_priority[0][1]
    second_priority = reader_by_priority[1][1] if len(reader_by_priority) > 1 else None
    if second_priority is not None and top_priority == second_priority:
        raise UnknownFileError(f"Multiple readers with same priority for file {fname}")
    selected_reader = reader_by_priority[0][0]
    return selected_reader(fname, **kwargs)
