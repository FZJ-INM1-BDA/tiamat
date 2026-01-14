"""
Custom error types used across the Tiamat image processing library.
"""


class TiamatError(Exception):
    """
    Base class for all custom exceptions in the Tiamat library.
    """

    pass


class UnknownFileError(TiamatError):
    """
    Raised when no compatible reader is found for a given file type or file name.
    """

    pass


class ReaderExistsError(TiamatError):
    """
    Raised when attempting to register a reader that already exists in the registry.
    """

    pass
