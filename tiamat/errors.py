"""
Error handling.
"""


class UnknownFileError(Exception):
    """
    Error that is raised in case an unknown file type is encountered.
    """


class ReaderExistsError(Exception):
    """
    Error that is raised if a reader already exists during registration.
    """
