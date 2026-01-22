.. title:: Contributing to tiamat - tiamat

=======================
Contributing to tiamat
=======================

**We welcome contributions!**

This project follows the `git-flow workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_.
Releases are merged into ``master`` from ``develop`` on a regular basis. Please follow the guidelines below to contribute code to the tiamat project.

.. include:: ../../README.rst
    :start-after: development-start
    :end-before: development-end
.. include:: ../../README.rst
    :start-after: contributing-start
    :end-before: contributing-end

-------
Readers
-------

Readers must implement:

- ``check_file(path)`` → bool.
- ``read_metadata(path)`` → ``ImageMetadata``.
- ``read_image(accessor)`` → NumPy ndarray.

**Guidelines**:

- Keep ``check_file()`` fast to avoid slowing down pipelines.
- If the file has multiple resolutions, expose all scales in the metadata.
- Include tests for reading metadata only, without loading full image data.

------------
Transformers
------------

Each transformer should be purely functional and implement:

- ``transform_access(accessor, metadata)``
- ``transform_metadata(metadata)``
- ``transform_image(image)``

**Guidelines**:

- Keep transformations tile-local to maintain lazy evaluation.

----------
Interfaces
----------

To add an interface:

- Implement the adapter that calls ``Pipeline`` for incoming requests.
- Ensure responses match the client’s expected format, including metadata, image data, and error handling.
- Include a launcher script or helper to start the interface or mount it into a larger system.

------------
Tests and CI
------------

Testing is critical to maintain project quality. Contributions should include:

- Unit tests for ``check_file`` and ``read_metadata`` for all supported formats.
- Integration tests for the pipeline end-to-end.
