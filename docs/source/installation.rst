.. title:: Installation - tiamat

============
Installation
============

-------------------
System requirements
-------------------

- Python **≥ 3.8**
- Linux is the primary target and most thoroughly tested
- Optional: ``torch``/``onnxruntime`` if you want to run AI-based transformers

----------------------------
PyPI (recommended for users)
----------------------------

If you just want to try the toolkit locally

.. code-block:: bash

   pip install tiamat
   # optional companion modules
   pip install tiamat-ng

This installs the core runtime and the neuroglancer interface.

-----------------------------------
Source (recommended for developers)
-----------------------------------

If you want to develop or contribute to tiamat, clone the repository and install in editable mode

.. code-block:: bash

   git clone https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat.git
   cd tiamat
   pip install -e .

For working on companion modules, clone and ``pip install -e`` them too. This lets you iterate on transformers and tests locally.

-----------------------------------
Docker (recommended for production)
-----------------------------------

We publish Docker images in our
`JuGit registry <https://jugit.fz-juelich.de/groups/inm-1/bda/software/data_access/tiamat/-/container_registries>`__.
Running in a container is excellent if you need consistent dependencies or want to reproduce a deployment

.. code-block:: bash

   docker pull registry.fz-juelich.de/inm-1/bda/software/data_access/tiamat-ng:latest
   docker run -it --rm -p 8080:8080 tiamat-ng:latest

Containers also help when registering a pipeline configuration as a reproducible artifact (see :ref:`section on Performance & Reproducibility <performance>`).

**TODO**: add using a Dockerfile for building custom images
