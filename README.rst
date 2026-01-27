.. image:: https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat/-/raw/develop/assets/logo512.png
   :height: 128px
   :align: center

========
tiamat
========

----

````````````````````````````````````````````````````````````````
Tiled Image Access, Manipulation, and Analysis Toolkit
````````````````````````````````````````````````````````````````

----

**tiamat** is a modular Python toolkit for accessing, transforming, and exposing large scientific image datasets.
It provides a flexible, pluggable pipeline model that separates data access (*readers*), transformation (*transformers*), and delivery (*interfaces*) — allowing on-the-fly, tool-agnostic image workflows without data duplication or format conversion.

Supported outputs include NumPy arrays, Napari, Neuroglancer, OpenSeadragon, and FUSE-mounted virtual filesystems.

----

========================
📑 Table of Contents
========================

1. `Quick Start <#quick-start>`_
2. `Installation <#installation>`_
3. `Core Concepts <#core-concepts>`_
4. `Examples <#examples>`_
5. `Development Guidelines <#development-guidelines>`_
6. `Contributing <#contributing>`_
7. `Data <#data>`_
8. `Acknowledgements <#acknowledgements>`_
9. `Contributors <#contributors>`_
10. `License <#license>`_

----

==================
🚀 Quick Start
==================

.. quickstart-start

.. code-block:: python

   from tiamat.pipeline import Pipeline
   from tiamat.io import ImageAccessor
   from tiamat.transformers.color import LUTTransformer

   # Create a pipeline with fractional coordinate access and a rainbow colormap
   pipeline = Pipeline(
       image_transformers=[LUTTransformer(colormap="rainbow")]
   )

   # Request the central 50% of the image
   accessor = ImageAccessor(x=(0.25, 0.75), y=(0.25, 0.75))
   result = pipeline("example_image.tif", accessor=accessor)

   # Get the transformed NumPy image and metadata
   image = result.image
   metadata = result.metadata

.. quickstart-end

----

==================
📦 Installation
==================

.. installation-start

Install the latest development version directly from GitLab:

.. code-block:: bash

   pip install git+https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat

.. installation-end

----

==================
🧠 Core Concepts
==================

.. coreconcepts-start

Tiamat defines a modular pipeline composed of:

- **Readers** – Load image data from formats like TIFF, NIfTI, HDF5, or memory arrays.
- **Transformers** – Apply dynamic, on-the-fly transformations (e.g., colormaps, axis reordering, tiling).
- **Interfaces** – Serve data to tools like Napari, Neuroglancer, OpenSeadragon, or directly as arrays.

This decoupled architecture allows you to:

- Build pipelines from reusable components
- Extend with custom readers or transformers
- Avoid costly format conversions

.. coreconcepts-end

----

============
📁 Examples
============

.. examples-start

See the ``examples/`` directory for usage demonstrations and pipeline configurations.

.. examples-end

----

==============================
🛠️ Development Guidelines
==============================

.. development-start

- Follow `PEP 561 <https://peps.python.org/pep-0561/>`_ type hinting
- Use `Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
- Formatting: ``flake8`` with line length 120
- Tests: ``pytest`` unit tests
- Feature development follows ``git-flow``

.. development-end

----

==================
🤝 Contributing
==================

**We welcome contributions!**

.. contributing-start

- Fork the repository and work on a feature branch
- Submit a Merge Request (MR) into ``develop``
- All contributions are reviewed and tested before merging

.. contributing-end

This project follows the `git-flow workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_.
Releases are merged into ``master`` from ``develop`` on a regular basis.

----

============
📂 Data
============

.. data-start

Some test and example datasets require ``git-lfs`` for download.

.. data-end

----

==================
👥 Contributors
==================

.. contributors-start

- Forschungszentrum Jülich, Institute of Neuroscience and Medicine (INM-1)
- Community contributors via GitLab merge requests

.. contributors-end

----

============
📄 License
============

.. license-start

Apache 2.0 – see `LICENSE <./LICENSE>`_ for details.

.. license-end

----

========================
🙏 Acknowledgements
========================

.. acknowledgements-start

This project received funding from the European Union's Horizon 2020 Research and Innovation Programme, grant agreement
101147319 (EBRAINS 2.0 Project), the Helmholtz Association port-folio theme "Supercomputing and Modeling for the Human
Brain", the Helmholtz Association's Initiative and Networking Fund through the Helmholtz International BigBrain
Analytics and Learning Laboratory (HIBALL) under the Helmholtz International Lab grant agreement InterLabs-0015,
from HELMHOLTZ IMAGING, a platform of the Helmholtz Information & Data Science Incubator [X-BRAIN, grant number:
ZT-I-PF-4-061], and from the Deutsche Forschungsgemeinschaft (DFG, German  Research Foundation) under the National
Research Data Infrastructure – NFDI 46/1 – 501864659.

.. acknowledgements-end
