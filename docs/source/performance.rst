.. _performance:
.. title:: Performance & Reproducibility - tiamat

=============================
Performance & Reproducibility
=============================

Tiamat is designed to provide flexible, on-demand data access for large-scale imaging datasets. This section discusses 
when to use tiamat over traditional conversion workflows, performance considerations, and best practices for ensuring
reproducible pipelines.

----------------
Why not convert?
----------------

The RTI architecture avoids many limitations of traditional conversion-based workflows. Instead of pre-generating 
entire datasets in each tool's preferred format, tiamat reads datasets directly in their original form and
performs all processing on demand. Tiamat is best when:

* datasets are **huge** and you don’t want duplicates,
* datasets are **changing** (during preprocessing, annotation, or iterative scanning),
* you need **on-demand transforms** (different users want different normalizations or colormaps),
* or you want **a single, reproducible pipeline** that can feed many clients.

If you have a stable dataset with heavy global traffic and compute/storage is cheap, converting may still be optimal.

-----------------------------------
Profiling and performance guidance
-----------------------------------

tiamat prioritizes flexibility over raw throughput.
Benchmarks indicate that dynamic pipelines incur <15% latency overhead compared to precomputed formats like 
Neuroglancer-precomputed or Zarr-pyramids, while saving up to 80% storage by avoiding duplication.

.. tip::

    `tiamat-benchmarking <https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat_benchmarking>`__ contains scripts for performance profiling.

If you plan to deploy tiamat at scale:

- Record representative user interactions in the viewer to mimic real-world access patterns.
- Use local caching for frequently requested tiles.
- If using HDF5-backed formats on network storage, test thread-safety and random access behavior — HDF5/parallel HDF5 has known caveats when used concurrently.

------------------------
Pipeline reproducibility
------------------------

If your pipeline performs analyses that must be reproducible, don’t rely on local state. Instead:

- Pin the **tiamat version** (container image digest or PyPI version).
- Pin transformer versions and model artifacts (hash the ONNX file).
- Save the pipeline configuration as a JSON or YAML spec and version it alongside the code repository.
