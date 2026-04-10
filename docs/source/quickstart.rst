.. _quickstart:
.. title:: Quickstart - tiamat

================
Quickstart Guide
================

Welcome to tiamat! This quickstart guide will help you get started with reading large images and processing them
using tiamat's pipeline and transformers.
In this guide, we assume you have **Python 3.8+ installed**.

-----------
Read a tile
-----------

1. Install core

.. code-block:: bash

      pip install tiamat

2. Start Python and run

.. code-block:: python

      from tiamat.pipeline import Pipeline
      from tiamat.io import ImageAccessor
      from tiamat.transformers.normalization import MinMaxNormalizationTransformer
      import matplotlib.pyplot as plt

      pipeline = Pipeline(transformers=[MinMaxNormalizationTransformer()])
      accessor = ImageAccessor(x=(0,512), y=(0,512))
      result = pipeline("path/to/big_image.tiff", accessor=accessor)

      print("shape:", result.image.shape)
      print("dtype:", result.image.dtype)
      print("metadata:", result.metadata)
      plt.imshow(result.image)
      plt.show()

What happened: the pipeline selected an appropriate reader for ``"big_image.tiff"``, read the specified region, normalized values, and returned a NumPy array and metadata object.

---------------------
Serve to Neuroglancer
---------------------

If you want a web UI, install the Neuroglancer interface and run

.. code-block:: bash

   pip install tiamat-ng
   tiamat-ng

Open Neuroglancer pointing to the server — Neuroglancer supports the precomputed format and tiamat’s Neuroglancer-compatible API allows streaming tiles directly to it.

--------------------------
Try a colormap transformer
--------------------------

You can create a transformer that converts single-channel data to RGB with a matplotlib colormap:

.. code-block:: python

   from tiamat.transformers import LUTTransformer

   pipeline = Pipeline(transformers=[LUTTransformer(colormap="viridis")])
   result = pipeline("example_image.zarr", accessor=ImageAccessor(x=(120,420), y=(300,400)))
   # result.image is now RGB

----------------------------------------------
Chaining transformers — a realistic pipeline
----------------------------------------------

Suppose you want:

1. Axis fix (some datasets are stored as YXZ, not XYZ).
2. Intensity normalization per tile.
3. Colormap.
4. On-tile segmentation using an ONNX model.

You build the pipeline in that order

.. code-block:: python

   from tiamat.pipeline import Pipeline
   from tiamat.transformers import AxisFixTransformer, NormalizationTransformer, LUTTransformer, AITransformer
   from my_models import ONNXCellSegmenter

   pipeline = Pipeline(transformers=[
       AxisFixTransformer(order=('x','y','z')),
       NormalizationTransformer(method='per_tile'),
       LUTTransformer(colormap='magma'),
       AITransformer(ONNXCellSegmenter("cells.onnx"))
   ])

   result = pipeline("large_dataset.zarr", accessor=ImageAccessor(x=(1000,1512), y=(2048,2560)))
