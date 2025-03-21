"""
Reading HDF5 images.
"""

from tiamat.transformers.color import GrayscaleTransformer, LUTTransformer
from tiamat.pipeline import Pipeline
from tiamat.readers import register_reader, hdf5
from tiamat.array import Array

fname = "./data/B20_0001_Pyramid.hdf5"
register_reader(hdf5.HDF5Reader)

# Convert an image to grayscale, then apply a colormap.
pipeline = Pipeline(
    image_transformers=[GrayscaleTransformer(), LUTTransformer(color_map="plasma")]
)
array = Array(
    file_name=fname,
    pipeline=pipeline,
    scale=1.0,
)
print(array.shape)
print(array.ndim)
print(array.scale)

array_pyramid = Array.create_arrays_for_scales(file_name=fname, pipeline=pipeline)
print([array.scale for array in array_pyramid])
print([array.shape for array in array_pyramid])

pipeline = Pipeline(
    image_transformers=[
        GrayscaleTransformer(),
    ]
)
array = Array(
    file_name=fname,
    pipeline=pipeline,
    scale=1.0,
)
print(array.shape)
print(array[:20, :20].shape)
