"""
Simple example of using a pipeline.
"""

from tiamat.pipeline import Pipeline
from tiamat.array import Array

array = Array(file_name="./data/Koala.jpg", pipeline=Pipeline(), scale=1.0)
print(array.shape, array.dtype)
print("row", array[42].shape)
print("col", array[:, 42].shape)
print("channel", array[..., 0].shape)
print("multiple channels", array[..., 1:].shape)
print("negative indices", array[..., -1].shape)
