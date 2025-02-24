"""
Reading BigTiff images.
"""

import matplotlib

import matplotlib.pyplot as plt
from tiamat.transformers.color import GrayscaleTransformer, LUTTransformer
from tiamat.transformers.access import FractionTransformer
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.readers import register_reader, get_reader, hdf5

fname = "./data/B20_0001_Pyramid.hdf5"
register_reader(hdf5.HDF5Reader)
reader: hdf5.HDF5Reader = get_reader(fname)

print(f"Shapes: {reader.pyramid_shapes}")
print(f"Scales: {reader.scales}")
print(f"dtype: {reader.dtype}")
print(f"Spacing: {reader.image_spacing}")
print(f"Metadata: {reader.read_metadata()}")

# Convert an image to grayscale, then apply a colormap.
# Let's also combine it with some coordinate transformers.
pipeline = Pipeline(
    access_transformers=[FractionTransformer()],
    image_transformers=[GrayscaleTransformer(), LUTTransformer(color_map="plasma")],
)
result_0 = pipeline(
    file_name=fname, accessor=ImageAccessor(x=(0, 1.0), y=(0, 1.0), scale=0.10)
)
result_1 = pipeline(
    file_name=fname, accessor=ImageAccessor(x=(0, 0.5), y=(0, 0.5), scale=0.50)
)

fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(result_0.image)
ax1.imshow(result_1.image)
plt.show()
