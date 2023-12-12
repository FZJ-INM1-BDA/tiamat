"""
Reading h5pli images.
"""
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.readers import get_reader, h5pli
from tiamat.transformers.access import FractionTransformer
from tiamat.transformers.color import GrayscaleTransformer, LUTTransformer

np.random.seed(42)

# generate dummy pli hdf5 file
tmp_file = tempfile.NamedTemporaryFile(suffix=".h5")
with h5py.File(tmp_file.name, "w") as f:
    shape = (100, 100)
    data = np.random.random(shape).astype(np.float32)
    f.create_dataset("Image", data=data)

    # pyramidal structure
    lvl = 0
    f["pyramid/00"] = f["Image"]
    while np.all(np.array(data.shape) > 1):
        lvl += 1
        data = data[::2, ::2]
        f.create_dataset(f"pyramid/{lvl:02}", data=data)

    # nesscessary metadata
    f["/Image"].attrs["usage"] = "PLI"
    f["/Image"].attrs["image_modality"] = "NTransmittance"
    f["/Image"].attrs["pixel_width"] = 1.0
    f["/Image"].attrs["pixel_height"] = 1.0
    f["/Image"].attrs["channels"] = "BrainId"
    f["/Image"].attrs["measurement_time"] = "1970-01-01 00:00:00"

# Testing Reader
reader: h5pli.H5Pli = get_reader(tmp_file.name)
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
result_0 = pipeline(file_name=tmp_file.name, accessor=ImageAccessor(x=(0, 0.5), y=(0, 0.5), scale=0.10))
result_1 = pipeline(file_name=tmp_file.name, accessor=ImageAccessor(x=(0, 0.5), y=(0, 0.5), scale=0.50))

fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(result_0.image)
ax1.imshow(result_1.image)
plt.show()
