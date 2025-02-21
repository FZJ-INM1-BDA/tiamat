"""
Reading BigTiff images.
"""
import matplotlib.pyplot as plt
from tiamat.transformers.color import GrayscaleTransformer, LUTTransformer
from tiamat.transformers.access import FractionTransformer
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.readers import get_reader, bigtiff_zstack


fname = "/p/fastdata/bigbrains/scans/zstack-incoming/B20/0100_0199/B20_0197.zstack"
reader: bigtiff_zstack.ZstackBigTiffReader = get_reader(fname)

print(f"Slices: {reader.slices}")
print(f"Number of slices: {reader.num_slices}")
print(f"Pages: {reader.num_pages}")
print(f"Shapes: {reader.page_sizes}")
print(f"Scales: {reader.scales}")
print(f"dtype: {reader.dtype}")
print(f"Spacing: {reader.image_spacing}")
print(f"Metadata: {reader.read_metadata()}")

# Convert an image to grayscale, then apply a colormap.
# Let's also combine it with some coordinate transformers.
pipeline = Pipeline(
    access_transformers=[FractionTransformer()],
)
result_0 = pipeline(file_name=fname, accessor=ImageAccessor(x=(0, 0.5), y=(0, 0.5), scale=0.10))
result_1 = pipeline(file_name=fname, accessor=ImageAccessor(x=(0, 0.5), y=(0, 0.5), scale=0.50))

fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(result_0.image[..., 14])
ax1.imshow(result_1.image[..., 14])
plt.show()
