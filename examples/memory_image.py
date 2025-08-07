"""
Read a generic image.
"""

from imageio.v3 import imread
from tiamat.readers import get_reader
from tiamat.io import ImageAccessor

array = imread("./data/Koala.jpg")
reader = get_reader(array)

x = (0, None)
y = (0, None)

metadata = reader.read_metadata()
print(metadata)

# read the entire image
crop = reader.read_image(ImageAccessor(scale=1.0, metadata=metadata))
print(crop.image.shape)

crop = reader.read_image(ImageAccessor(x=x, y=y, scale=1.0, metadata=metadata))
print(crop.image.shape)

crop = reader.read_image(ImageAccessor(x=x, y=x, c=1, scale=1.0, metadata=metadata))
print(crop.image.shape)

crop = reader.read_image(
    ImageAccessor(x=(0, 100), y=y, c=1, scale=1.0, metadata=metadata)
)
print(crop.image.shape)

crop = reader.read_image(
    ImageAccessor(x=(0, 100), y=(0, 100), c=1, scale=1.0, metadata=metadata)
)
print(crop.image.shape)

crop = reader.read_image(
    ImageAccessor(x=(0, 100), y=(0, 100), c=(0, 2), scale=1.0, metadata=metadata)
)
print(crop.image.shape)

crop = reader.read_image(
    ImageAccessor(x=(0, 100), y=(0, 100), scale=0.5, metadata=metadata)
)
print(crop.image.shape)
