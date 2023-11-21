"""
Read a generic image.
"""
from tiamat.readers import get_reader

fname = "./data/Koala.jpg"
reader = get_reader(fname)

x = (0, None)
y = (0, None)

metadata = reader.get_metadata()
print(metadata)

crop = reader.get_crop(x, y, scale=1.0)
print(crop.shape)

crop = reader.get_crop(x, y, c=1, scale=1.0)
print(crop.shape)

crop = reader.get_crop((0, 100), y, c=1, scale=1.0)
print(crop.shape)

crop = reader.get_crop((0, 100), (0, 100), c=1, scale=1.0)
print(crop.shape)

crop = reader.get_crop((0, 100), (0, 100), c=(0, 2), scale=1.0)
print(crop.shape)

crop = reader.get_crop((0, 100), (0, 100), scale=0.5)
print(crop.shape)
