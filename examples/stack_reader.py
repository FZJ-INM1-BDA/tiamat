"""
Using pipeline as readers.
"""

from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.readers.stack import ImageStackReader

pipeline = Pipeline(transformers=[], reader_factory=ImageStackReader)
result = pipeline(file_name=["./data/Koala.jpg", "./data/Koala.jpg"], accessor=ImageAccessor())
print(result.image.shape)
