"""
Using pipeline as readers.
"""

from tiamat.pipeline import Pipeline
from tiamat.io import ImageAccessor
from tiamat.readers.stack import StackReader

pipeline = Pipeline(
    transformers=[],
    reader_factory=StackReader
)
result = pipeline(file_name=["./data/Koala.jpg", "./data/Koala.jpg"], accessor=ImageAccessor())
print(result.image.shape)
