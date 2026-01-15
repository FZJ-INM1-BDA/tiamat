"""
Simple example of using a pipeline.
"""

from functools import lru_cache, partial

from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.readers.generic import GenericReader


@lru_cache(maxsize=10)
def reader_cache(cls, file_name, **reader_kwargs):
    # This print should appear only once
    print(f"Create reader from class {cls} for file {file_name}")
    return cls(file_name, **reader_kwargs)


cached_reader_factory = partial(reader_cache, GenericReader)

# Define a simple pipeline
pipeline = Pipeline(
    reader_factory=cached_reader_factory,
)

# Use the same accessor to access the pipeline multiple times
accessor = ImageAccessor(scale=0.5)

result = pipeline(file_name="./data/Koala.jpg", accessor=accessor)
result = pipeline(file_name="./data/Koala.jpg", accessor=accessor)
result = pipeline(file_name="./data/Koala.jpg", accessor=accessor)
result = pipeline(file_name="./data/Koala.jpg", accessor=accessor)
result = pipeline(file_name="./data/Koala.jpg", accessor=accessor)

print(result.image.shape)
