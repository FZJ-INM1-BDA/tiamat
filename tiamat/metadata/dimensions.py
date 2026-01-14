"""
Specification of dimension in image data.
"""

X = "x"  # spatial x-axis
Y = "y"  # spatial y-axis
Z = "z"  # spatial z-axis
C = "c"  # generic channel
T = "t"  # time
RGB = "rgb"  # color channels (e.g., red, green, blue)
RGBA = "rgba"  # color channels (e.g., red, green, blue, alpha)
VECTOR = "vec"  # vector valued data (e.g., deformations)

# Order for spatial dimensions in the image (shape)
SPATIAL_DIMENSIONS = (Z, Y, X)

# Order of dimensions in image metadata (spacing, scale, ...)
META_DIMENSIONS = (X, Y, Z)
