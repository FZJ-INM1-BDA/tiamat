"""
Deformation field transformers.
"""
from functools import cached_property, cache
from typing import Tuple

from .protocol import Transformer
from ..io import ImageResult, ImageAccessor
from ..metadata import ImageMetadata

import numpy as np
import h5py as h5

class DeformationFieldTransformer(Transformer):
    
    def __init__(
            self,
            dfield_file: str,
            request_margin: int = 0,
        ):
        # TODO: Make file reading more generic
        self.dfield_file = dfield_file

        # Pixel margin around requested image to avoid resampling artifacts
        self.request_margin = request_margin

    @cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=self.shape,
            dtype=self.dtype,
            file_path=self.dfield_file,
            value_range=None,
            spacing=self.spacing,
            channel_dimension=2,
            channel_interpretation=md.CHANNEL_INTERPRETATION_COLOR,
        )

    @cached_property
    def file_handle(self) -> h5.File:
        import h5py

        return h5py.File(self.dfield_file)
    
    @cached_property
    def dtype(self):
        return self.file_handle["deformation"].dtype
    
    @cached_property
    def spacing(self) -> float:
        x_range = self.file_handle["xrange"][:2]
        y_range = self.file_handle["yrange"][:2]

        x_spacing = float(x_range[1] - x_range[0])
        y_spacing = float(y_range[1] - y_range[0])

        assert x_spacing == y_spacing, "DeformationFieldTransformer only supports isotropic spacing"

        return x_spacing
    
    @cached_property
    def offset(self) -> Tuple[float, float]:
        x_offset = self.file_handle["xrange"][0]
        y_offset = self.file_handle["yrange"][0]

        return (x_offset, y_offset)
    
    @cached_property
    def shape(self) -> tuple:
        return self.file_handle["deformation"].shape
    
    @staticmethod
    def apply_deformation(
        image: np.ndarray,
        image_spacing: float,
        image_origin: Tuple[float, float],
        dfield: np.ndarray,
        dfield_spacing: float,
        dfield_origin: Tuple[float, float],
        out_origin: Tuple[float, float],
        fill_value = 0,
        interpolation = "nearest",
    ):
        import SimpleITK as sitk
        from tiamat.readers.processing import SITK_INTERPOLATION_CODES

        # Convert image to SimpleITK format
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing((image_spacing, image_spacing))
        sitk_image.SetOrigin(image_origin)
        sitk_image.SetDirection([1.0, 0.0, 0.0, 1.0])  # Identity matrix for 2D

        # Convert deformation field to SimpleITK format
        displacement_field = sitk.GetImageFromArray(dfield.astype(np.float64), isVector=True)
        displacement_field.SetSpacing((dfield_spacing, dfield_spacing))
        displacement_field.SetOrigin(dfield_origin)
        displacement_field.SetDirection([1.0, 0.0, 0.0, 1.0])  # Identity matrix for 2D

        dfield_size = (int((dfield_spacing / image_spacing) * s) for s in displacement_field.GetSize())

        displacement_transform = sitk.DisplacementFieldTransform(displacement_field)

        # Define resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(dfield_size)  # Match the deformation field size
        resampler.SetOutputSpacing((image_spacing, image_spacing))  # Keep the image resolution
        resampler.SetOutputOrigin(out_origin)  # Match the deformation field's origin
        resampler.SetOutputDirection(displacement_field.GetDirection())  # Ensure correct spatial alignment
        resampler.SetInterpolator(SITK_INTERPOLATION_CODES[interpolation])
        resampler.SetTransform(displacement_transform)
        resampler.SetDefaultPixelValue(fill_value)

        # Execute deformation
        deformed_image = resampler.Execute(sitk_image)

        out_image = sitk.GetArrayFromImage(deformed_image)

        return out_image

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace
        import math

        from tiamat.readers.processing import access_image
        from tiamat.readers.processing import _prepare_coordinates, _resolve_coordinate

        # Read coordinates for requested frame
        x, y, *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)
        x_from, x_to = _resolve_coordinate(x, int(self.shape[1] * self.spacing / accessor.coordinate_spacing))
        y_from, y_to = _resolve_coordinate(y, int(self.shape[0] * self.spacing / accessor.coordinate_spacing))

        # Change to dfield metadata for access
        tmp_accessor = replace(accessor)
        tmp_accessor.metadata = self.read_metadata()
        tmp_accessor.x = (x_from, x_to)
        tmp_accessor.y = (y_from, y_to)

        # TODO: If x_from is not a multiple of the deformation field scale,
        # this probably also induces an offset through rounding

        dfield_crop = access_image(image=self.file_handle["deformation"], accessor=tmp_accessor, image_scale=(1 / self.spacing, 1 / self.spacing))

        scaled_margin = self.request_margin / accessor.scale

        # Determine frame for request
        locations_x = np.arange(0, dfield_crop.shape[1], 1) * self.spacing + x_from
        locations_x = dfield_crop[..., 1] + locations_x[None] + self.offset[0]
        locations_y = np.arange(0, dfield_crop.shape[0], 1) * self.spacing + y_from
        locations_y = dfield_crop[..., 0] + locations_y[:, None] + self.offset[1]
        min_x = np.min(locations_x) - scaled_margin
        max_x = np.max(locations_x) + scaled_margin
        min_y = np.min(locations_y) - scaled_margin
        max_y = np.max(locations_y) + scaled_margin

        accessor = replace(accessor)
        accessor.x = (math.floor(min_x), math.ceil(max_x) + 1)
        accessor.y = (math.floor(min_y), math.ceil(max_y) + 1)

        accessor.history[id(self)] = (x_from, y_from, dfield_crop)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        # TODO: Implement
        raise NotImplementedError

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        from tiamat.readers.processing import get_interpolation_for_accessor, _prepare_coordinates

        try:
            x_from, y_from, dfield_crop = image_result.accessor.history[id(self)]
        except KeyError:
            raise Exception("transform_access has to be called once before transform_image")

        accessor = image_result.accessor
        (x_from_input, _), (y_from_input, _), *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)

        image_origin = (
            float(x_from_input),
            float(y_from_input)
        )
        dfield_origin = (
            float(x_from) + self.offset[0],
            float(y_from) + self.offset[1]
        )
        out_origin = (
            float(x_from),
            float(y_from)
        )

        interpolation = get_interpolation_for_accessor(accessor=image_result.accessor)

        image_result.image = DeformationFieldTransformer.apply_deformation(
            image=image_result.image,
            image_spacing=(1 / image_result.accessor.scale),
            image_origin=image_origin,
            dfield=dfield_crop[..., ::-1], # Mirror dfield, ITK requires (x, y)
            dfield_spacing=self.spacing,
            dfield_origin=dfield_origin,
            out_origin=out_origin,
            fill_value=accessor.fill_value,
            interpolation=interpolation,
        )

        return image_result
