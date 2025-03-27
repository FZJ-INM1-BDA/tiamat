"""
Deformation field transformers.
"""
from functools import cached_property, cache
from typing import Tuple, Iterable, Callable

from tiamat.readers.protocol import ImageReader
from tiamat.transformers.protocol import Transformer
from tiamat.io import ImageResult, ImageAccessor
from tiamat.metadata import ImageMetadata

import numpy as np


class DeformationFieldTransformer(Transformer):
    
    def __init__(
            self,
            dfield_file: str,
            request_margin: int = 2,
            reader_factory: Callable[[str], ImageReader] | None = None,
        ):
        """Creates an instance of DeformationFieldTransformer

        Parameters
        ----------
        dfield_file : str
            Path to file storing the deformation field
        request_margin : int, optional
            Pixel margin around requested image to avoid resampling artifacts, by default 2
        """
        from tiamat.readers.factory import get_reader

        reader_factory = reader_factory or get_reader
        self.reader = reader_factory(dfield_file)

        self.meta = self.reader.read_metadata()
        self.dfield_origin = (0., 0.)
        if self.meta.additional_metadata is not None:
            if 'dfield_origin' in self.meta.additional_metadata.keys():
                self.dfield_origin = self.meta.additional_metadata['dfield_origin']
        
        self.request_margin = request_margin

    @staticmethod
    def get_coordinates(
        dfield: np.ndarray,
        dfield_scale: Tuple[int, int],
        dfield_origin: Tuple[float, float],
        coord_dim: int = 2,
        yx: bool = True,
    ):
        # Determine if deformation vectors are xy or yx order
        if yx:
            x_i, y_i = 1, 0
        else:
            x_i, y_i = 0, 1
        locations_x = np.arange(0, dfield.shape[1]) / dfield_scale[1] + dfield_origin[1]
        locations_x = np.take(dfield, x_i, axis=coord_dim) + locations_x[None]

        locations_y = np.arange(0, dfield.shape[0]) / dfield_scale[0] + dfield_origin[0]
        locations_y = np.take(dfield, y_i, axis=coord_dim) + locations_y[:, None]

        return np.stack((locations_y[None], locations_x[None]), axis=0)

    @staticmethod
    def apply_deformation(
        image: np.ndarray,
        image_scale: float,
        image_origin: Tuple[float, float],
        dfield: np.ndarray,
        dfield_scale: float,
        dfield_origin: Tuple[float, float],
        out_origin: Tuple[float, float],
        fill_value = 0,
        interpolation = "nearest",
    ):
        import SimpleITK as sitk
        from tiamat.readers.processing import SITK_INTERPOLATION_CODES

        # Convert image to SimpleITK format
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing([1 / s for s in image_scale])
        sitk_image.SetOrigin(image_origin)
        sitk_image.SetDirection([1.0, 0.0, 0.0, 1.0])  # Identity matrix for 2D

        # Convert deformation field to SimpleITK format
        displacement_field = sitk.GetImageFromArray(dfield.astype(np.float64), isVector=True)
        displacement_field.SetSpacing([1 / s for s in dfield_scale])
        displacement_field.SetOrigin(dfield_origin)
        displacement_field.SetDirection([1.0, 0.0, 0.0, 1.0])  # Identity matrix for 2D

        dfield_size = (int((image_scale[i] / dfield_scale[i]) * s) for i, s in enumerate(displacement_field.GetSize()))

        displacement_transform = sitk.DisplacementFieldTransform(displacement_field)

        # Define resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(dfield_size)  # Match the deformation field size
        resampler.SetOutputSpacing([1 / s for s in image_scale])  # Keep the image resolution
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

        from tiamat.readers.processing import _prepare_coordinates, _resolve_coordinate, expand_to_length

        # TODO: Account for spacing
        dfield_scale = expand_to_length(self.meta.scales[0], 2)
        coord_scale = expand_to_length(accessor.coordinate_scale, 2)

        # Read coordinates for requested frame
        x, y, *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)
        x_from, x_to = _resolve_coordinate(x, int((self.meta.shape[1] / dfield_scale[1]) * coord_scale[1]))
        y_from, y_to = _resolve_coordinate(y, int((self.meta.shape[0] / dfield_scale[0]) * coord_scale[0]))

        # Change access metadata to dfield
        tmp_accessor = replace(accessor)
        tmp_accessor.metadata = self.meta
        tmp_accessor.x = (x_from, x_to)
        tmp_accessor.y = (y_from, y_to)

        # Read the corresponding crop from dfield
        dfield_crop = self.reader.read_image(tmp_accessor)

        # Determine requested coordinates from deformation vectors
        coordinates = DeformationFieldTransformer.get_coordinates(
            dfield=dfield_crop,
            dfield_scale=dfield_scale,
            dfield_origin=(y_from + self.dfield_origin[0], x_from + self.dfield_origin[1]),
            yx=True, # TODO: Read this somehow from file
        )

        # Build requested frame from coordinates with margin
        scaled_margin = self.request_margin / accessor.scale

        min_x = np.min(coordinates[1]) - scaled_margin
        max_x = np.max(coordinates[1]) + scaled_margin
        min_y = np.min(coordinates[0]) - scaled_margin
        max_y = np.max(coordinates[0]) + scaled_margin

        # Store requested frame and coordinates in accessor
        accessor = replace(accessor)
        accessor.x = (math.floor(min_x), math.ceil(max_x) + 1)
        accessor.y = (math.floor(min_y), math.ceil(max_y) + 1)
        accessor.fill_value = 0 if accessor.fill_value is None else accessor.fill_value
        accessor.history[id(self)] = (x_from, y_from, dfield_crop)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        # TODO: Implement
        raise NotImplementedError

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        from tiamat.readers.processing import get_interpolation_for_accessor, _prepare_coordinates, expand_to_length

        try:
            x_from, y_from, dfield_crop = image_result.accessor.history[id(self)]
        except KeyError:
            raise Exception("transform_access has to be called once before transform_image")

        dfield_scale = expand_to_length(self.meta.scales[0], 2)
        image_scale = expand_to_length(image_result.accessor.scale, 2)

        accessor = image_result.accessor
        (x_from_input, _), (y_from_input, _), *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)

        image_origin = (
            float(x_from_input),
            float(y_from_input)
        )
        dfield_origin = (
            float(x_from) + self.dfield_origin[1],
            float(y_from) + self.dfield_origin[0]
        )
        out_origin = (
            float(x_from),
            float(y_from)
        )

        interpolation = get_interpolation_for_accessor(accessor=image_result.accessor)

        image_result.image = DeformationFieldTransformer.apply_deformation(
            image=image_result.image,
            image_scale=image_scale,
            image_origin=image_origin,
            dfield=dfield_crop[..., ::-1], # Mirror dfield, ITK requires (x, y)
            dfield_scale=dfield_scale,
            dfield_origin=dfield_origin,
            out_origin=out_origin,
            fill_value=accessor.fill_value,
            interpolation=interpolation,
        )

        return image_result
