"""
Deformation field transformers.
"""
from typing import Any, Dict, Tuple, Callable
from tiamat.cache import instance_cached_property

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
            interpolation = 'linear',
            fill_value: int | float | None = None,
            xy_coordinates = True,
            reader_factory: Callable[[str], ImageReader] | None = None,
        ):
        """Creates an instance of DeformationFieldTransformer

        Parameters
        ----------
        dfield_file : str
            Path to file storing the deformation field
        request_margin : int, optional
            Pixel margin around requested image to avoid resampling artifacts, by default 2
        interpolation: str, optional
            Interpolation to be used to scale the deformation field to the requested scale
        """
        from tiamat.readers.factory import get_reader

        self.dfield_file = dfield_file
        self.reader_factory = reader_factory or get_reader

        self.xy_coordinates = xy_coordinates
        
        self.request_margin = request_margin
        self.interpolation = interpolation
        self.fill_value = fill_value

    @staticmethod
    def get_pixel_coordinates(
        dfield: np.ndarray,
        dfield_spacing: Tuple[float, float],
        dfield_scale: Tuple[int, int],
        dfield_origin: Tuple[float, float],
        image_spacing: Tuple[float, float],
        coord_dim: int = 2,
        xy: bool = True,
    ):
        """
        Convert vectors of physical coordinates to pixel coordinates in the requested image space at scale image_scale
        """

        # Determine if deformation vectors are xy or yx order
        if xy:
            x_i, y_i = 0, 1
        else:
            x_i, y_i = 1, 0

        # Convert relative deformation vectors to absolute physical coordinates
        offset_x = np.arange(0, dfield.shape[1]) * dfield_spacing[0] / dfield_scale[0] + dfield_origin[0]
        offset_y = np.arange(0, dfield.shape[0]) * dfield_spacing[1] / dfield_scale[1] + dfield_origin[1]

        phys_x = np.take(dfield, x_i, axis=coord_dim) + offset_x[None]
        phys_y = np.take(dfield, y_i, axis=coord_dim) + offset_y[:, None]

        # Convert absolute physical coordinates to pixel coordinates of the image
        coord_x = phys_x / image_spacing[0]
        coord_y = phys_y / image_spacing[1]

        return np.stack((coord_y, coord_x), axis=0)

    @instance_cached_property
    def dfield_file_handle(self) -> ImageReader:
        # TODO: Need to connect this reader to a reader_post_creation_hook somehow and change
        # to @property to manage caching of open file handles at a central location
        return self.reader_factory(self.dfield_file)


    @instance_cached_property
    def dfield_metadata(self) -> ImageMetadata:
        return self.dfield_file_handle.read_metadata()

    @instance_cached_property
    def dfield_spacing(self) -> Tuple[float, float]:
        from tiamat.readers.processing import expand_to_length

        return expand_to_length(self.dfield_file_handle.spacing, 2)

    @instance_cached_property
    def dfield_origin(self) -> Tuple[float, float]:
        dfield_origin = (0., 0.)
        if self.dfield_metadata.additional_metadata is not None:
            if 'dfield_origin' in self.dfield_metadata.additional_metadata.keys():
                dfield_origin = self.dfield_metadata.additional_metadata['dfield_origin']

        return dfield_origin


    @staticmethod
    def apply_deformation(
        image: np.ndarray,
        coordinates: np.ndarray,
        channel_dim: int = None,
        fill_value = 0,
        interpolation = "nearest",
    ):
        from scipy.ndimage import map_coordinates
        from tiamat.readers.processing import SCIPY_INTERPOLATION_CODES

        if channel_dim is None:
            out_image = map_coordinates(
                image,
                coordinates,
                order=SCIPY_INTERPOLATION_CODES[interpolation],
                mode='constant',
                cval=fill_value,
            )
        else:
            img_standard = np.moveaxis(image, channel_dim, -1)
            num_channels = img_standard.shape[-1]

            # Allocate memory for result
            result = np.empty((*coordinates.shape[1:], num_channels), dtype=image.dtype)

            # Loop over each channel
            for c in range(num_channels):
                image_channel = img_standard[..., c]

                result[..., c] = map_coordinates(
                    image_channel,
                    coordinates,
                    order=SCIPY_INTERPOLATION_CODES[interpolation],
                    mode='constant',
                    cval=fill_value,
                )

            out_image = np.moveaxis(result, -1, channel_dim)
    
        return out_image


    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace
        import math

        from tiamat.readers.processing import _prepare_coordinates, expand_to_length, rescale_shape
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        image_scale = expand_to_length(accessor.scale, 2)[:2]
        image_spacing = expand_to_length(accessor.metadata.spacing, 2)[:2]
        image_shape = accessor.metadata.shape

        coord_scale = expand_to_length(accessor.coordinate_scale, 2)

        # Input are pixel coordinates for scale=1.0
        prepared_coordinates = _prepare_coordinates(x=accessor.x, y=accessor.y)
        x, y = prepared_coordinates["x"], prepared_coordinates["y"]

        spatial_dims = accessor.metadata.spatial_dimensions
        x_from, x_to = resolve_coordinate_slice(x, image_shape[spatial_dims[-1]])
        y_from, y_to = resolve_coordinate_slice(y, image_shape[spatial_dims[-2]])

        scale_factor = (
            self.dfield_spacing[0] / image_spacing[0],
            self.dfield_spacing[1] / image_spacing[1]
        )

        # Request fitting scale of dfield that matches physical resolution of the request
        target_scale = (
            image_scale[0] * scale_factor[0],
            image_scale[1] * scale_factor[1],
        )

        tmp_coord_scale = (
            coord_scale[0] * scale_factor[0],
            coord_scale[1] * scale_factor[1],
        )

        # Target shape of coordinates is same as image shape
        target_shape = rescale_shape(
            ((y_to - y_from), (x_to - x_from)),
            (target_scale[0] / tmp_coord_scale[0] , target_scale[1] / tmp_coord_scale[1])
        )

        # Request more coordinates if dfield needs to be upscaled to avoid artifacts at corners
        tmp_margin = (
            tmp_coord_scale[0] if target_scale[0] > 1. else 0,
            tmp_coord_scale[1] if target_scale[1] > 1. else 0,
        )

        # Build temporary accessor to read dfield vectors
        tmp_accessor = replace(accessor)
        tmp_accessor.metadata = self.dfield_metadata
        tmp_accessor.scale = target_scale
        tmp_accessor.coordinate_scale = tmp_coord_scale
        tmp_accessor.interpolation = 'linear'
        tmp_accessor.fill_value = -1

        # Request margin if dfield needs upscaling to avoid artifacts
        tmp_accessor.x = (
            x_from - tmp_margin[0],
            x_to + tmp_margin[0]
        )
        tmp_accessor.y = (
            y_from - tmp_margin[1],
            y_to + tmp_margin[1]
        )

        # Read the corresponding crop from dfield
        dfield_crop = self.dfield_file_handle.read_image(tmp_accessor)

        # Crop the output to valid pixels not affected by the margin
        # TODO: Check if rounding up or down here
        offset = (
            math.floor((dfield_crop.image.shape[0] - target_shape[0]) / 2),
            math.floor((dfield_crop.image.shape[1] - target_shape[1]) / 2),
        )
        dfield_vectors = dfield_crop.image[
            offset[0]:(offset[0] + target_shape[0]),
            offset[1]:(offset[1] + target_shape[1])
        ]

        # Convert pixel coordinates to physical coordinates
        x_from_phys = x_from * image_spacing[0]
        y_from_phys = y_from * image_spacing[1]

        # Determine requested coordinates from deformation vectors
        coordinates = DeformationFieldTransformer.get_pixel_coordinates(
            dfield=dfield_vectors,
            dfield_spacing=self.dfield_spacing,
            dfield_scale=target_scale,
            dfield_origin=(x_from_phys + self.dfield_origin[0], y_from_phys + self.dfield_origin[1]),
            image_spacing=image_spacing,
            xy=self.xy_coordinates,
        )

        # Build requested frame from coordinates with margin
        scaled_margin = np.divide(self.request_margin, image_scale)

        min_xy = np.min(coordinates, axis=(1, 2)) - scaled_margin
        max_xy = np.max(coordinates, axis=(1, 2)) + scaled_margin
        min_x, max_x = min_xy[1], max_xy[1]
        min_y, max_y = min_xy[0], max_xy[0]

        # Store requested frame and coordinates in accessor
        accessor = replace(accessor)
        accessor.x = (math.floor(min_x), math.ceil(max_x) + 1)
        accessor.y = (math.floor(min_y), math.ceil(max_y) + 1)
        if self.fill_value is not None:
            accessor.fill_value = self.fill_value
        elif accessor.fill_value is None:
            accessor.fill_value = 0

        # Store pixel coordinates for transforming image
        coord_origin = np.array((accessor.y[0], accessor.x[0]), dtype=float)
        px_coordinates = (coordinates - coord_origin[:, np.newaxis, np.newaxis]) * np.array(image_scale)[:, np.newaxis, np.newaxis]
        accessor.history[id(self)] = px_coordinates

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace
        from tiamat.readers.processing import expand_to_length

        metadata = replace(metadata)

        image_spacing = expand_to_length(metadata.spacing, 2)

        out_shape = (
            int(self.dfield_metadata.spatial_shape[-2] * self.dfield_spacing[1] / image_spacing[1]),
            int(self.dfield_metadata.spatial_shape[-1] * self.dfield_spacing[0] / image_spacing[0]),
        )

        metadata.spatial_shape = (*metadata.shape[:-2], *out_shape)

        return metadata

    def transform_image(self, image_result: ImageResult, accessor: ImageAccessor) -> ImageResult:
        try:
            px_coordinates = accessor.history[id(self)]
        except KeyError:
            raise Exception("transform_access has to be called once before transform_image")

        if self.fill_value is None:
            fill_value = accessor.fill_value
        else:
            fill_value = self.fill_value

        ch_dim = image_result.metadata.channel_dimensions
        if len(ch_dim) > 1:
            raise Exception(f"Multiple channels {ch_dim} not supported in DeformationField Transformer")
        elif len(ch_dim) == 1:
            ch_dim = ch_dim[0]
        else:
            ch_dim = None

        spatial_dimensions = image_result.metadata.spatial_dimensions
        if len(spatial_dimensions) > 2:
            result_imgs = []
            # Loop over first spatial dimension
            for i in range(image_result.image.shape[spatial_dimensions[0]]):
                result_imgs.append(
                    DeformationFieldTransformer.apply_deformation(
                        image=image_result.image[i],
                        coordinates=px_coordinates,
                        channel_dim=ch_dim,
                        fill_value=fill_value,
                        interpolation=self.interpolation,
                    )
                )
            result_image = np.stack(result_imgs, axis=0)
            image_result.image = result_image
        else:
            image_result.image = DeformationFieldTransformer.apply_deformation(
                image=image_result.image,
                coordinates=px_coordinates,
                channel_dim=ch_dim,
                fill_value=fill_value,
                interpolation=self.interpolation,
            )

        return image_result

    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        from tiamat.serialization import get_reader_from_config

        return cls(
            dfield_file=args["dfield_file"],
            request_margin=args.get("request_margin", 2),
            interpolation=args.get("interpolation", 'linear'),
            fill_value=args.get("fill_value"),
            xy_coordinates=args.get("xy_coordinates", True),
            reader_factory=get_reader_from_config(args.get("reader_factory")),
        )
