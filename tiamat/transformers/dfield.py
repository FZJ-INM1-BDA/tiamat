"""
Deformation field transformers.
"""
from typing import Any, Dict, Tuple, Callable

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

        reader_factory = reader_factory or get_reader
        self.reader = reader_factory(dfield_file)

        self.xy_coordinates = xy_coordinates

        self.meta = self.reader.read_metadata()

        self.dfield_origin = (0., 0.)
        if self.meta.additional_metadata is not None:
            if 'dfield_origin' in self.meta.additional_metadata.keys():
                self.dfield_origin = self.meta.additional_metadata['dfield_origin']
        
        self.request_margin = request_margin
        self.interpolation = interpolation
        self.fill_value = fill_value

    @staticmethod
    def get_coordinates(
        dfield: np.ndarray,
        dfield_scale: Tuple[int, int],
        dfield_origin: Tuple[float, float],
        coord_dim: int = 2,
        xy: bool = True,
    ):
        # Determine if deformation vectors are xy or yx order
        if xy:
            x_i, y_i = 0, 1
        else:
            x_i, y_i = 1, 0

        locations_x = np.arange(0, dfield.shape[1]) / dfield_scale[1] + dfield_origin[1]
        locations_x = np.take(dfield, x_i, axis=coord_dim) + locations_x[None]

        locations_y = np.arange(0, dfield.shape[0]) / dfield_scale[0] + dfield_origin[0]
        locations_y = np.take(dfield, y_i, axis=coord_dim) + locations_y[:, None]

        return np.stack((locations_y, locations_x), axis=0)

    @staticmethod
    def apply_deformation(
        image: np.ndarray,
        image_scale: Tuple[float, float],
        image_origin: Tuple[float, float],
        coordinates: np.ndarray,
        fill_value = 0,
        interpolation = "nearest",
    ):
        from scipy.ndimage import map_coordinates
        from tiamat.readers.processing import SCIPY_INTERPOLATION_CODES

        coordinates = (coordinates - np.array(image_origin)[:, np.newaxis, np.newaxis]) * np.array(image_scale)[:, np.newaxis, np.newaxis]

        out_image = map_coordinates(
            image,
            coordinates,
            order=SCIPY_INTERPOLATION_CODES[interpolation],
            cval=fill_value,
        )

        return out_image

    @property
    def shape(self) -> tuple:
        from tiamat.readers.processing import expand_to_length

        dfield_scale = expand_to_length(self.meta.scales[0], 2)

        shape = (
            int((self.meta.shape[0] / dfield_scale[0])),
            int((self.meta.shape[1] / dfield_scale[1])),
        )

        return shape

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace
        import math

        from tiamat.readers.processing import _prepare_coordinates, expand_to_length
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        # dfield_scale = expand_to_length(self.meta.scales[0], 2)
        image_scale = expand_to_length(accessor.scale, 2)
        coord_scale = expand_to_length(accessor.coordinate_scale, 2)

        # Read coordinates for requested frame
        prepared_coordinates = _prepare_coordinates(x=accessor.x, y=accessor.y)
        x, y = prepared_coordinates["x"], prepared_coordinates["y"]
        dfield_shape = self.shape
        x_from, x_to = resolve_coordinate_slice(x, dfield_shape[1] * coord_scale[1])
        y_from, y_to = resolve_coordinate_slice(y, dfield_shape[0] * coord_scale[0])

        # Change access metadata to dfield
        tmp_accessor = replace(accessor)
        tmp_accessor.metadata = self.meta
        tmp_accessor.x = (x_from, x_to)
        tmp_accessor.y = (y_from, y_to)
        tmp_accessor.fill_value = 0

        # Read the corresponding crop from dfield
        dfield_crop = self.reader.read_image(tmp_accessor)
        dfield_vectors = dfield_crop.image

        # Determine requested coordinates from deformation vectors
        coordinates = DeformationFieldTransformer.get_coordinates(
            dfield=dfield_vectors,
            dfield_scale=image_scale,
            dfield_origin=(y_from + self.dfield_origin[0], x_from + self.dfield_origin[1]),
            xy=self.xy_coordinates,
        )

        # Build requested frame from coordinates with margin
        scaled_margin = np.divide(self.request_margin, accessor.scale)

        min_xy = np.min(coordinates, axis=(1, 2)) - scaled_margin
        max_xy = np.max(coordinates, axis=(1, 2)) + scaled_margin
        min_x, max_x = min_xy[1], max_xy[1]
        min_y, max_y = min_xy[0], max_xy[0]

        # Store requested frame and coordinates in accessor
        accessor = replace(accessor)
        accessor.x = (math.floor(min_x), math.ceil(max_x) + 1)
        accessor.y = (math.floor(min_y), math.ceil(max_y) + 1)
        accessor.fill_value = 0 if accessor.fill_value is None else accessor.fill_value
        accessor.history[id(self)] = coordinates # Cache coordinates so we don't need to read twice

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(metadata)
        metadata.shape = self.shape

        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        from tiamat.readers.processing import get_interpolation_for_accessor, _prepare_coordinates, expand_to_length

        try:
            coordinates = image_result.accessor.history[id(self)]
        except KeyError:
            raise Exception("transform_access has to be called once before transform_image")

        image_scale = expand_to_length(image_result.accessor.scale, 2)
        accessor = image_result.accessor
        prepared_coordinates = _prepare_coordinates(x=accessor.x, y=accessor.y)
        (x_from_input, _), (y_from_input, _) = prepared_coordinates["x"], prepared_coordinates["y"]
        image_origin = (
            float(y_from_input),
            float(x_from_input)
        )

        if self.fill_value is None:
            fill_value = accessor.fill_value
        else:
            fill_value = self.fill_value

        image_result.image = DeformationFieldTransformer.apply_deformation(
            image=image_result.image,
            image_scale=image_scale,
            image_origin=image_origin,
            coordinates=coordinates,
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
            reader_factory=get_reader_from_config(args.get("reader_factory")),
        )
