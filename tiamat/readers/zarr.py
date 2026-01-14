"""
Reader for OME-Zarr (Zarr v3 / OME-NGFF ≥0.5).
"""

from __future__ import annotations

import json
import os
import zipfile
from functools import cached_property
from typing import Any, Dict, List, Tuple

import numpy as np

# We try to import zarr here. If it is not available, this reader will not register itself as available (see check_file)
try:
    import zarr  # pylint: disable=import-error
    from zarr.storage import ZipStore  # pylint: disable=import-error

    _ZARR_AVAILABLE = True
except Exception:  # pragma: no cover
    zarr = None  # pylint: disable=import-error
    ZipStore = None  # pylint: disable=import-error
    _ZARR_AVAILABLE = False

from tiamat.cache import instance_cache
from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.readers.protocol import ImageReader


def _axis_to_dim(axis_name: str, md) -> Any:
    """
    Map ome-zarr specific axis labels to tiamat dimensions.
    """
    n = axis_name.lower()
    return {
        "x": md.dimensions.X,
        "y": md.dimensions.Y,
        "z": md.dimensions.Z,
        "c": md.dimensions.C,
        "t": md.dimensions.T,
    }.get(n, md.dimensions.C)


class OmeZarrReader(ImageReader):
    def __init__(self, fname: str):
        self.fname = fname

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        if not _ZARR_AVAILABLE:
            # zarr not installed, we cannot do anything.
            return False

        lower = str(fname).lower()
        is_zip = lower.endswith(".zarr.zip") or lower.endswith(".ome.zarr.zip")
        is_dir_like = lower.endswith(".zarr") or lower.endswith(".ome.zarr")

        if not (is_zip or is_dir_like):
            return False

        # Function to load zarr.json content
        def load_zarr_json(json_bytes: bytes) -> dict | None:
            try:
                return json.loads(json_bytes)
            except Exception:
                return None

        zarr_data = None

        if is_zip:
            # Check inside ZIP
            try:
                with zipfile.ZipFile(fname, "r") as zf:
                    if "zarr.json" not in zf.namelist():
                        return False
                    with zf.open("zarr.json") as f:
                        zarr_data = load_zarr_json(f.read())
            except Exception:
                return False

        elif is_dir_like:
            # Check inside directory
            zarr_json_path = os.path.join(fname, "zarr.json")
            if not os.path.exists(zarr_json_path):
                return False
            try:
                with open(zarr_json_path, "r", encoding="utf-8") as f:
                    zarr_data = json.load(f)
            except Exception:
                return False

        # Validate JSON structure
        if not isinstance(zarr_data, dict):
            return False

        attributes = zarr_data.get("attributes")
        if not isinstance(attributes, dict):
            return False

        if "ome" in attributes:
            return 10  # or True, depending on your scoring scheme
        return False

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        from tiamat.readers.processing import access_and_rescale_image

        axes = self.axes_names
        # Normalize accessor.scale to a sequence
        raw_scale = accessor.scale
        if isinstance(raw_scale, (float, int)):
            # single float: treat as uniform on all spatial axes
            target_xy = (float(raw_scale), float(raw_scale))
            target_z = float(raw_scale)
        else:
            # assume sequence
            seq = list(raw_scale)
            if len(seq) >= 3:
                # interpret as (Z, Y, X)
                target_z = float(seq[0])
                target_xy = (float(seq[1]), float(seq[2]))
            elif len(seq) == 2:
                target_z = float(min(seq))
                target_xy = (float(seq[0]), float(seq[1]))
            else:
                # fallback: treat everything as uniform
                target_xy = (float(seq[0]), float(seq[0]))
                target_z = float(seq[0])

        per_axis_target: List[float] = []
        for a in axes:
            if a == "x":
                per_axis_target.append(target_xy[1])
            elif a == "y":
                per_axis_target.append(target_xy[0])
            elif a == "z":
                per_axis_target.append(target_z)
            else:
                per_axis_target.append(1.0)

        chosen_level = self._select_level_from_target(per_axis_target)
        level_factors = self._level_factors(chosen_level)
        arr = self._get_level_array(chosen_level)

        return access_and_rescale_image(
            image=arr,
            metadata=self.read_metadata(),
            accessor=accessor,
            image_scale=tuple(level_factors),
        )

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        dims = [_axis_to_dim(ax, md) for ax in self.axes_names]
        vmin, vmax = self.value_range

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=self.shape,
            dtype=self.dtype,
            file_path=self.fname,
            value_range=(vmin, vmax),
            spacing=self.pixel_spacing,
            dimensions=dims,
            scales=self.scales,
            additional_metadata=self._root_attrs,
        )

    # ---------------- Cached / internal helpers ---------------- #

    @cached_property
    def _root_group(self):
        if not _ZARR_AVAILABLE:
            raise ImportError("zarr is not installed. Install `zarr` to read OME-Zarr data.")
        lower = self.fname.lower()
        if lower.endswith(".zip"):
            if ZipStore is None:
                raise ImportError("zarr ZipStore is not available.")
            if not os.path.exists(self.fname):
                raise FileNotFoundError(self.fname)
            store = ZipStore(self.fname, mode="r")
            return zarr.open_group(store=store, mode="r")
        if not (os.path.isdir(self.fname) or os.path.exists(self.fname)):
            raise FileNotFoundError(self.fname)
        return zarr.open_group(self.fname, mode="r")

    @cached_property
    def _root_attrs(self) -> Dict[str, Any]:
        # Zarr v3: metadata is surfaced via .attrs (backed by zarr.json)
        attrs = dict(getattr(self._root_group, "attrs", {}) or {})
        return attrs

    @cached_property
    def _ome_meta(self) -> Dict[str, Any]:
        # Prefer "ome" namespace if present, otherwise treat root attrs as container
        ome = self._root_attrs.get("ome")
        if isinstance(ome, dict) and ome:
            return ome
        return self._root_attrs

    @cached_property
    def _multiscales(self) -> List[Dict[str, Any]]:
        ms = self._ome_meta.get("multiscales")
        if not isinstance(ms, list) or not ms:
            raise ValueError("OME-Zarr attributes lack a valid 'multiscales' list.")
        return ms

    @cached_property
    def _pyramid(self) -> Dict[str, Any]:
        return self._multiscales[0]

    @cached_property
    def axes_names(self) -> List[str]:
        axes = self._pyramid.get("axes")
        if isinstance(axes, list) and axes:
            if isinstance(axes[0], str):
                return [a.lower() for a in axes]
            if isinstance(axes[0], dict):
                return [str(a.get("name", "")).lower() for a in axes]
        # Fallback using array rank
        a0 = self._get_level_array(0)
        rank = a0.ndim
        if rank == 2:
            return ["y", "x"]
        if rank == 3:
            return ["c", "y", "x"]
        if rank == 4:
            return ["z", "c", "y", "x"]
        if rank >= 5:
            return ["t", "z", "c", "y", "x"][-rank:]
        return ["y", "x"]

    @cached_property
    def _datasets(self) -> List[Dict[str, Any]]:
        ds = self._pyramid.get("datasets")
        if not isinstance(ds, list) or not ds:
            raise ValueError("OME-Zarr multiscales.datasets is missing or empty.")
        return ds

    @cached_property
    def dtype(self) -> np.dtype:
        return self._get_level_array(0).dtype

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self._get_level_array(0).shape)

    @cached_property
    def value_range(self) -> Tuple[float | int, float | int]:
        dt = self.dtype
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
        elif np.issubdtype(dt, np.floating):
            info = np.finfo(dt)
        else:
            raise TypeError(f"Cannot determine value range for dtype {dt}")
        return info.min, info.max

    @cached_property
    def pixel_spacing(self) -> Tuple[float, ...]:
        # Read level-0 coordinateTransformations -> scale vector; fallback 1.0
        axes = self.axes_names
        transforms = self._datasets[0].get("coordinateTransformations", [])
        scale_vec = None
        for t in transforms or []:
            if isinstance(t, dict) and t.get("type") == "scale":
                v = t.get("scale")
                if isinstance(v, list) and len(v) == len(axes):
                    scale_vec = [float(x) for x in v]
                    break
        if scale_vec is None:
            return tuple([1.0] * len(axes))
        return tuple(scale_vec)

    @cached_property
    def scales(self) -> List[float]:
        # XY downsample factors per level (relative to level 0); fallback heuristic if XY unknown
        axes = self.axes_names
        y_idx = axes.index("y") if "y" in axes else None
        base = self._get_level_array(0).shape
        base_y = base[y_idx] if y_idx is not None else None

        factors: List[float] = []
        for i, _ in enumerate(self._datasets):
            a = self._get_level_array(i)
            if base_y is not None:
                f = base_y / float(a.shape[y_idx])
                factors.append(float(f))
            else:
                factors.append(1.0 if i == 0 else max(2.0, factors[-1] * 2))
        return factors

    @instance_cache
    def _get_level_array(self, level: int):
        ds = self._datasets[level]
        rel = ds.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"Invalid dataset path at level {level}.")
        if rel not in self._root_group:
            raise ValueError(f"Dataset '{rel}' not found in Zarr group.")
        return self._root_group[rel]

    @cached_property
    def _spatial_indices(self) -> List[int]:
        order = []
        for a in ("z", "y", "x"):
            if a in self.axes_names:
                order.append(self.axes_names.index(a))
        return order

    @instance_cache
    def _level_factors(self, level: int) -> List[float]:
        base = self._get_level_array(0).shape
        cur = self._get_level_array(level).shape
        factors: List[float] = []
        spatial = {"z", "y", "x"}
        for i, a in enumerate(self.axes_names):
            if a in spatial:
                factors.append(float(base[i]) / float(cur[i]))
            else:
                factors.append(1.0)
        return factors

    def _select_level_from_target(self, per_axis_target: List[float]) -> int:
        spatial_idx = self._spatial_indices

        def meets_target(factors: List[float]) -> bool:
            return all(factors[i] >= per_axis_target[i] for i in spatial_idx)

        candidates = []
        for lvl in range(len(self._datasets)):
            f = self._level_factors(lvl)
            if meets_target(f):
                score = max((f[i] for i in spatial_idx), default=1.0)
                candidates.append((score, lvl))

        if candidates:
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]

        # Fall back to coarsest
        coarsest = 0
        coarsest_score = -np.inf
        for lvl in range(len(self._datasets)):
            f = self._level_factors(lvl)
            score = max((f[i] for i in spatial_idx), default=1.0)
            if score > coarsest_score:
                coarsest_score = score
                coarsest = lvl
        return coarsest
