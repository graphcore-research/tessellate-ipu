# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Sequence

import cppimport
import numpy as np
from numpy.typing import DTypeLike

# Pybind11 extension import (and compilation if necessary).
# Explicit path is more robust to different `pip install` usages.
ext_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "tile_array_primitives_impl.cpp"))
tile_array_primitives_impl = cppimport.imp_from_filepath(
    ext_filename, "jax_ipu_experimental_addons.tile.tile_array_primitives_impl"
)

from .tile_array_primitives_impl import Base64Data, IpuShapedArray, IpuType  # noqa: E402, F401

_numpy_dtype_to_ipu_type = {
    np.dtype(np.bool_): IpuType.BOOL,
    np.dtype(np.uint8): IpuType.UNSIGNED_CHAR,
    np.dtype(np.uint16): IpuType.UNSIGNED_SHORT,
    np.dtype(np.uint32): IpuType.UNSIGNED_INT,
    np.dtype(np.int8): IpuType.CHAR,
    np.dtype(np.int16): IpuType.SHORT,
    np.dtype(np.int32): IpuType.INT,
    np.dtype(np.float16): IpuType.HALF,
    np.dtype(np.float32): IpuType.FLOAT,
}
"""Mapping from NumPy dtype to IPU datatype.
"""

_ipu_type_to_numpy_dtype = {
    IpuType.BOOL: np.dtype(np.bool_),
    IpuType.UNSIGNED_CHAR: np.dtype(np.uint8),
    IpuType.UNSIGNED_SHORT: np.dtype(np.uint16),
    IpuType.UNSIGNED_INT: np.dtype(np.uint32),
    IpuType.CHAR: np.dtype(np.int8),
    IpuType.SHORT: np.dtype(np.int16),
    IpuType.INT: np.dtype(np.int32),
    IpuType.HALF: np.dtype(np.float16),
    IpuType.FLOAT: np.dtype(np.float32),
}
"""Mapping from IPU type to NumPy dtype.
"""

_ipu_type_to_name = {
    IpuType.BOOL: "bool",
    IpuType.UNSIGNED_CHAR: "unsigned char",
    IpuType.UNSIGNED_SHORT: "unsigned short",
    IpuType.UNSIGNED_INT: "unsigned int",
    IpuType.CHAR: "signed char",
    IpuType.SHORT: "short",
    IpuType.INT: "int",
    IpuType.HALF: "half",
    IpuType.FLOAT: "float",
}
"""Mapping from IPU type to name (used in vertex naming convention).
"""


def from_numpy_dtype_to_ipu_type(v: Any) -> IpuType:
    """Convert from NumPy dtype to IPU type."""
    if isinstance(v, IpuType):
        return v
    return _numpy_dtype_to_ipu_type[np.dtype(v)]


def from_ipu_type_to_numpy_dtype(v: IpuType) -> Any:
    """Convert from IPU type to NumPy dtype."""
    if isinstance(v, np.dtype) or (isinstance(v, type) and issubclass(v, np.number)):
        return np.dtype(v)
    return _ipu_type_to_numpy_dtype[v]


def get_ipu_type_name(v: Any) -> str:
    """Get the (vertex dtype) name of an IPU type."""
    return _ipu_type_to_name[from_numpy_dtype_to_ipu_type(v)]


def make_ipu_shaped_array(shape: Sequence[int], dtype: DTypeLike) -> IpuShapedArray:
    """Convert to IPU shaped array."""
    return IpuShapedArray(shape, from_numpy_dtype_to_ipu_type(dtype))
