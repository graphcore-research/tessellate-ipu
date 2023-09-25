# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from jax.core import Primitive, ShapedArray
from jax.lax import (
    GatherScatterMode,
    ScatterDimensionNumbers,
    scatter_add_p,
    scatter_max_p,
    scatter_min_p,
    scatter_mul_p,
    scatter_p,
)

from tessellate_ipu.core import (
    IpuTileMapEquation,
    make_ipu_vertex_attributes,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_in_info,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_name_templated,
    register_ipu_tile_primitive,
)
from tessellate_ipu.utils import DTypeLike

_scatter_primitive_to_properties: Dict[Primitive, Any] = {
    scatter_add_p: (1, "ADD"),
    scatter_min_p: (None, "MIN"),
    scatter_max_p: (None, "MAX"),
    scatter_mul_p: (None, "MUL"),
}
"""IPU translation properties for every JAX LAX scatter primitive.
"""


def make_scatter_vertex_fullname(dtype: DTypeLike, opname: str, scale: Any) -> str:
    """Generate popops Scatter/MultiUpdateOp vertex name."""
    opname = f"popops::Operation::{opname}"
    if scale is not None:
        basename = "popops::ScaledMultiUpdateOp"
        return make_ipu_vertex_name_templated(basename, dtype, dtype, False, opname)
    else:
        basename = "popops::MultiUpdateOp"
        return make_ipu_vertex_name_templated(basename, dtype, False, opname)


def check_scatter_dimension_numbers(dimension_numbers: ScatterDimensionNumbers):
    """Check `scatter` dimension_numbers is supported on TessellateIPU.

    At the moment: basically only supporting a single configuration!
    We need to expand on this at some point!
    """
    dim_numbers_default = ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
    )
    if dimension_numbers != dim_numbers_default:
        raise NotImplementedError(f"TessellateIPU `scatter` only support dimension numbers: {dim_numbers_default}.")


def ipu_scatter_op_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `scatter_xx` primitive translation rule to IPU vertex.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input data + start indices arrays.
        attributes: Gather operator attributes
    Returns:
        IPU tile map primitive structure.
    """
    # TODO: query for JAX device.
    num_context_workers = 6

    assert len(inavals) == 3
    assert attributes is not None
    operand, scatter_indices, updates = inavals
    # Extract scatter attributes
    dimension_numbers = attributes["dimension_numbers"]
    # Default values from JAX LAX interface.
    indices_are_sorted = attributes.get("indices_are_sorted", False)
    unique_indices = attributes.get("unique_indices", False)
    mode = attributes.get("mode", GatherScatterMode.PROMISE_IN_BOUNDS)

    # Check scatter attributes are supported by TessellateIPU.
    assert operand.ndim == 1
    assert scatter_indices.ndim == 2
    assert operand.dtype == updates.dtype
    assert scatter_indices.dtype == np.uint32, "TessellateIPU `scatter` only supports `uint32` indices."
    if indices_are_sorted:
        logging.warning("TessellateIPU `scatter` operation does not make use of `indices_are_sorted` argument.")
    if unique_indices:
        logging.warning("TessellateIPU `scatter` operation does not make use of `unique_indices` argument.")
    assert (
        mode == GatherScatterMode.PROMISE_IN_BOUNDS
    ), "Only `PROMISE_IN_BOUNDS` scatter mode supported in TessellateIPU."
    check_scatter_dimension_numbers(dimension_numbers)

    # Primitive translation properties.
    scale, opname = _scatter_primitive_to_properties[p]
    vname = make_scatter_vertex_fullname(operand.dtype, opname, scale)
    # Construct poplibs MultiSlice vertex attributes.
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(
        baseOffset=0,  # unused?
        numBaseElements=operand.size,  # Number of elements in input.
        maxElementsPerWorker=int(np.ceil(operand.size / num_context_workers)),
        regionSize=1,  # TODO: understand?
        indicesAreSorted=False,
    )

    # Constant `scale` (if required by the vertex).
    constants_info = []
    if scale is not None:
        constants_info = [make_ipu_vertex_constant_info("scale", np.array(scale, dtype=operand.dtype), vertex_dim2=-1)]
    # For now: need to do it manually at the Python `tile_map` level.
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[
            make_ipu_vertex_inout_info("baseT", operand),
            make_ipu_vertex_in_info("offsets", scatter_indices),
            make_ipu_vertex_in_info("subT", updates),
        ]
        + constants_info,
        outputs_info=[make_ipu_vertex_inout_info("baseT", operand)],
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


def ipu_scatter_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `scatter` primitive translation rule to IPU vertex.

    Note: using a specific translation, as the poplibs vertex is different.
    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input data + start indices arrays.
        attributes: Gather operator attributes
    Returns:
        IPU tile map primitive structure.
    """
    # TODO: query for JAX device.
    num_context_workers = 6

    assert len(inavals) == 3
    assert attributes is not None
    operand, scatter_indices, updates = inavals
    # Extract scatter attributes
    dimension_numbers = attributes["dimension_numbers"]
    # Default values from JAX LAX interface.
    indices_are_sorted = attributes.get("indices_are_sorted", False)
    unique_indices = attributes.get("unique_indices", False)
    mode = attributes.get("mode", GatherScatterMode.PROMISE_IN_BOUNDS)

    # Check scatter attributes are supported by TessellateIPU.
    assert operand.ndim == 1
    assert scatter_indices.ndim == 2
    assert operand.dtype == updates.dtype
    assert scatter_indices.dtype == np.uint32, "TessellateIPU `scatter` only supports `uint32` indices."
    if indices_are_sorted:
        logging.warning("TessellateIPU `scatter` operation does not make use of `indices_are_sorted` argument.")
    if unique_indices:
        logging.warning("TessellateIPU `scatter` operation does not make use of `unique_indices` argument.")
    assert (
        mode == GatherScatterMode.PROMISE_IN_BOUNDS
    ), "Only `PROMISE_IN_BOUNDS` scatter mode supported in TessellateIPU."
    check_scatter_dimension_numbers(dimension_numbers)

    vname = make_ipu_vertex_name_templated("popops::MultiUpdate", operand.dtype)
    # Construct poplibs MultiSlice vertex attributes.
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(
        baseOffset=0,  # unused?
        numBaseElements=operand.size,  # Number of elements in input.
        maxElementsPerWorker=int(np.ceil(operand.size / num_context_workers)),
        regionSize=1,  # TODO: understand?
        indicesAreSorted=False,
        splitSingleRegion=False,  # Split regions between threads? TODO: understand!
    )
    # For now: need to do it manually at the Python `tile_map` level.
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[
            make_ipu_vertex_inout_info("baseT", operand),
            make_ipu_vertex_in_info("offsets", scatter_indices),
            make_ipu_vertex_in_info("subT", updates),
        ],
        outputs_info=[make_ipu_vertex_inout_info("baseT", operand)],
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


# Register JAX `scatter` primitives with update op.
for p in _scatter_primitive_to_properties.keys():
    register_ipu_tile_primitive(p, ipu_scatter_op_primitive_translation)
# Specific translation for the simple `scatter` case
register_ipu_tile_primitive(scatter_p, ipu_scatter_primitive_translation)
