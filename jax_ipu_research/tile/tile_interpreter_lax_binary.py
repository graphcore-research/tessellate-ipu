# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

import numpy as np
from jax import lax
from jax.core import Primitive, ShapedArray

from .tile_interpreter import register_ipu_tile_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_io_info,
)

_binary_primitive_to_vertex_basename: Dict[Primitive, Tuple[str, Any]] = {
    lax.add_p: ("ADD", None),
    lax.atan2_p: ("ATAN2", None),
    lax.div_p: ("DIVIDE", None),
    lax.eq_p: ("EQUAL", np.bool_),
    lax.ge_p: ("GREATER_THAN_EQUAL", np.bool_),
    lax.gt_p: ("GREATER_THAN", np.bool_),
    lax.le_p: ("LESS_THAN_EQUAL", np.bool_),
    lax.lt_p: ("LESS_THAN", np.bool_),
    lax.max_p: ("MAXIMUM", None),
    lax.min_p: ("MINIMUM", None),
    lax.mul_p: ("MULTIPLY", None),
    lax.ne_p: ("EQUAL", np.bool_),
    lax.pow_p: ("POWER", None),
    lax.rem_p: ("REMAINDER", None),
    lax.sub_p: ("SUBTRACT", None),
}
"""Binary JAX primitive to to IPU vertex basename (and output optional dtype).
"""


def make_binary1d_vertex_fullname(basename: str, dtype: Any) -> str:
    """Create the full vertex name from the basename and dtype."""
    ipu_dtype = from_numpy_dtype_to_ipu_type(dtype).name.lower()
    return f"popops::BinaryOp1D<popops::expr::BinaryOpType::{basename},{ipu_dtype}>"


def ipu_binary_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU binary primitive translation rule to IPU vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 2
    vertex_basename, outdtype = _binary_primitive_to_vertex_basename[p]
    vname = make_binary1d_vertex_fullname(vertex_basename, inavals[0].dtype)
    outaval = ShapedArray(inavals[0].shape, outdtype or inavals[0].dtype)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[
            make_ipu_vertex_io_info("in1", IpuVertexIOType.In, inavals[0]),
            make_ipu_vertex_io_info("in2", IpuVertexIOType.In, inavals[1]),
        ],
        outputs_info=[make_ipu_vertex_io_info("out", IpuVertexIOType.Out, outaval)],
        attributes_i32=[],
        attributes_f32=[],
    )
    return ipu_prim_info


# Register all supported JAX unary ops.
for p in _binary_primitive_to_vertex_basename.keys():
    register_ipu_tile_primitive(p, ipu_binary_primitive_translation)
