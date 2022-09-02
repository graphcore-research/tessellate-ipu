# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

from jax import lax
from jax.core import Primitive, ShapedArray

from .tile_interpreter import register_ipu_tile_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_io_info,
)

_unary_primitive_to_vertex_basename: Dict[Primitive, str] = {
    lax.abs_p: "ABSOLUTE",
    lax.asin_p: "ASIN",
    # lax.cbrt_p: "CBRT",
    lax.ceil_p: "CEIL",
    lax.cos_p: "COS",
    lax.erf_p: "ERF",
    lax.exp_p: "EXPONENT",
    lax.expm1_p: "EXPONENT_MINUS_ONE",
    lax.floor_p: "FLOOR",
    lax.is_finite_p: "IS_FINITE",
    lax.log_p: "LOGARITHM",
    lax.log1p_p: "LOGARITHM_ONE_PLUS",
    lax.neg_p: "NEGATE",
    lax.sin_p: "SIN",
    lax.tan_p: "TAN",
    lax.tanh_p: "TANH",
    lax.round_p: "ROUND",
    lax.rsqrt_p: "RSQRT",
    lax.sqrt_p: "SQRT",
    # lax.integer_pow_p: "",
}
"""Mapping from unary JAX primitives to IPU vertex basename.
"""


def make_unary1d_vertex_fullname(basename: str, dtype: Any) -> str:
    """Create the full vertex name from the basename and dtype."""
    ipu_dtype = from_numpy_dtype_to_ipu_type(dtype).name.lower()
    return f"popops::UnaryOp1D<popops::expr::UnaryOpType::{basename},{ipu_dtype}>"


def ipu_unary_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU unary primitive translation rule to IPU vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    vertex_basename = _unary_primitive_to_vertex_basename[p]
    vname = make_unary1d_vertex_fullname(vertex_basename, inavals[0].dtype)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_io_info("in", IpuVertexIOType.In, inavals[0])],
        outputs_info=[make_ipu_vertex_io_info("out", IpuVertexIOType.Out, inavals[0])],
        attributes_u32=[],
        attributes_f32=[],
    )
    return ipu_prim_info


# Register all supported JAX unary ops.
for p in _unary_primitive_to_vertex_basename.keys():
    register_ipu_tile_primitive(p, ipu_unary_primitive_translation)
