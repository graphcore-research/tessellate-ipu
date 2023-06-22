# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

from jax import lax
from jax._src.lax.lax import copy_p
from jax.core import Primitive, ShapedArray

from tessellate_ipu.utils import DTypeLike

from .tile_array import TileShardedArray
from .tile_interpreter import register_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    from_numpy_dtype_to_ipu_type,
    get_ipu_type_name,
    make_ipu_vertex_attributes,
    make_ipu_vertex_in_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
)

# popops definition.
# enum class UnaryOpType {
#   ABSOLUTE,
#   ASIN,
#   BITWISE_NOT,
#   CBRT,
#   CEIL,
#   COS,
#   COUNT_LEADING_ZEROS,
#   ERF,
#   EXPONENT,
#   EXPONENT_MINUS_ONE,
#   FLOOR,
#   GELU_ERF,
#   INVERSE,
#   IS_FINITE,
#   IS_INF,
#   IS_NAN,
#   LOGARITHM,
#   LOGARITHM_ONE_PLUS,
#   LOGICAL_NOT,
#   NEGATE,
#   POPCOUNT,
#   RELU,
#   SIGNUM,
#   SIN,
#   TAN,
#   TANH,
#   ROUND,
#   SQRT,
#   SQUARE,
#   SIGMOID,
#   RSQRT,
#   TRUNC
# };

_unary_primitive_to_vertex_basename: Dict[Primitive, str] = {
    lax.abs_p: "ABSOLUTE",
    lax.asin_p: "ASIN",
    lax.cbrt_p: "CBRT",
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
    lax.population_count_p: "POPCOUNT",
    lax.sin_p: "SIN",
    lax.sign_p: "SIGNUM",
    lax.tan_p: "TAN",
    lax.tanh_p: "TANH",
    lax.round_p: "ROUND",
    lax.rsqrt_p: "RSQRT",
    lax.sqrt_p: "SQRT",
    # lax.integer_pow_p: "",
}
"""Mapping from unary JAX primitives to IPU vertex basename.
"""


def make_unary1d_vertex_fullname(basename: str, dtype: DTypeLike) -> str:
    """Create the full vertex name from the basename and dtype."""
    ipu_dtype = get_ipu_type_name(dtype)
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
        inputs_info=[make_ipu_vertex_in_info("in", inavals[0])],
        outputs_info=[make_ipu_vertex_out_info("out", inavals[0])],
        attributes_i32=[],
        attributes_f32=[],
    )
    return ipu_prim_info


def make_cast1d_vertex_fullname(indtype: DTypeLike, outdtype: DTypeLike) -> str:
    """Create the full cast/convert_element_dtype vertex name."""
    ipu_indtype = from_numpy_dtype_to_ipu_type(indtype).name.lower()
    ipu_outdtype = from_numpy_dtype_to_ipu_type(outdtype).name.lower()
    return f"popops::Cast1D<{ipu_indtype},{ipu_outdtype}>"


def ipu_cast_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU cast/convert dtype primitive translation rule to IPU vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    assert attributes is not None
    inaval = inavals[0]
    outdtype = attributes["new_dtype"]
    outaval = ShapedArray(inaval.shape, outdtype)
    # IPU cast arguments.
    vname = make_cast1d_vertex_fullname(inaval.dtype, outaval.dtype)
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(numElems=inaval.size)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_in_info("src", inaval)],
        outputs_info=[make_ipu_vertex_out_info("dst", outaval)],
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


def ipu_integer_pow_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `integer_pow` primitive translation rule to IPU vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    assert attributes is not None
    inaval = inavals[0]
    pow = attributes["y"]
    if pow != 2:
        # TODO: general vertex?
        raise ValueError("Only supporting integer power of 2 on IPU tile primitives.")

    # IPU cast arguments.
    vname = make_unary1d_vertex_fullname("SQUARE", inaval.dtype)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_in_info("in", inaval)],
        outputs_info=[make_ipu_vertex_out_info("out", inaval)],
        attributes_i32=[],
        attributes_f32=[],
    )
    return ipu_prim_info


# Register all supported JAX unary ops.
for p in _unary_primitive_to_vertex_basename.keys():
    register_ipu_tile_primitive(p, ipu_unary_primitive_translation)

register_ipu_tile_primitive(lax.convert_element_type_p, ipu_cast_primitive_translation)
register_ipu_tile_primitive(lax.integer_pow_p, ipu_integer_pow_translation)


# On tile (mem)copy primitive.
def ipu_tile_memcpy(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `copy` primitive translation rule to IPU vertex.

    TODO: not using Poplar optimized ASM vertex yet. How to do
    reinterpret_cast to `char` in general case? Custom `TileMemcpy`
    in the meantime.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    inaval = inavals[0]

    gp_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "vertex", "tile_prim_vertex.cpp"))
    vname = make_ipu_vertex_name_templated("TileMemcpyVertex", inaval.dtype)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_in_info("in", inaval)],
        outputs_info=[make_ipu_vertex_out_info("out", inaval)],
        attributes_i32=[],
        attributes_f32=[],
        gp_filename=gp_filename,
        # Very approximate perf. estimate. TODO: refine, once optimized!
        perf_estimate=inaval.size * inaval.dtype.itemsize,
    )
    return ipu_prim_info


register_ipu_tile_primitive(copy_p, ipu_tile_memcpy)


def tile_copy(input: TileShardedArray) -> TileShardedArray:
    """On tile-copy of sharded array (using custom vertex for now).

    NOTE: `jnp.copy` calls can sometimes be optimized out by Poplar compiler,
    whereas `tile_copy` will ALWAYS be performing a copy, even unnecessary ones.

    Args:
        input: Tile sharded array.
    Returns:
        Copied array, with same tile-mapping.
    """
    r: TileShardedArray = tile_map_primitive(copy_p, input)  # type:ignore
    return r
