# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from jax import lax
from jax._src.lax.lax import copy_p
from jax.core import Primitive, ShapedArray

from tessellate_ipu.core import (
    IpuTileMapEquation,
    TileShardedArray,
    from_numpy_dtype_to_ipu_type,
    get_ipu_tile_primitive_translation,
    get_ipu_type_name,
    make_ipu_vertex_attributes,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_in_info,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
    primitive_clone,
    primitive_num_inout_alias_args,
    register_ipu_tile_primitive,
    tile_map,
)
from tessellate_ipu.utils import DTypeLike

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


def make_unary1d_vertex_fullname(basename: str, dtype: DTypeLike, inplace: bool) -> str:
    """Create the full vertex name from the basename, dtype and inplace."""
    ipu_dtype = get_ipu_type_name(dtype)
    unary_basename = "UnaryOp1DInPlace" if inplace else "UnaryOp1D"
    return f"popops::{unary_basename}<popops::expr::UnaryOpType::{basename},{ipu_dtype}>"


def make_unary1d_vertex_io_infos(inaval: ShapedArray, inplace: bool) -> Tuple[List[Any], List[Any]]:
    """Build Poplibs unary1d vertex IO infos.

    Naming of the vertex interface depends on whether it is inplace or not.
    """
    if inplace:
        return [make_ipu_vertex_inout_info("inOut", inaval)], [make_ipu_vertex_inout_info("inOut", inaval)]
    else:
        return [make_ipu_vertex_in_info("in", inaval)], [make_ipu_vertex_out_info("out", inaval)]


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
    # Is it an inplace primitive?
    inplace_prim = primitive_num_inout_alias_args(p) > 0
    vname = make_unary1d_vertex_fullname(vertex_basename, inavals[0].dtype, inplace_prim)
    inputs_info, outputs_info = make_unary1d_vertex_io_infos(inavals[0], inplace_prim)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
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

    Only supporting -1 and 2 exponents at the moment.

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
    supported_powers = {-1: "INVERSE", 2: "SQUARE"}
    if pow not in supported_powers:
        # TODO: general vertex?
        raise ValueError(f"Only supporting integer powers '{tuple(supported_powers.keys())}' in TessellateIPU library.")

    # Used proper vertex depending on the power!
    vname = make_unary1d_vertex_fullname(supported_powers[pow], inaval.dtype, inplace=False)
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


def ipu_iota_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `iota` primitive translation rule to IPU vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 0
    assert attributes is not None
    print(attributes)
    dtype = attributes["dtype"]
    dimension = int(attributes["dimension"])
    shape = attributes["shape"]

    assert dimension == 0
    assert len(shape) == 1
    # Iota vertex in/outs
    vname = make_ipu_vertex_name_templated("popops::Iota", dtype)
    outaval = p.abstract_eval(dtype=dtype, dimension=dimension, shape=shape)[0]
    inputs_info = [make_ipu_vertex_constant_info("offsets", np.array([0], dtype=dtype))]
    outputs_info = [make_ipu_vertex_out_info("out", outaval, vertex_dim2=shape[0])]
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
        attributes_i32=[],
        attributes_f32=[],
    )
    return ipu_prim_info


register_ipu_tile_primitive(lax.iota_p, ipu_iota_translation)


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

    gp_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "../core", "vertex", "tile_prim_vertex.cpp"))
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
    r: TileShardedArray = tile_map(copy_p, input)  # type:ignore
    return r


def register_ipu_unary_inplace_tile_primitive(orig_prim):
    """Create and register IPU unary inplace tile primitive.

    Args:
        orig_prim: Original non-inplace unary primitive.
    Returns:
        Inplace unary primitive, registered.
    """
    inplace_prim = primitive_clone(orig_prim, f"{orig_prim.name}_inplace")
    _, tl_translation = get_ipu_tile_primitive_translation(orig_prim.name)
    register_ipu_tile_primitive(inplace_prim, tl_translation)
    _unary_primitive_to_vertex_basename[inplace_prim] = _unary_primitive_to_vertex_basename[orig_prim]
    # TODO: depreciate this field!
    inplace_prim.num_inout_alias_args = 1
    return inplace_prim


# Inplace variants of support JAX LAX unary ops.
abs_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.abs_p)
asin_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.asin_p)
cbrt_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.cbrt_p)
ceil_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.ceil_p)
erf_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.erf_p)
exp_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.exp_p)
expm1_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.expm1_p)
floor_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.floor_p)
is_finite_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.is_finite_p)
log_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.log_p)
log1p_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.log1p_p)
neg_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.neg_p)
population_count_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.population_count_p)
sin_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.sin_p)
sign_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.sign_p)
tan_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.tan_p)
# tanh_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.tanh_p) # Weird inaccuray issue. TODO: investigate the problem?
round_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.round_p)
rsqrt_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.rsqrt_p)
sqrt_inplace_p = register_ipu_unary_inplace_tile_primitive(lax.sqrt_p)
