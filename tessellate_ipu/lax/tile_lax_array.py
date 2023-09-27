# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

from jax.core import Primitive, ShapedArray
from jax.lax import bitcast_convert_type_p, reshape_p

from tessellate_ipu.core import IpuTileMapEquation, make_ipu_vertex_inout_info, register_ipu_tile_primitive


def ipu_reshape_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `reshape` LAX primitive translation rule to IPU vertex.

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
    new_sizes = attributes["new_sizes"]
    dimensions = attributes.get("dimensions", None)
    if dimensions is not None:
        raise NotImplementedError("TessellateIPU `reshape` does not support a custom `dimensions` argument.")

    outaval = ShapedArray(new_sizes, dtype=inaval.dtype, weak_type=inaval.dtype)
    # Empty vertex name trick => identity function with inout argument, just doing reshaping.
    vname = ""
    inputs_info = [
        make_ipu_vertex_inout_info("x", inaval),
    ]
    outputs_info = [make_ipu_vertex_inout_info("x", outaval)]
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


# Register JAX LAX reshape primitive.
register_ipu_tile_primitive(reshape_p, ipu_reshape_primitive_translation)


def ipu_bitcast_convert_type_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `bitcast_convert_type` LAX primitive translation rule to IPU vertex.

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
    new_dtype = attributes["new_dtype"]
    outaval = ShapedArray(inaval.shape, dtype=new_dtype, weak_type=inaval.dtype)
    # Empty vertex name trick => identity function with inout argument, just doing reshaping.
    vname = ""
    inputs_info = [
        make_ipu_vertex_inout_info("x", inaval),
    ]
    outputs_info = [make_ipu_vertex_inout_info("x", outaval)]
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


# Register JAX LAX bitcast_convert_type_p primitive.
register_ipu_tile_primitive(bitcast_convert_type_p, ipu_bitcast_convert_type_primitive_translation)
