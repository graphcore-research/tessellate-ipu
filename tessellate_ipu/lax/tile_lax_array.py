# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Sequence, Tuple, Union

import jax.lax
import numpy as np
from jax.core import Primitive, ShapedArray
from jax.interpreters import mlir
from jax.interpreters.mlir import LoweringRuleContext, ir
from jax.lax import bitcast_convert_type_p, reshape_p, scatter_p

from tessellate_ipu.core import (
    IpuTileMapEquation,
    TileShardedArray,
    make_ipu_vertex_attributes,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
    register_ipu_tile_primitive,
    tile_constant_replicated,
    tile_constant_sharded,
    tile_map,
)
from tessellate_ipu.utils import DTypeLike


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


fill_p = Primitive("fill")
"""Fill primitive: create an array, and fill it with a constant.
Note: compared to `jax.lax.full`, it guarantees allocation of the full array instead of broadcasting.
"""


def fill(shape: Tuple[int, ...], fill_value: Any, dtype: DTypeLike):
    """Fill a tensor with given shape and value."""
    return fill_p.bind(shape=shape, fill_value=fill_value, dtype=dtype)


def fill_numpy_impl(shape: Tuple[int, ...], fill_value: Any, dtype: DTypeLike):
    return np.full(shape, fill_value, dtype=dtype)


def fill_abstract_eval(shape: Tuple[int, ...], fill_value: Any, dtype: DTypeLike):
    aval = jax.lax.full(shape, fill_value=fill_value, dtype=dtype)
    return ShapedArray(aval.shape, dtype=aval.dtype)


def ipu_fill_primitive_translation_ipu(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU tile translation for `fill`

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: Op attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 0
    assert attributes is not None
    shape = attributes["shape"]
    fill_value = attributes["fill_value"]
    dtype = attributes["dtype"]

    outaval = fill_abstract_eval(shape, fill_value, dtype)
    # Translation rule to IPU vertex
    vname = make_ipu_vertex_name_templated("popops::Fill", outaval.dtype)
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(**{"in": fill_value})
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[],
        outputs_info=[make_ipu_vertex_out_info("out", outaval)],
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


def fill_mlir_translation_default(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    """`fill` default MLIR translation, for CPU/GPU/IPU/... backends."""
    outaval = ctx.avals_out[0]
    fill_value = params["fill_value"]

    def fill_fn(*inputs):
        return jax.lax.full(outaval.shape, fill_value, outaval.dtype)

    # Lower to MLIR using JAX tooling. TODO: cache lowering?
    fill_lower_fn = mlir.lower_fun(fill_fn, multiple_results=False)
    return fill_lower_fn(ctx, *args)


fill_p.map_primitive = False
# Register the primal implementation with JAX.
fill_p.def_impl(fill_numpy_impl)
# Register the abstract evaluation with JAX.
fill_p.def_abstract_eval(fill_abstract_eval)
# Default MLIR translation for all backends.
mlir.register_lowering(fill_p, fill_mlir_translation_default)
# Register TessellateIPU translation.
register_ipu_tile_primitive(fill_p, ipu_fill_primitive_translation_ipu)


def tile_fill(shape: Tuple[int, ...], fill_value: Any, dtype: DTypeLike, tiles: Tuple[int, ...]) -> TileShardedArray:
    """Tile `fill` a tensor with given shape and value."""
    return tile_map(fill_p, shape=shape, fill_value=fill_value, dtype=dtype, tiles=tiles)  # type:ignore


def tile_sharded_identity(dtype: DTypeLike, tiles: Tuple[int, ...]) -> TileShardedArray:
    """Create a tile sharded identity matrix, i.e. sharded on tiles across the first axis.

    Args:
        dtype: Dtype of the identity matrix.
        tiles: Sharding tiles.
    Returns:
        Sharded identity matrix (N, N), with N = len(tiles)
    """
    with jax.named_scope("tile_sharded_identity"):
        N = len(tiles)
        # Build zero matrix + update diagonal entries.
        arr = tile_fill((N,), 0, dtype=dtype, tiles=tiles)
        # Requiring constants for indices + updates. Something more efficient?s
        indices = tile_constant_sharded(np.arange(0, N, dtype=np.uint32).reshape(N, 1, 1), tiles=tiles)
        updates = tile_constant_replicated(np.array([1], dtype=dtype), tiles=tiles)
        # Not the simplest way ever of updating diagonal terms!
        scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
        )
        arr = tile_map(
            scatter_p,
            arr,
            indices,
            updates,
            dimension_numbers=scatter_dnums,
            indices_are_sorted=False,
            unique_indices=False,
            mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            update_jaxpr=None,
            update_consts=None,
        )  # type:ignore
        return arr
