# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import json
import os
from typing import Dict, Sequence, Tuple, Union

import cppimport.import_hook  # noqa: F401
import jax.lax
import numpy as np
from jax import core
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import LoweringRuleContext, ir
from jax.interpreters.xla import ShapedArray
from jax.lib import xla_client
from jax_ipu_addons.primitives.custom_primitive_utils import ipu_xla_custom_primitive_call
from jax_ipu_addons.utils import xla_shape_to_aval

from jax_ipu_research.utils import DType

from . import tile_array_primitives_impl  # type:ignore

TilePutShardedPrimitive = tile_array_primitives_impl.TilePutShardedPrimitive
TilePutReplicatedPrimitive = tile_array_primitives_impl.TilePutReplicatedPrimitive
TileGatherPrimitive = tile_array_primitives_impl.TileGatherPrimitive
TileGatherParams = tile_array_primitives_impl.TileGatherParams
TileDataBarrierPrimitive = tile_array_primitives_impl.TileDataBarrierPrimitive
TileDataBarrierParams = tile_array_primitives_impl.TileDataBarrierParams

tile_put_sharded_prim_p = core.Primitive("tile_put_sharded")
tile_put_replicated_prim_p = core.Primitive("tile_put_replicated")
tile_gather_prim_p = core.Primitive("tile_gather")
tile_data_barrier_prim_p = core.Primitive("tile_data_barrier")

default_backends = ["cpu", "cuda", "tpu", "rocm"]


def make_tiles_raw_attributes(tiles: Tuple[int, ...]) -> str:
    """Make raw JSON attributes corresponding to a collection of tiles."""
    tiles_json = json.dumps(tuple(tiles))
    return tiles_json


def tile_put_sharded_prim(x, tiles):
    return tile_put_sharded_prim_p.bind(x, tiles=tiles)


def tile_put_sharded_prim_impl(x, tiles):
    # No-op when not jitted.
    assert x.shape[0] == len(tiles)
    return x


def tile_put_sharded_prim_abstract_eval(xs, tiles) -> ShapedArray:
    assert xs.shape[0] == len(tiles)
    return xs


def tile_put_sharded_prim_xla_translation_default(ctx, xc, tiles):
    """`tile_put_sharded_prim` default XLA translation, for CPU/GPU backends: no-op"""
    return xc


def tile_put_sharded_prim_xla_translation_ipu(ctx, xc, tiles):
    """`tile_put_sharded_prim` IPU backend XLA translation, as a custom primitive."""
    # TODO: Check the IPU tile out of the memory (roughly)
    inputs = [xc]
    # Passing the tiles collections as a raw attributes to the C++ implementation.
    raw_attributes = make_tiles_raw_attributes(tiles)
    outputs_aval = [xla_shape_to_aval(ctx.get_shape(xc))]
    # TODO: Add Github permanent link to C++.
    outputs = ipu_xla_custom_primitive_call(
        TilePutShardedPrimitive, ctx, inputs, outputs_aval, attributes=raw_attributes
    )
    return outputs[0]


# Register the primal implementation with JAX
tile_put_sharded_prim_p.def_impl(tile_put_sharded_prim_impl)
# Register the abstract evaluation with JAX
tile_put_sharded_prim_p.def_abstract_eval(tile_put_sharded_prim_abstract_eval)
# Register XLA translation, for different backends.
xla.backend_specific_translations["ipu"][tile_put_sharded_prim_p] = tile_put_sharded_prim_xla_translation_ipu
for b in default_backends:
    xla.backend_specific_translations[b][tile_put_sharded_prim_p] = tile_put_sharded_prim_xla_translation_default


def tile_put_replicated_prim(x, tiles):
    return tile_put_replicated_prim_p.bind(x, tiles=tiles)


def tile_put_replicated_prim_impl(x, tiles):
    return np.stack([x for _ in range(len(tiles))], axis=0)


def tile_put_replicated_prim_abstract_eval(xs, tiles) -> ShapedArray:
    outshape = (len(tiles), *xs.shape)
    return ShapedArray(outshape, xs.dtype, xs.weak_type)


def tile_put_replicated_prim_mlir_translation_default(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    """`tile_put_replicated_prim` default MLIR translation, for CPU/GPU backends: simple concat."""
    tiles = params["tiles"]

    # Not sure using a local function is a good idea?
    def tile_replicated_fn(input):
        N = len(tiles)
        input = jax.lax.expand_dims(input, dimensions=(0,))
        return jax.lax.concatenate([input] * N, dimension=0)

    # Lower to MLIR using JAX tooling. TODO: cache lowering?
    tile_replicated_lower_fn = mlir.lower_fun(tile_replicated_fn, multiple_results=False)
    return tile_replicated_lower_fn(ctx, *args)


def tile_put_replicated_prim_xla_translation_ipu(ctx, xc, tiles):
    """`tile_put_replicated_prim` IPU backend XLA translation, as a custom primitive."""
    inputs = [xc]
    # Passing the tiles collections as a raw attributes to the C++ implementation.
    raw_attributes = make_tiles_raw_attributes(tiles)
    outputs_aval = [tile_put_replicated_prim_abstract_eval(xla_shape_to_aval(ctx.get_shape(xc)), tiles)]
    outputs = ipu_xla_custom_primitive_call(
        TilePutReplicatedPrimitive, ctx, inputs, outputs_aval, attributes=raw_attributes
    )
    return outputs[0]


# Register the primal implementation with JAX
tile_put_replicated_prim_p.def_impl(tile_put_replicated_prim_impl)
# Register the abstract evaluation with JAX
tile_put_replicated_prim_p.def_abstract_eval(tile_put_replicated_prim_abstract_eval)
# Register XLA translation, for different backends.
xla.backend_specific_translations["ipu"][tile_put_replicated_prim_p] = tile_put_replicated_prim_xla_translation_ipu
# Register MLIR translation for other backends.
for b in default_backends:
    mlir.register_lowering(tile_put_replicated_prim_p, tile_put_replicated_prim_mlir_translation_default, b)


def tile_gather_prim(x, previous_tiles, indices, tiles):
    return tile_gather_prim_p.bind(x, previous_tiles=previous_tiles, indices=indices, tiles=tiles)


def tile_gather_prim_impl(x, previous_tiles, indices, tiles):
    # Numpy basic gather on axis=0
    return x[list(indices)]


def tile_gather_prim_abstract_eval(xs, previous_tiles, indices, tiles) -> ShapedArray:
    item_shape = xs.shape[1:]
    outshape = (len(tiles), *item_shape)
    return ShapedArray(outshape, xs.dtype, xs.weak_type)


def tile_gather_prim_xla_translation_default(ctx, xc, previous_tiles, indices, tiles):
    """`tile_gather_prim` default XLA translation, for CPU/GPU backends: simple JAX static gather"""
    # TODO: implementation from JAX?
    raise NotImplementedError()


def tile_gather_prim_xla_translation_ipu(ctx, xc, previous_tiles, indices, tiles):
    """`tile_gather_prim` IPU backend XLA translation, as a custom primitive."""
    inputs = [xc]
    # Til gather parameters, to pass to the XLA op.
    gather_params = TileGatherParams(previous_tiles, indices, tiles)
    raw_attributes = gather_params.to_json_str()
    outputs_aval = [
        tile_gather_prim_abstract_eval(xla_shape_to_aval(ctx.get_shape(xc)), previous_tiles, indices, tiles)
    ]
    outputs = ipu_xla_custom_primitive_call(TileGatherPrimitive, ctx, inputs, outputs_aval, attributes=raw_attributes)
    return outputs[0]


# Register the primal implementation with JAX
tile_gather_prim_p.def_impl(tile_gather_prim_impl)
# Register the abstract evaluation with JAX
tile_gather_prim_p.def_abstract_eval(tile_gather_prim_abstract_eval)
# Register XLA translation, for different backends.
xla.backend_specific_translations["ipu"][tile_gather_prim_p] = tile_gather_prim_xla_translation_ipu
for b in default_backends:
    xla.backend_specific_translations[b][tile_gather_prim_p] = tile_gather_prim_xla_translation_default


def tile_data_barrier_prim(inputs, inputs_tiles):
    return tile_data_barrier_prim_p.bind(*inputs, inputs_tiles=list(inputs_tiles))


def tile_data_barrier_prim_impl(*args, **params):
    return tuple(args)


def tile_data_barrier_prim_abstract_eval(*args: ShapedArray, **params) -> Tuple[ShapedArray, ...]:
    return args


def tile_data_barrier_prim_xla_translation_default(ctx, *args, **params):
    """`tile_data_barrier_prim` default XLA translation, for CPU/GPU backends."""
    # Translate into standard optimization barrier on CPU/GPU/TPU.
    return xla_client.ops.OptimizationBarrier(xla_client.ops.Tuple(ctx, args))


_tile_barrier_dtype_mapping: Dict[DType, DType] = {
    np.dtype(np.int8): np.dtype(np.uint8),
    np.dtype(np.uint8): np.dtype(np.uint8),
    np.dtype(np.int16): np.dtype(np.uint16),
    np.dtype(np.uint16): np.dtype(np.uint16),
    np.dtype(np.float16): np.dtype(np.uint16),
    np.dtype(np.int32): np.dtype(np.uint32),
    np.dtype(np.uint32): np.dtype(np.uint32),
    np.dtype(np.float32): np.dtype(np.uint32),
}


def tile_data_barrier_refdtype(dtype: DType) -> DType:
    """Find the reference dtype to use in IPU tile data barrier."""
    return _tile_barrier_dtype_mapping[dtype]


def tile_data_barrier_prim_xla_translation_ipu(ctx, *args, **params):
    """`tile_data_barrier_prim` IPU backend XLA translation, as a custom primitive."""
    from .tile_interpreter_primitives import make_ipu_vertex_name_templated

    inputs = list(args)
    inputs_aval = [xla_shape_to_aval(ctx.get_shape(xc)) for xc in inputs]
    dtypes = list({aval.dtype for aval in inputs_aval})
    dtypes_size = {dt.itemsize for dt in dtypes}
    if len(dtypes_size) > 1:
        raise TypeError(f"Only supporting dtypes of same size in Tile data barrier: {dtypes}.")

    inputs_tiles = params["inputs_tiles"]
    max_tile = max([max(s) for s in inputs_tiles])
    # Passing the tiles collections as a raw attributes to the C++ implementation.
    refdtype = tile_data_barrier_refdtype(dtypes[0])
    vname = make_ipu_vertex_name_templated("TileDataBarrierVertex", refdtype)
    barrier_params = TileDataBarrierParams(vname, inputs_tiles, max_tile)
    raw_attributes = barrier_params.to_json_str()

    gp_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "vertex", "tile_prim_vertex.cpp"))
    outputs_aval = inputs_aval
    outputs = ipu_xla_custom_primitive_call(
        TileDataBarrierPrimitive,
        ctx,
        inputs,
        outputs_aval,
        attributes=raw_attributes,
        ipu_gp_filename=gp_filename,
    )
    # Re-construct the XLA tuple (TODO: clean this back & forth mess!)
    return xla_client.ops.Tuple(ctx, outputs)


tile_data_barrier_prim_p.multiple_results = True
# Register the primal implementation with JAX
tile_data_barrier_prim_p.def_impl(tile_data_barrier_prim_impl)
# Register the abstract evaluation with JAX
tile_data_barrier_prim_p.def_abstract_eval(tile_data_barrier_prim_abstract_eval)
# Register XLA translation, for different backends.
xla.backend_specific_translations["ipu"][tile_data_barrier_prim_p] = tile_data_barrier_prim_xla_translation_ipu
for b in default_backends:
    xla.backend_specific_translations[b][tile_data_barrier_prim_p] = tile_data_barrier_prim_xla_translation_default
