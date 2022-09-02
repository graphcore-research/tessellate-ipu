# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple

import cppimport.import_hook  # noqa: F401
import numpy as np
from jax import core
from jax.interpreters import xla
from jax.interpreters.xla import ShapedArray
from jax_ipu_addons.primitives.custom_primitive_utils import ipu_xla_custom_primitive_call
from jax_ipu_addons.utils import xla_shape_to_aval

from . import tile_array_primitives_impl  # type:ignore

TilePutShardedPrimitive = tile_array_primitives_impl.TilePutShardedPrimitive
TilePutReplicatedPrimitive = tile_array_primitives_impl.TilePutReplicatedPrimitive

tile_put_sharded_prim_p = core.Primitive("tile_put_sharded")
tile_put_replicated_prim_p = core.Primitive("tile_put_replicated")


def make_tiles_raw_attributes(tiles: Tuple[int, ...]) -> str:
    """Make raw attributes corresponding to a collection of tiles."""
    raw_attributes = np.asarray(tiles, np.int32).tobytes().decode(errors="ignore")
    return raw_attributes


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
xla.backend_specific_translations["cpu"][tile_put_sharded_prim_p] = tile_put_sharded_prim_xla_translation_default
xla.backend_specific_translations["gpu"][tile_put_sharded_prim_p] = tile_put_sharded_prim_xla_translation_default


def tile_put_replicated_prim(x, tiles):
    return tile_put_replicated_prim_p.bind(x, tiles=tiles)


def tile_put_replicated_prim_impl(x, tiles):
    return np.stack([x for _ in range(len(tiles))], axis=0)


def tile_put_replicated_prim_abstract_eval(xs, tiles) -> ShapedArray:
    outshape = (len(tiles), *xs.shape)
    return ShapedArray(outshape, xs.dtype, xs.weak_type)


def tile_put_replicated_prim_xla_translation_default(ctx, xc, tiles):
    """`tile_put_replicated_prim` default XLA translation, for CPU/GPU backends: simple concat"""
    # TODO: implementation from JAX?
    raise NotImplementedError()


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
xla.backend_specific_translations["cpu"][tile_put_replicated_prim_p] = tile_put_replicated_prim_xla_translation_default
xla.backend_specific_translations["gpu"][tile_put_replicated_prim_p] = tile_put_replicated_prim_xla_translation_default
