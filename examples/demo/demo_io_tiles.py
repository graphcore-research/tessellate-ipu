# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax
import numpy as np

from tessellate_ipu import tile_data_barrier, tile_map, tile_put_sharded

# Number of tiles per "group"
Ntiles = 32
# Size of the array, per tile.
N = 512

# Splitting between IO tiles and compute tiles.
io_tiles = tuple(range(0, Ntiles))
compute_tiles = tuple(range(Ntiles, 2 * Ntiles))


@partial(jax.jit, backend="ipu", donate_argnums=(1,))
def compute_fn(input, iobuffer):
    """`compute_fn` acts as one stage pipeline:
    - `input` gets transfered to IO tiles, and then copied into `iobuffer` at the end of the program;
    - `iobuffer`, with previous call data, is already available on compute tiles for crunching numbers!
    """
    assert input.shape == iobuffer.shape

    # IO tiles => transfer from host (not blocking IO tiles.)
    input_sharded = tile_put_sharded(input, io_tiles)

    # Compute tiles => use iobuffer already available..
    iobuffer_sharded = tile_put_sharded(iobuffer, compute_tiles)
    # Some arbitrary computation...
    output = iobuffer_sharded[:, :64]
    output = tile_map(jax.lax.sin_p, output)  # type:ignore
    output = tile_map(jax.lax.reduce_sum_p, output, axes=(0,))  # type:ignore

    # Transfer to input compute tiles, to be copied in `iobuffer`.
    input_sharded = tile_put_sharded(input_sharded.array, compute_tiles)
    # Blocking to make sure compute is finished.
    input_sharded, output = tile_data_barrier(input_sharded, output)

    iobuffer = input_sharded.array
    return output, iobuffer


# Initialize IO buffer with zeros => first call output == 0
iobuffer_init = np.zeros((Ntiles, N), dtype=np.float32)
iobuffer_init = np.random.rand(Ntiles, N).astype(np.float32)

data0 = np.random.rand(Ntiles, N).astype(np.float32)
data1 = np.random.rand(Ntiles, N).astype(np.float32)

# First run, using "zeros"
iobuffer_init = jax.device_put(iobuffer_init, jax.devices("ipu")[0])  # [Note 1]
output0, iobuffer0 = compute_fn(data0, iobuffer_init)
# Second run, using `data0` previously transfered.
output1, iobuffer1 = compute_fn(data1, iobuffer0)

print("Output 0:", np.asarray(output0))
print("Output 1:", np.asarray(output1))

print("Expected output 0:", np.sum(np.sin(iobuffer_init[:, :64]), axis=-1))
print("Expected output 1:", np.sum(np.sin(data0[:, :64]), axis=-1))

print("NOTE: is iobuffer0 deleted (i.e. overwritten)?", iobuffer0.is_deleted())

# Note 1: JAX bug .... Error msg: "blablah"
