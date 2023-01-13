# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import numpy as np

from jax_ipu_research.utils import Array

from .tile_array import tile_put_replicated, tile_put_sharded


def ipu_argsort_quadratic_unique(x: Array) -> Array:
    """Argsort of a 1d array, in ascending array.

    This algorithm is a very basic inefficient quadratic algorithm! But has the advantage
    of being simple for small arrays.

    Args:
        x: 1d array to sort.
    Returns:
        (Ascending) argsorting indices.
    """
    # Should fit on tiles!
    assert x.ndim == 1
    assert x.size <= 1472
    N = x.size
    tiles = tuple(range(N))

    x_replicated = tile_put_replicated(x, tiles=tiles)
    x_sharded = tile_put_sharded(x, tiles=tiles)
    # Indices of every value in the array.
    sort_indices = jnp.sum(x_replicated.array < x_sharded.array.reshape((N, 1)), axis=1).astype(np.int32)
    # Re-shard over tiles.
    arange_sharded = tile_put_sharded(np.arange(N, dtype=np.int32), tiles)
    arange_replicated = tile_put_replicated(np.arange(N, dtype=np.int32), tiles)
    sort_indices_replicated = tile_put_replicated(sort_indices, tiles)
    # Convert to argsort indices.
    indices = (sort_indices_replicated.array == arange_sharded.array.reshape((N, 1))) * arange_replicated.array
    indices = jnp.sum(indices, axis=1)
    return indices
