from functools import partial

import jax
import numpy as np

from jax_ipu_research.tile import tile_data_barrier, tile_map_primitive, tile_put_replicated

num_tiles = 64
gp0_tiles = tuple(range(num_tiles))
gp1_tiles = tuple(range(num_tiles, 2 * num_tiles))

sqrt_size = 4
size = sqrt_size * sqrt_size
data = np.random.rand(size).astype(np.float32)


@partial(jax.jit, backend="ipu")
def compute_unsync_fn(data):
    gp0_data = tile_put_replicated(data, gp0_tiles)
    gp1_data = tile_put_replicated(data, gp1_tiles)

    # First loop: GP0 and GP1 sync + comms.
    for idx in range(sqrt_size):
        # Inner loop: only GP0 is doing comms.
        for _ in range(sqrt_size):
            # Gp1 compute: no sync or comms.
            gp1_data = tile_map_primitive(jax.lax.add_p, gp1_data, gp1_data)  # type:ignore
            gp1_data = tile_map_primitive(jax.lax.mul_p, gp1_data, gp1_data)  # type:ignore

            # Gp0 compute + comms.
            gp0_data = tile_map_primitive(jax.lax.add_p, gp0_data, gp0_data)  # type:ignore
            gp0_data = tile_put_replicated(gp0_data.array[idx], gp0_tiles)

        # sync + comms in GP1.
        gp1_data = tile_put_replicated(gp1_data.array[idx], gp1_tiles)

        # Tile barrier: to avoid Poplar or XLA re-org.
        gp0_data, gp1_data = tile_data_barrier(gp0_data, gp1_data)

    # Mixup between GP0 and GP1
    gp1_data = tile_put_replicated(gp1_data.array[0], gp0_tiles)
    gp0_data = tile_map_primitive(jax.lax.add_p, gp0_data, gp1_data)  # type:ignore
    return gp0_data, gp1_data


rgp0, rgp1 = compute_unsync_fn(data)
