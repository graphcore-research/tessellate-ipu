from functools import partial

import jax
import numpy as np

from jax_ipu_research.tile import tile_map_primitive, tile_put_sharded

data = np.array([1, -2, 3], np.float32)
tiles = (0, 2, 5)


@partial(jax.jit, backend="ipu")
def compute_fn(input):
    input = tile_put_sharded(input, tiles)
    # input = tile_put_replicated(input, tiles)
    return tile_map_primitive(jax.lax.neg_p, input)


output = compute_fn(data)
print(output, output.array.device())
print("SHAPE:", output.shape)
print("RESULT:", output.array)
