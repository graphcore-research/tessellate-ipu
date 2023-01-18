from functools import partial
import jax
import numpy as np
from jax_ipu_research.tile import tile_put_replicated, tile_put_sharded, tile_map_primitive

N = 5
tiles = (0, 2, 5)
data0 = np.random.randn(len(tiles), N).astype(np.float32)
data1 = np.random.randn(N).astype(np.float32)


@partial(jax.jit, backend="ipu")
def compute_fn(input0, input1):
    input0 = tile_put_sharded(input0, tiles)

    input1 = tile_put_replicated(input1, (0, 1, 3))
    input1 = tile_map_primitive(jax.lax.neg_p, input1)
    input1 = tile_put_sharded(input1.array, tiles)
    
    return tile_map_primitive(jax.lax.add_p, input0, input1)


output = compute_fn(data0, data1)
print(output, output.array.device())
print("SHAPE:", output.shape)
print("RESULT:", output.array)
