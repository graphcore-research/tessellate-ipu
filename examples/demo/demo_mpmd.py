# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax
import numpy as np

from tessellate_ipu import tile_map, tile_put_replicated, tile_put_sharded

N = 5
tiles0 = (0, 2, 5)
tiles1 = (1, 3, 7)

data0 = np.random.randn(len(tiles0), 5).astype(np.float32)
data1 = np.random.randn(5).astype(np.float32)


@partial(jax.jit, backend="ipu")
def compute_fn(input0, input1):
    # TODO: group in a single call with JAX pytree
    input00 = tile_put_sharded(input0, tiles0)
    input10 = tile_put_replicated(input1, tiles0)

    input01 = tile_put_sharded(input0, tiles1)
    input11 = tile_put_replicated(input1, tiles1)

    # No need for explicit parallel call: Poplar compiler doing a great job here!
    output0 = tile_map(jax.lax.add_p, input00, input10)
    output1 = tile_map(jax.lax.mul_p, input01, input11)
    return output0, output1


output0, output1 = compute_fn(data0, data1)
print(np.array(output0.tiles))
print(np.array(output1.tiles))
