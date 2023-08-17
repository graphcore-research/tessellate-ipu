# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax
import numpy as np

from tessellate_ipu import tile_map, tile_put_sharded
from tessellate_ipu.lax import sqrt_inplace_p

data = np.array([1, 2, 3], np.float32)
tiles = (1, 2, 5)


def inner(_, carry):
    # Inplace operation optimal for loops on IPU.
    # Keep re-using the same buffer, no additional copy required.
    return tile_map(sqrt_inplace_p, carry)


@partial(jax.jit, backend="ipu")
def compute_fn(input):
    x = tile_put_sharded(input, tiles)
    x = jax.lax.fori_loop(0, 4, inner, x)
    return x


output = compute_fn(data)
print(output)
print(output, output.array.device())
print("SHAPE:", output.shape)
print("RESULT:", output.array)
