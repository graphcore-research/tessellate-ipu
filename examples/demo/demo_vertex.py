from functools import partial
import jax
import os
import numpy as np
from jax_ipu_research.tile import (
    tile_put_replicated,
    tile_put_sharded,
    tile_map_primitive,
    create_ipu_tile_primitive,
    ipu_hw_cycle_count,
)

vertex_filename = os.path.join(os.path.dirname(__file__), "demo_vertex.cpp")
# Declaring a tile primitive in a very simple & fast way.
ipu_custom_vertex_p = create_ipu_tile_primitive(
    "ipu_custom_vertex",
    "CustomMultiOutVertex<{in}>",  # Support templated dtype from input.
    inputs=["in"],
    outputs={"out0": 0, "out1": 0},
    constants={"constant_scale": lambda ins, *_: np.array([ins[0].size], ins[0].dtype)},
    tmp_space=0,
    gp_filename=vertex_filename,
    perf_estimate=100,
)


N = 4
tiles = (0, 2, 5)
data0 = np.random.randn(len(tiles), N).astype(np.float32)


@partial(jax.jit, backend="ipu")
def compute_fn(input):
    input = tile_put_sharded(input, tiles)
    input, start = ipu_hw_cycle_count(input)
    out0, out1 = tile_map_primitive(ipu_custom_vertex_p, input, scale_value=1)
    out0, end = ipu_hw_cycle_count(out0)
    return start, end


out0, out1 = compute_fn(data0)
print("SHAPE:", out0.shape, out1.shape)
print("RESULT:", out0.array, out1.array)


# x, start = ipu_hw_cycle_count(x)
# r = tile_map_primitive(dot_product1d_p, x, y)
# # r = tile_map_primitive(reduce_sum_p, r, axes=(0,)) # Optional final reduction.
# r, end = ipu_hw_cycle_count(r)  # type:ignore

