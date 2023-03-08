import os

import jax
import jax.numpy as jnp
import numpy as np

from jax_ipu_research.tile import declare_ipu_tile_primitive, tile_map_primitive, tile_put_sharded

# fmt: off
demo_vertex_filename = os.path.join(os.path.dirname(__file__), "demo_vertex.cpp")


# Declaring a tile primitive in a very simple & fast way.
@declare_ipu_tile_primitive("DemoVertex<{x}>", gp_filename=demo_vertex_filename)
def demo_vertex_p(x: jax.ShapedArray):
    """
    This is called with ShapedArrays for each input,
    and should return dicts of "name":ShapedArray for each
      output
      constant
      temporary
    as well as a
      perf_estimate (see https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/classpoplar_1_1Graph.html#affc4b9033058a3b936a475621a27b919)

    This extends a typical JAX abstract evaluator, which would return the output
    ShapeArrays, to also describe the constants and temporary space it uses.

    The "name"s should correspond to the names in the Vertex class in the
    corresponding C++ file.  In this case the C++ contains

    ```
      template <typename T> class DemoVertex: public Vertex {
      public:
        Input<Vector<T>> x;
        Input<Vector<T>> constant_scale;

        Output<Vector<T>> out0;
        Output<Vector<T>> out1;
        ...
      };
    ```
    """
    r, c = x.shape

    outputs = {
        "out0": jax.ShapedArray((r, c // 2), x.dtype),
        "out1": jax.ShapedArray((r, c // 2), x.dtype)
    }
    constants = {
        "constant_scale": 0.5 * np.ones((r, c), dtype=x.dtype)
    }
    temps = {
        "tmp": jax.ShapedArray((r, c), x.dtype)
    }
    perf_estimate = r * c * 12
    return outputs, constants, temps, perf_estimate
# fmt: on


N = 4
M = 3
tiles = (0, 2, 5)
tiles_t = (7, 8, 9, 10)
assert len(tiles) == M
assert len(tiles_t) == N


def compute_fn_cpu(input):
    M, N, N1 = input.shape
    assert N == N1
    assert M == len(tiles)
    input_sharded = tile_put_sharded(input, tiles)

    out0, out1 = tile_map_primitive(demo_vertex_p, input_sharded, scale_value=1.23)  # type:ignore
    print("First out0 shape", out0.shape)
    assert out0.shape == (M, N, N // 2)

    # Reshuffle data and call demo_vertex_p again
    transpose = jnp.reshape(out0.array, (N, N // 2, M))
    print("Transposed shape", transpose.shape)
    assert transpose.shape[0] == len(tiles_t)
    transpose_sharded = tile_put_sharded(transpose, tiles_t)

    out0, out1 = tile_map_primitive(demo_vertex_p, transpose_sharded, scale_value=1.23)  # type:ignore

    return out0, out1


compute_fn = jax.jit(compute_fn_cpu, backend="ipu")

np.set_printoptions(formatter={"float": "{: 0.3f}".format})


data0 = np.arange(M * N * N, dtype=np.float32).reshape(M, N, N)
print(data0)

# out0, out1 = compute_fn_cpu(data0)

print("Input shape:", data0.shape)
out0, out1 = compute_fn(data0)
print("Output shape:", out0.shape, out1.shape)
print("RESULT:", out0.array, out1.array, sep="\n")
