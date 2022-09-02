from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax import lax

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_sharded


class IpuTileMapPrimitiveTests(chex.TestCase, parameterized.TestCase):
    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__unary__no_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input = np.random.randn(*inshape).astype(dtype)
        input = tile_put_sharded(input, tiles)
        output = tile_map_primitive(lax.abs_p, [input])

        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input.dtype
        npt.assert_array_equal(output.array, np.abs(input.array))

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__unary__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(array):
            input = tile_put_sharded(array, tiles)
            output = tile_map_primitive(lax.abs_p, [input])
            return output

        output = compute_fn(input)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input.dtype
        npt.assert_array_equal(output.array, np.abs(input))

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__binary__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            output = tile_map_primitive(lax.add_p, [input0, input1])
            return output

        output = compute_fn(input0, input1)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input0.dtype
        npt.assert_array_almost_equal(output.array, input0 + input1, decimal=2)
