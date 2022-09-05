from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from custom_arange_primitive import custom_arange_p
from jax import lax

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_sharded


class IpuTileMapPrimitiveTests(chex.TestCase, parameterized.TestCase):
    def test__tile_map_primitive__no_primitive__noop(self):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 2)
        input = tile_put_sharded(np.random.randn(*inshape), tiles)
        outputs = tile_map_primitive(None, [input])
        assert outputs[0] is input

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

    # @parameterized.parameters([np.float32])
    # def test__tile_map_primitive__custom_vertex__no_jitting(self, dtype):
    #     size = 128
    #     tiles = (3, 4, 5)

    #     def compute_fn():
    #         output = tile_map_primitive(custom_arange_p, [], attributes={"size": size}, tiles=tiles)
    #         return output

    #     output = compute_fn()
    #     assert isinstance(output, TileShardedArray)
    #     assert output.tiles == tiles
    #     assert output.dtype == dtype

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__custom_vertex__ipu_jitting(self, dtype):
        size = 128
        tiles = (3, 4, 5)

        @partial(jax.jit, backend="ipu")
        def compute_fn():
            output = tile_map_primitive(custom_arange_p, [], attributes={"size": size, "dtype": dtype}, tiles=tiles)
            return output

        output = compute_fn()
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == dtype
        for idx in range(len(tiles)):
            npt.assert_array_almost_equal(output.array[idx], np.arange(size).astype(dtype))
