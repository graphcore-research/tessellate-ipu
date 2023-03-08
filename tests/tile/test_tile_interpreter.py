from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from custom_arange_primitive import custom_arange_p, custom_multi_out_p
from jax import lax

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_sharded


class IpuTileMapPrimitiveTests(chex.TestCase, parameterized.TestCase):
    def test__tile_map_primitive__no_primitive__noop(self):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 2)
        input = tile_put_sharded(np.random.randn(*inshape), tiles)
        outputs = tile_map_primitive(None, input)
        assert outputs[0] is input

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__unary__no_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input = np.random.randn(*inshape).astype(dtype)
        input = tile_put_sharded(input, tiles)
        output = tile_map_primitive(lax.abs_p, input)

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
            output = tile_map_primitive(lax.abs_p, input)
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
            output = tile_map_primitive(lax.add_p, input0, input1)
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

    @parameterized.parameters([np.float32, np.int32])
    def test__tile_map_primitive__custom_vertex__single_output__ipu_jitting(self, dtype):
        size = 128
        tiles = (3, 4, 5)
        # 2d scaling tensor, per tile.
        scales = np.random.rand(len(tiles), 2, size).astype(dtype) + 1
        global_scale = 7

        @partial(jax.jit, backend="ipu")
        def compute_fn(scales):
            scales = tile_put_sharded(scales, tiles)
            output = tile_map_primitive(custom_arange_p, scales, size=size, dtype=dtype, tiles=tiles)
            return output

        output = compute_fn(scales)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == dtype
        for idx in range(len(tiles)):
            # Proper scaled arange on every tile.
            npt.assert_array_almost_equal(
                output.array[idx],
                np.arange(size).astype(dtype) * scales[idx, 0] * scales[idx, 1] * global_scale,
                decimal=0,
            )

    @parameterized.parameters([(np.float32, "ipu"), (np.int32, "ipu"), (np.float32, "cpu")])
    def test__tile_map_primitive__custom_vertex__multi_outputs__ipu_jitting(self, dtype, backend):
        size = 128
        tiles = (3, 4, 5)
        input = np.random.rand(len(tiles), size).astype(dtype) + 1
        scale_value = 3

        @partial(jax.jit, backend=backend)
        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            out0, out1 = tile_map_primitive(custom_multi_out_p, input, scale_value=scale_value)  # type:ignore
            return out0, out1

        out0, out1 = compute_fn(input)
        assert isinstance(out0, TileShardedArray)
        assert out0.tiles == tiles
        assert out0.dtype == dtype
        assert out1.tiles == tiles
        assert out1.dtype == dtype

        npt.assert_array_equal(out0, size * scale_value * input)
        npt.assert_array_equal(out1, -size * scale_value * input)
