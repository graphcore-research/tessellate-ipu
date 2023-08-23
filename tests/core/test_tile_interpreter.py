# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized
from custom_arange_primitive import custom_arange_p, custom_inplace_op_p, custom_multi_out_p, custom_single_out_p
from jax import lax

from tessellate_ipu.core import TileShardedArray, ipu_cycle_count, tile_map, tile_put_replicated, tile_put_sharded


class IpuTileMapPrimitiveTests(chex.TestCase, parameterized.TestCase):
    def test__tile_map__no_primitive__noop(self):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 2)
        input = tile_put_sharded(np.random.randn(*inshape), tiles)
        outputs = tile_map(None, input)
        assert outputs[0] is input

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map__unary__no_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        data = np.random.randn(*inshape).astype(dtype)
        input = tile_put_sharded(data, tiles)
        output = tile_map(lax.abs_p, input)

        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input.dtype
        npt.assert_array_equal(output.array, np.abs(input.array))

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map__unary__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(array):
            input = tile_put_sharded(array, tiles)
            output = tile_map(lax.abs_p, input)
            return output

        output = compute_fn(input)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input.dtype
        npt.assert_array_equal(output.array, np.abs(input))

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map__binary__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            output = tile_map(lax.add_p, input0, input1)
            return output

        output = compute_fn(input0, input1)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input0.dtype
        npt.assert_array_almost_equal(output.array, input0 + input1, decimal=2)

    # @parameterized.parameters([np.float32])
    # def test__tile_map__custom_vertex__no_jitting(self, dtype):
    #     size = 128
    #     tiles = (3, 4, 5)

    #     def compute_fn():
    #         output = tile_map(custom_arange_p, [], attributes={"size": size}, tiles=tiles)
    #         return output

    #     output = compute_fn()
    #     assert isinstance(output, TileShardedArray)
    #     assert output.tiles == tiles
    #     assert output.dtype == dtype

    @parameterized.parameters([np.float32, np.int32])
    def test__tile_map__custom_arange_vertex__single_output__ipu_jitting(self, dtype):
        size = 128
        tiles = (3, 4, 5)
        # 2d scaling tensor, per tile.
        scales = np.random.rand(len(tiles), 2, size).astype(dtype) + 1
        global_scale = 7

        @partial(jax.jit, backend="ipu")
        def compute_fn(scales):
            scales = tile_put_sharded(scales, tiles)
            output = tile_map(custom_arange_p, scales, size=size, dtype=dtype, tiles=tiles)
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
    def test__tile_map__custom_vertex__single_output__ipu_jitting(self, dtype, backend):
        size = 128
        tiles = (3, 4, 5)
        input = np.random.rand(len(tiles), size).astype(dtype) + 1

        @partial(jax.jit, backend=backend)
        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map(custom_single_out_p, input)

        output = compute_fn(input)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == dtype
        npt.assert_array_equal(output, -input)

    @parameterized.parameters([(np.float32, "ipu"), (np.int32, "ipu"), (np.float32, "cpu")])
    def test__tile_map__custom_vertex__multi_outputs__ipu_jitting(self, dtype, backend):
        size = 128
        tiles = (3, 4, 5)
        input = np.random.rand(len(tiles), size).astype(dtype) + 1
        scale_value = 3

        @partial(jax.jit, backend=backend)
        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            out0, out1 = tile_map(custom_multi_out_p, input, scale_value=scale_value)  # type:ignore
            return out0, out1

        out0, out1 = compute_fn(input)
        assert isinstance(out0, TileShardedArray)
        assert out0.tiles == tiles
        assert out0.dtype == dtype
        assert out1.tiles == tiles
        assert out1.dtype == dtype

        npt.assert_array_equal(out0, size * scale_value * input)
        npt.assert_array_equal(out1, -size * scale_value * input)


@pytest.mark.ipu_hardware
class IpuTileMapPrimitiveHwTests(chex.TestCase, parameterized.TestCase):
    def test__tile_map__custom_inplace_vertex__fori_loop__zero_cycle_count_overhead(self):
        N = 1024
        Niter = 32
        # NOTE: skip tile zero where poplar inject loop control vertices.
        tiles = (2,)
        inshape = (len(tiles), N)
        indata = np.random.rand(*inshape).astype(np.float32)

        def inner(_, x):
            return tile_map(custom_inplace_op_p, x)

        def compute_fn(input, num_iters):
            # Hacky way of forcing to wait until all inputs are transferred from HOST.
            # Otherwise: work on tile 2 start before tile 0 (still transferring).
            input = input * num_iters
            input = tile_put_sharded(input, tiles)
            # Make sure number of iterations is on tile0 to avoid additional sync + comms.
            num_iters = tile_put_replicated(num_iters, (0,)).array[0]
            # Benchmark single call.
            input, start = ipu_cycle_count(input)
            x = tile_map(custom_inplace_op_p, input)
            x, mid = ipu_cycle_count(x)  # type:ignore
            # Benchmark fori_loop.
            y = jax.lax.fori_loop(0, num_iters, inner, x)
            y, end = ipu_cycle_count(y)
            return x, y, (start, mid, end)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        xcpu, ycpu, _ = compute_fn_cpu(indata, Niter)
        xipu, yipu, (start, mid, end) = compute_fn_ipu(indata, Niter)

        # Check IPU vs CPU result (especially `fori_loop`).
        npt.assert_array_almost_equal(xcpu, xipu, decimal=3)
        npt.assert_array_almost_equal(ycpu, yipu, decimal=3)

        (start, mid, end) = map(lambda v: np.asarray(v)[0], (start, mid, end))
        single_call_cycles = mid[0] - start[0]
        for_loop_cycles = end[0] - mid[0]
        # Less than 5% overhead `fori_loop` vs unrolling.
        # In particular: using inplace should not introduce additional copies.
        threshold = 0.95
        assert single_call_cycles * Niter >= for_loop_cycles * threshold
