# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized
from jax import lax
from jax.core import ShapedArray

import tessellate_ipu.lax
from tessellate_ipu import (
    TileShardedArray,
    ipu_cycle_count,
    tile_data_barrier,
    tile_map,
    tile_put_replicated,
    tile_put_sharded,
)
from tessellate_ipu.core.tile_interpreter_primitives import IpuTileMapEquation, IpuType
from tessellate_ipu.lax.tile_lax_binary import scaled_add_p, scaled_sub_p
from tessellate_ipu.lax.tile_lax_unary import ipu_unary_primitive_translation, make_unary1d_vertex_fullname, tile_copy


class IpuTileUnaryPrimitiveTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    def test__make_unary1d_vertex_fullname__proper_result(self):
        assert (
            make_unary1d_vertex_fullname("COS", np.float16, False)
            == "popops::UnaryOp1D<popops::expr::UnaryOpType::COS,half>"
        )
        assert (
            make_unary1d_vertex_fullname("COS", np.float16, True)
            == "popops::UnaryOp1DInPlace<popops::expr::UnaryOpType::COS,half>"
        )

    def test__ipu_unary_primitive_translation__proper_data_structure(self):
        tile_map_p = ipu_unary_primitive_translation(lax.cos_p, (4, 5, 6), [ShapedArray([10, 12], np.float16)], {})
        assert isinstance(tile_map_p, IpuTileMapEquation)
        assert tile_map_p.vname == "popops::UnaryOp1D<popops::expr::UnaryOpType::COS,half>"
        assert tile_map_p.pname == "cos"
        assert tile_map_p.tiles == [4, 5, 6]
        assert tile_map_p.inputs_info[0].shape == [10, 12]
        assert tile_map_p.inputs_info[0].dtype == IpuType.HALF
        assert tile_map_p.outputs_info[0].shape == [10, 12]
        assert tile_map_p.outputs_info[0].dtype == IpuType.HALF

    # TODO: unified test to speed up things?
    @parameterized.parameters(
        [
            (lax.abs_p, np.float32),
            (lax.asin_p, np.float32),
            (lax.cbrt_p, np.float32),
            (lax.ceil_p, np.float32),
            (lax.cos_p, np.float32),
            (lax.erf_p, np.float32),
            (lax.exp_p, np.float32),
            (lax.expm1_p, np.float32),
            (lax.floor_p, np.float32),
            (lax.log_p, np.float32),
            (lax.log1p_p, np.float32),
            (lax.neg_p, np.float32),
            (lax.population_count_p, np.int32),
            (lax.sign_p, np.float32),
            (lax.sin_p, np.float32),
            (lax.tan_p, np.float32),
            (lax.tanh_p, np.float32),
            # lax.round_p, # TODO: not working, to fix!
            (lax.rsqrt_p, np.float32),
            (lax.sqrt_p, np.float32),
        ]
    )
    def test__tile_map__unary_ops__ipu_jitting__proper_result(self, unary_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(dtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map(unary_p, input)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_cpu(indata)
        output_ipu = compute_fn_ipu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indata.dtype
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=3)

    # TODO: unified test to speed up things?
    @parameterized.parameters(
        [
            (tessellate_ipu.lax.abs_inplace_p, np.float32),
            (tessellate_ipu.lax.asin_inplace_p, np.float32),
            (tessellate_ipu.lax.cbrt_inplace_p, np.float32),
            (tessellate_ipu.lax.ceil_inplace_p, np.float32),
            (tessellate_ipu.lax.erf_inplace_p, np.float32),
            (tessellate_ipu.lax.exp_inplace_p, np.float32),
            (tessellate_ipu.lax.expm1_inplace_p, np.float32),
            (tessellate_ipu.lax.floor_inplace_p, np.float32),
            (tessellate_ipu.lax.log_inplace_p, np.float32),
            (tessellate_ipu.lax.log1p_inplace_p, np.float32),
            (tessellate_ipu.lax.neg_inplace_p, np.float32),
            (tessellate_ipu.lax.population_count_inplace_p, np.int32),
            (tessellate_ipu.lax.sign_inplace_p, np.float32),
            (tessellate_ipu.lax.sin_inplace_p, np.float32),
            (tessellate_ipu.lax.tan_inplace_p, np.float32),
            # (tessellate_ipu.lax.tanh_inplace_p, np.float32), accuracy issue?
            (tessellate_ipu.lax.rsqrt_inplace_p, np.float32),
            (tessellate_ipu.lax.sqrt_inplace_p, np.float32),
        ]
    )
    def test__tile_map__unary_inplace_ops__ipu_jitting__proper_result(self, unary_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(dtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map(unary_p, input)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_cpu(indata)
        output_ipu = compute_fn_ipu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indata.dtype
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=3)

    @parameterized.parameters(
        [
            (np.float32, np.float16),
            (np.float16, np.float32),
            (np.float32, np.int32),
        ]
    )
    def test__tile_map__convert_element_type_op__ipu_jitting__proper_result(self, indtype, outdtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(indtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            # TODO: understand why we need `weak_type` argument as well.
            return tile_map(lax.convert_element_type_p, input, new_dtype=outdtype, weak_type=False)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_ipu(indata)
        output_ipu = compute_fn_cpu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == outdtype
        assert output_ipu.shape == inshape
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=2)

    @parameterized.parameters(
        [
            (np.float32, -1),  # reciprocal/inverse vertex
            (np.float32, 2),  # square vertex
            (np.float16, 2),
            (np.int32, 2),
        ]
    )
    def test__tile_map__integer_pow_op__ipu_jitting__proper_result(self, indtype, pow):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(indtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map(lax.integer_pow_p, input, y=pow)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_ipu(indata)
        output_ipu = compute_fn_cpu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indtype
        assert output_ipu.shape == inshape
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=2)

    @parameterized.parameters(
        [
            (np.int32,),
        ]
    )
    def test__tile_map__iota__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        N = 64

        def compute_fn():
            return tile_map(lax.iota_p, dtype=dtype, dimension=0, shape=(N,), tiles=tiles)

        # compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_ipu = compute_fn_ipu()
        expected_output = np.stack([np.arange(0, N, dtype=dtype)] * len(tiles))
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == dtype
        assert output_ipu.shape == (len(tiles), N)
        npt.assert_array_equal(output_ipu, expected_output)


@pytest.mark.ipu_hardware
class IpuTileUnaryPrimitiveHwTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    def test__tile_map__sqrt_inplace_ops__fori_loop__zero_copy_cycle_count(self):
        N = 1024
        Niter = 32
        # NOTE: skip tile zero where poplar inject loop control vertices.
        tiles = (2,)
        inshape = (len(tiles), N)
        indata = np.random.rand(*inshape).astype(np.float32)

        def inner(_, x):
            return tile_map(tessellate_ipu.lax.sqrt_inplace_p, x)

        def compute_fn(input, num_iters):
            # Hacky way of forcing to wait until all inputs are transferred from HOST.
            # Otherwise: work on tile 2 start before tile 0 (still transferring).
            input = input * num_iters
            input = tile_put_sharded(input, tiles)
            # Make sure number of iterations is on tile0 to avoid additional sync + comms.
            num_iters = tile_put_replicated(num_iters, (0,))

            # Benchmark single call (once all data properly sharded).
            input, num_iters, start = ipu_cycle_count(input, num_iters)
            x = tile_map(tessellate_ipu.lax.sqrt_inplace_p, input)
            x, mid = ipu_cycle_count(x)  # type:ignore
            # Benchmark fori_loop.
            num_iters = num_iters.array[0]
            y = jax.lax.fori_loop(0, num_iters, inner, x)
            y, end = ipu_cycle_count(y)
            start, mid, end = tile_data_barrier(start, mid, end)
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


class IpuTileBinaryPrimitiveTests(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    @parameterized.parameters(
        [
            (lax.add_p, np.float32),
            (lax.atan2_p, np.float32),
            (lax.eq_p, np.float32),
            (lax.ne_p, np.float32),
            (lax.ge_p, np.float32),
            (lax.gt_p, np.float32),
            (lax.le_p, np.float32),
            (lax.lt_p, np.float32),
            (lax.mul_p, np.float32),
            (lax.sub_p, np.float32),
            (lax.div_p, np.float32),
            (lax.max_p, np.float32),
            (lax.min_p, np.float32),
            (lax.pow_p, np.float32),
            (lax.rem_p, np.float32),
        ]
    )
    def test__tile_map__binary_ops__ipu_jitting__proper_result(self, binary_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            return tile_map(binary_p, input0, input1)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_cpu = compute_fn_cpu(input0, input1)
        output_ipu = compute_fn_ipu(input0, input1)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype in (np.bool_, input0.dtype)
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=2)

    @parameterized.parameters(
        [
            (tessellate_ipu.lax.add_inplace_p, np.float32),
            (tessellate_ipu.lax.atan2_inplace_p, np.float32),
            (tessellate_ipu.lax.div_inplace_p, np.float32),
            (tessellate_ipu.lax.max_inplace_p, np.float32),
            (tessellate_ipu.lax.min_inplace_p, np.float32),
            (tessellate_ipu.lax.mul_inplace_p, np.float32),
            (tessellate_ipu.lax.pow_inplace_p, np.float32),
            (tessellate_ipu.lax.rem_inplace_p, np.float32),
            (tessellate_ipu.lax.sub_inplace_p, np.float32),
        ]
    )
    def test__tile_map__binary_inplace_ops__ipu_jitting__proper_result(self, binary_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            return tile_map(binary_p, input0, input1)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_cpu = compute_fn_cpu(input0, input1)
        output_ipu = compute_fn_ipu(input0, input1)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype in (np.bool_, input0.dtype)
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=2)

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map__binary_add__ipu_jitting__proper_result(self, dtype):
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

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map__binary_compare__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            output = tile_map(lax.ge_p, input0, input1)
            return output

        output = compute_fn(input0, input1)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == np.bool_
        npt.assert_array_equal(output.array, input0 >= input1)

    @parameterized.parameters(
        [
            (scaled_add_p, np.float32),
            (scaled_add_p, np.float16),
            (scaled_sub_p, np.float32),
            (scaled_sub_p, np.float16),
        ]
    )
    def test__tile_map__scaled_ops__ipu_jitting__proper_result(self, scale_op_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9, 5)
        A = np.random.randn(*inshape).astype(dtype)
        B = np.random.randn(*inshape).astype(dtype)
        sB = np.random.randn(1).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(A, B, sB):
            A = tile_put_sharded(A, tiles)
            B = tile_put_sharded(B, tiles)
            sB = tile_put_replicated(sB, tiles)
            return tile_map(scale_op_p, A, B, sB)

        output = compute_fn(A, B, sB)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == A.dtype
        npt.assert_array_almost_equal(output.array, scale_op_p.impl(A, B, sB), decimal=2)

    @parameterized.parameters([np.float32, np.float16])
    def test__tile_map__select__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        mask = np.random.rand(*inshape) >= 0.5
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(mask, in0, in1):
            mask = tile_put_sharded(mask, tiles)
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            output = tile_map(lax.select_n_p, mask, input0, input1)
            return output

        output = compute_fn(mask, input0, input1)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input0.dtype
        npt.assert_array_almost_equal(output.array, np.where(mask, input0, input1))

    @parameterized.parameters([np.float32, np.float16])
    def test__tile_map__clamp__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        min = np.random.randn(*inshape).astype(dtype)
        input = np.random.randn(*inshape).astype(dtype)
        max = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(min, x, max):
            min = tile_put_sharded(min, tiles)
            x = tile_put_sharded(x, tiles)
            max = tile_put_sharded(max, tiles)
            output = tile_map(lax.clamp_p, min, x, max)
            return output

        output = compute_fn(min, input, max)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == input.dtype
        npt.assert_array_almost_equal(output.array, np.clip(input, min, max))


class IpuTileShiftPrimitivesTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    @parameterized.parameters([0, 1, 16, 31])  # NOTE: 32 failing!
    def test__tile_map__shift_left__ipu_jitting__proper_result(self, shift):
        tiles = (0,)
        dtype = np.int32
        input0 = np.array([0, 1, 2, 4, 8], dtype=dtype)
        input1 = np.array([shift] * len(input0), dtype=dtype)

        def compute_fn(in0, shift):
            input0 = tile_put_replicated(in0, tiles)
            shift = tile_put_replicated(shift, tiles)
            return tile_map(lax.shift_left_p, input0, shift)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_cpu = compute_fn_cpu(input0, input1)
        output_ipu = compute_fn_ipu(input0, input1)
        npt.assert_array_equal(output_ipu, output_cpu)

    @parameterized.parameters([0, 1, 16, 31])  # NOTE: 32 failing!
    def test__tile_map__shift_right_logical__ipu_jitting__proper_result(self, shift):
        tiles = (0,)
        dtype = np.int32
        input0 = np.array([0, 1, 2, 4, 8], dtype=dtype)
        input1 = np.array([shift] * len(input0), dtype=dtype)

        def compute_fn(in0, shift):
            input0 = tile_put_replicated(in0, tiles)
            shift = tile_put_replicated(shift, tiles)
            return tile_map(lax.shift_right_logical_p, input0, shift)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_cpu = compute_fn_cpu(input0, input1)
        output_ipu = compute_fn_ipu(input0, input1)
        npt.assert_array_equal(output_ipu, output_cpu)


class IpuTileMemcpyTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    # TODO: fix the case of `np.int8` output?
    @parameterized.parameters([np.bool_, np.uint8, np.float16, np.float32])
    def test__tile_copy__multi_size_copy__multi_dtypes(self, dtype):
        tiles = (0,)
        data = np.random.randn(6, 3).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def copy_fn(indata):
            indata = tile_put_replicated(indata, tiles)
            # Check multiple sub-sizes.
            out0 = tile_copy(indata)
            out1 = tile_copy(indata[:, :6])
            out2 = tile_copy(indata[:, :12])
            return out0, out1, out2

        out0, out1, out2 = copy_fn(data)
        assert isinstance(out0, TileShardedArray)
        assert out0.shape == (len(tiles), *data.shape)
        assert out0.dtype == data.dtype
        npt.assert_array_equal(np.asarray(out0)[0], data)
        npt.assert_array_equal(np.asarray(out1)[0, :6], data)
        npt.assert_array_equal(np.asarray(out2)[0, :12], data)

    def test__tile_copy__benchmark_performance(self):
        N = 512
        tiles = (0,)
        data = np.random.randn(N).astype(np.float32)

        @partial(jax.jit, backend="ipu")
        def copy_fn(indata):
            indata = tile_put_replicated(indata, tiles)
            indata, start = ipu_cycle_count(indata)
            outdata = tile_copy(indata)
            outdata, end = ipu_cycle_count(outdata)
            return start, end

        start, end = copy_fn(data)
        # Cycle count. Reference for scale_add: 64(375), 128(467), 256(665), 512(1043)
        start, end = np.asarray(start)[0], np.asarray(end)[0]
        cycle_count = end[0] - start[0]
        # Fairly poor performance at the moment!
        assert cycle_count <= 5000
        # print("CYCLE count:", cycle_count)
        # assert False
