# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax import lax
from jax.core import ShapedArray

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_sharded
from jax_ipu_research.tile.tile_interpreter_lax_unary import (
    ipu_unary_primitive_translation,
    make_unary1d_vertex_fullname,
)
from jax_ipu_research.tile.tile_interpreter_primitives_impl import IpuTileMapEquation, IpuType


class IpuTileUnaryPrimitiveTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    def test__make_unary1d_vertex_fullname__proper_result(self):
        assert (
            make_unary1d_vertex_fullname("COS", np.float16) == "popops::UnaryOp1D<popops::expr::UnaryOpType::COS,half>"
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
    def test__tile_map_primitive__unary_ops__ipu_jitting__proper_result(self, unary_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(dtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map_primitive(unary_p, input)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_ipu(indata)
        output_ipu = compute_fn_cpu(indata)
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
    def test__tile_map_primitive__convert_element_type_op__ipu_jitting__proper_result(self, indtype, outdtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(indtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            # TODO: understand why we need `weak_type` argument as well.
            return tile_map_primitive(lax.convert_element_type_p, input, new_dtype=outdtype, weak_type=False)

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
            (np.float32, 2),
            (np.float16, 2),
            (np.int32, 2),
        ]
    )
    def test__tile_map_primitive__integer_pow_op__ipu_jitting__proper_result(self, indtype, pow):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        indata = np.random.randn(*inshape).astype(indtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map_primitive(lax.integer_pow_p, input, y=pow)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_ipu(indata)
        output_ipu = compute_fn_cpu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indtype
        assert output_ipu.shape == inshape
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=2)


class IpuTileBinaryPrimitiveTests(chex.TestCase, parameterized.TestCase):
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
    def test__tile_map_primitive__binary_ops__ipu_jitting__proper_result(self, binary_p, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            return tile_map_primitive(binary_p, input0, input1)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_cpu = compute_fn_cpu(input0, input1)
        output_ipu = compute_fn_ipu(input0, input1)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype in (np.bool_, input0.dtype)
        npt.assert_array_almost_equal(output_ipu, output_cpu, decimal=2)

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__binary_add__ipu_jitting__proper_result(self, dtype):
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

    @parameterized.parameters([np.float32, np.float16, np.int32])
    def test__tile_map_primitive__binary_compare__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 4, 5)
        inshape = (len(tiles), 7, 9)
        input0 = np.random.randn(*inshape).astype(dtype)
        input1 = np.random.randn(*inshape).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(in0, in1):
            input0 = tile_put_sharded(in0, tiles)
            input1 = tile_put_sharded(in1, tiles)
            output = tile_map_primitive(lax.ge_p, input0, input1)
            return output

        output = compute_fn(input0, input1)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == np.bool_
        npt.assert_array_equal(output.array, input0 >= input1)
