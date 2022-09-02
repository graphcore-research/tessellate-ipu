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


class IpuTileBinaryPrimitiveTests(chex.TestCase, parameterized.TestCase):
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
            output = tile_map_primitive(lax.ge_p, [input0, input1])
            return output

        output = compute_fn(input0, input1)
        assert isinstance(output, TileShardedArray)
        assert output.tiles == tiles
        assert output.dtype == np.bool_
        npt.assert_array_equal(output.array, input0 >= input1)
