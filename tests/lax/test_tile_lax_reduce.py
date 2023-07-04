# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from tessellate_ipu import TileShardedArray, tile_map_primitive, tile_put_sharded
from tessellate_ipu.lax.tile_lax_reduce import make_continuous_reduce_vertex_fullname


def test__make_continuous_reduce_vertex_fullname__proper_name():
    fullname = make_continuous_reduce_vertex_fullname(jax.lax.reduce_prod_p, np.float32, np.float16, False)
    assert fullname == "popops::ContinuousReduce<popops::ReduceMul,float,half,false>"


class IpuTilePrimitivesLaxReduce(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        # Not very clean, but for better reproducibility.
        np.random.seed(42)

    @parameterized.parameters(
        {"backend": "cpu", "jitting": False},
        {"backend": "cpu", "jitting": True},
        {"backend": "ipu", "jitting": False},
        {"backend": "ipu", "jitting": True},
    )
    def test__tile_map_primitive__reduce__jitting_options__proper_result(self, backend, jitting):
        device = jax.devices(backend)[0]
        tiles = (1, 2, self.num_tiles - 1)
        indata = np.random.randn(len(tiles), 100).astype(np.float32)
        reduce_p = jax.lax.reduce_sum_p

        def compute_fn(in0):
            input0 = tile_put_sharded(in0, tiles)
            output = tile_map_primitive(reduce_p, input0, axes=(0,))
            return output

        compute_fn = partial(jax.jit, backend=backend)(compute_fn) if jitting else compute_fn
        input_device = jax.device_put(indata, device)
        output_device = compute_fn(input_device)

        assert isinstance(output_device, TileShardedArray)
        assert output_device.tiles == tiles
        assert output_device.dtype == indata.dtype
        assert output_device.array.device().platform == device.platform
        npt.assert_array_almost_equal(output_device.array, np.sum(indata, axis=1), decimal=2)

    @parameterized.parameters(
        {"dtype": np.float32, "reduce_p": jax.lax.reduce_sum_p},
        {"dtype": np.float32, "reduce_p": jax.lax.reduce_max_p},
        {"dtype": np.float32, "reduce_p": jax.lax.reduce_min_p},
        {"dtype": np.float32, "reduce_p": jax.lax.reduce_prod_p},
        {"dtype": np.float16, "reduce_p": jax.lax.reduce_sum_p},
        {"dtype": np.float16, "reduce_p": jax.lax.reduce_max_p},
        {"dtype": np.float16, "reduce_p": jax.lax.reduce_min_p},
        {"dtype": np.float16, "reduce_p": jax.lax.reduce_prod_p},
        {"dtype": np.bool_, "reduce_p": jax.lax.reduce_or_p},
        {"dtype": np.bool_, "reduce_p": jax.lax.reduce_and_p},
    )
    def test__tile_map_primitive__reduce_primitives__ipu_jitting__proper_result(self, dtype, reduce_p):
        tiles = (1, 2, self.num_tiles - 1)
        indata = np.random.randn(len(tiles), 13).astype(dtype)

        def compute_fn(in0):
            input0 = tile_put_sharded(in0, tiles)
            output = tile_map_primitive(reduce_p, input0, axes=(0,))
            return output

        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_ipu = compute_fn_ipu(indata)
        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        output_cpu = compute_fn_cpu(indata)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indata.dtype
        assert output_ipu.shape == output_cpu.shape
        npt.assert_array_almost_equal(output_ipu.array, output_cpu, decimal=2)

    @parameterized.parameters(
        {"dtype": np.float32, "axes": (1, 2, 3)},
        {"dtype": np.float32, "axes": (2, 3)},
        {"dtype": np.float32, "axes": (3,)},
    )
    def test__tile_map_primitive__reduce_sum_partial__ipu_jitting__proper_result(self, dtype, axes):
        tiles = (1, 2, self.num_tiles - 1)
        shape = (3, 5, 7, 9)
        indata = np.random.randn(len(tiles), *shape).astype(dtype)
        reduce_p = jax.lax.reduce_sum_p

        def compute_fn(in0):
            input0 = tile_put_sharded(in0, tiles)
            output = tile_map_primitive(reduce_p, input0, axes=axes)
            return output

        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_ipu = compute_fn_ipu(indata)
        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        output_cpu = compute_fn_cpu(indata)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indata.dtype
        assert output_ipu.shape == output_cpu.shape
        npt.assert_array_almost_equal(output_ipu.array, output_cpu, decimal=2)

    @parameterized.parameters(
        {"dtype": np.float32, "axes": (1, 3)},
        {"dtype": np.float32, "axes": (2,)},
    )
    def test__tile_map_primitive__reduce_partial__unsupported_axes(self, dtype, axes):
        tiles = (1, 2, self.num_tiles - 1)
        shape = (3, 5, 7, 9)

        @partial(jax.jit, backend="ipu")
        def compute_fn(in0):
            input0 = tile_put_sharded(in0, tiles)
            output = tile_map_primitive(jax.lax.reduce_sum_p, input0, axes=axes)
            return output

        indata = np.random.randn(len(tiles), *shape).astype(dtype)
        with self.assertRaises(NotImplementedError):
            compute_fn(indata)
