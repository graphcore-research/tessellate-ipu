# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from tessellate_ipu import TileShardedArray, tile_map, tile_put_sharded
from tessellate_ipu.lax import bitcast_convert_type_p, fill, reshape_p, tile_fill, tile_sharded_identity


class IpuTileArrayPrimitiveTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    def test__tile_map__reshape__ipu_jitting__proper_result(self):
        tiles = (3, 4, 5)
        dtype = np.float32
        inshape = (len(tiles), 6, 4)
        indata = np.random.randn(*inshape).astype(dtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map(reshape_p, input, new_sizes=(3, 8), dimensions=None)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_cpu(indata)
        output_ipu = compute_fn_ipu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indata.dtype
        npt.assert_array_equal(output_ipu, output_cpu)

    def test__tile_map__bitcast_convert_type__ipu_jitting__proper_result(self):
        tiles = (3, 4, 5)
        dtype = np.float32
        inshape = (len(tiles), 6, 4)
        indata = np.random.randn(*inshape).astype(dtype)

        def compute_fn(input):
            input = tile_put_sharded(input, tiles)
            return tile_map(bitcast_convert_type_p, input, new_dtype=np.int32)

        compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)

        output_cpu = compute_fn_cpu(indata)
        output_ipu = compute_fn_ipu(indata)
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == np.int32
        npt.assert_array_equal(output_ipu, output_cpu)

    @parameterized.parameters(
        [
            (np.int32,),
            (np.float16,),
            (np.float32,),
        ]
    )
    def test__tile_map__fill__ipu_jitting__proper_result(self, dtype):
        tiles = (3, 1, 5)
        shape = (4, 5)
        fill_value = 1

        def compute_fn():
            return tile_fill(shape, fill_value, dtype=dtype, tiles=tiles)

        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_ipu = compute_fn_ipu()
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == dtype
        npt.assert_array_equal(output_ipu, np.full((len(tiles), *shape), fill_value, dtype=dtype))

    def test__tile_map__fill__cpu_jitting__proper_result(self):
        shape = (4, 5)
        fill_value = 2

        def compute_fn():
            return fill(shape, fill_value, np.float32)

        fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        output_cpu = fn_cpu()
        assert output_cpu.dtype == np.float32
        npt.assert_array_equal(output_cpu, np.full(shape, fill_value, dtype=np.float32))

    def test__tile_sharded_identity__ipu_jitting__proper_result(self):
        dtype = np.float32
        tiles = (1, 2, 5)
        N = len(tiles)

        def fn():
            # Comparison point with the "obvious" way using JAX Numpy.
            # return tile_put_sharded(jax.numpy.identity(N, dtype), tiles=tiles)
            return tile_sharded_identity(dtype, tiles)

        fn_ipu = partial(jax.jit, backend="ipu")(fn)
        output_ipu = fn_ipu()
        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == dtype
        npt.assert_array_equal(output_ipu, np.identity(N, dtype=dtype))
