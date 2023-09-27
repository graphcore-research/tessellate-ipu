# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt

from tessellate_ipu import TileShardedArray, tile_map, tile_put_sharded
from tessellate_ipu.lax import bitcast_convert_type_p, reshape_p


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
