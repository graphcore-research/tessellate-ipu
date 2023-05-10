# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized

from jax_ipu_experimental_addons import is_ipu_model
from jax_ipu_experimental_addons.tile import TileShardedArray, ipu_cycle_count, tile_put_replicated


@pytest.mark.ipu_hardware
class IpuTileHardwarePrimitives(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles

    @parameterized.parameters({"sync": True}, {"sync": False})
    def test__ipu_cycle_count__proper_hw_counter(self, sync: bool):
        tiles = (1, 2, self.num_tiles - 1)
        val = np.random.rand(1).astype(np.float32)

        @partial(jax.jit, backend="ipu")
        def compute_fn(val):
            val = tile_put_replicated(val, tiles)
            val, cycles0 = ipu_cycle_count(val, sync=sync)
            val, cycles1 = ipu_cycle_count(val, sync=False)
            return cycles0, cycles1

        cycles0_ipu, cycles1_ipu = compute_fn(val)

        assert isinstance(cycles0_ipu, TileShardedArray)
        assert cycles0_ipu.tiles == tiles
        assert cycles0_ipu.dtype == np.uint32
        assert cycles0_ipu.shape == (len(tiles), 2)

        cycles0_ipu = np.asarray(cycles0_ipu)
        cycles1_ipu = np.asarray(cycles1_ipu)
        diff_cycles_count = cycles1_ipu - cycles0_ipu

        if is_ipu_model(jax.devices("ipu")[0]):
            # IPU model: zero cycle count.
            npt.assert_equal(diff_cycles_count, 0)
        else:
            # Real IPU hw.
            npt.assert_equal(diff_cycles_count[:, 1], 0)
            npt.assert_equal(diff_cycles_count[:, 0] > 30, True)
            npt.assert_equal(diff_cycles_count[:, 0] <= 150, True)
            # Should be the same accross all tiles, even without synchronization.
            npt.assert_equal(diff_cycles_count[:, 0], diff_cycles_count[0, 0])

    @parameterized.parameters(
        {"dtype": np.float32},
        {"dtype": np.float16},
        {"dtype": np.int32},
        {"dtype": np.int16},
        {"dtype": np.int8},
        {"dtype": np.uint32},
        # {"dtype": np.uint16}, Not supported by XLA backend? FAILED_PRECONDITION error.
        {"dtype": np.uint8},
        {"dtype": np.bool_},
    )
    def test__ipu_cycle_count__proper_barrier_dtypes_support(self, dtype):
        tiles = (0,)
        val = np.random.randn(8).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(val):
            val = tile_put_replicated(val, tiles)
            val, cycles = ipu_cycle_count(val, sync=False)
            return val, cycles

        compute_fn(val)

    @parameterized.parameters(
        {"dtype": np.float32},
    )
    def test__ipu_cycle_count__multiple_inputs(self, dtype):
        tiles = (0, 1)
        data = np.random.randn(8).astype(dtype)

        @partial(jax.jit, backend="ipu")
        def compute_fn(data):
            val0 = tile_put_replicated(data, tiles)
            val1 = tile_put_replicated(data, tiles)
            val0, val1, cycles = ipu_cycle_count(val0, val1, sync=False)
            return val0, val1, cycles

        compute_fn(data)
