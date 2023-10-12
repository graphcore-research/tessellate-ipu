# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest

from tessellate_ipu import ipu_cycle_count, tile_map, tile_put_replicated
from tessellate_ipu.lax.tile_lax_small_dot import rotation2d_p


@pytest.mark.ipu_hardware
class IpuTileRotation2dHwTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(42)

    def test__tile_map__rotation2d_primitive__proper_result_and_cycle_count(self):
        N = 512
        tiles = (0,)
        indata = np.random.randn(2, N).astype(np.float32)
        cs = np.random.randn(2).astype(np.float32)
        rot2d = np.array([[cs[0], -cs[1]], [cs[1], cs[0]]]).astype(np.float32)

        def compute_fn(cs, row0, row1):
            cs = tile_put_replicated(cs, tiles)
            row0 = tile_put_replicated(row0, tiles)
            row1 = tile_put_replicated(row1, tiles)
            # Benchmark the raw 2d rotation vertex.
            cs, row0, row1, start = ipu_cycle_count(cs, row0, row1)
            outrow0, outrow1 = tile_map(rotation2d_p, cs, row0, row1)  # type:ignore
            outrow0, outrow1, end = ipu_cycle_count(outrow0, outrow1)

            return outrow0, outrow1, start, end

        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        outrow0, outrow1, start, end = compute_fn_ipu(cs, indata[0], indata[1])

        # Checking getting the proper result!
        expected_out = rot2d @ indata
        npt.assert_array_almost_equal(np.ravel(outrow0), expected_out[0], decimal=6)
        npt.assert_array_almost_equal(np.ravel(outrow1), expected_out[1], decimal=6)
        # Hardware cycle count bound.
        start, end = np.asarray(start)[0], np.asarray(end)[0]
        hw_cycle_count = end[0] - start[0]
        # Observe on IPU Mk2 hw ~1916 cycles.
        assert hw_cycle_count <= 2000
