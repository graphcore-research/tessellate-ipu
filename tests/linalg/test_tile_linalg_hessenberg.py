# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
import scipy
from absl.testing import parameterized

from tessellate_ipu import ipu_cycle_count, tile_data_barrier
from tessellate_ipu.linalg.tile_linalg_hessenberg import ipu_hessenberg, ipu_hessenberg_iterations
from tessellate_ipu.linalg.tile_linalg_qr import ipu_qr_shard_inputs as ipu_hessenberg_shard_inputs
from tessellate_ipu.utils import IpuTargetType

# Skipping some tests if no local IPU hardware.
ipu_hw_available = len(jax.devices("ipu")) > 0 and jax.devices("ipu")[0].target_type == IpuTargetType.IPU


@pytest.mark.ipu_hardware
class IpuTileLinalgHessenberg(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    @parameterized.parameters(
        {"N": 16},
        {"N": 64},
    )
    def test__linalg_hessenberg_ipu__result_close_to_scipy(self, N):
        # Random symmetric matrix...
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2

        def hessenberg_decomposition_fn(x):
            return ipu_hessenberg(x)

        hessenberg_decomposition_fn_ipu = jax.jit(hessenberg_decomposition_fn, backend="ipu")
        Q, R = hessenberg_decomposition_fn_ipu(x)
        # scipy as reference point!
        Rexp, Qexp = scipy.linalg.hessenberg(x, calc_q=True)

        npt.assert_array_almost_equal(np.abs(Q.array), np.abs(Qexp), decimal=4)  # QR tests have decimal=5
        npt.assert_array_almost_equal(np.abs(R.array), np.abs(Rexp), decimal=4)  # QR tests have decimal=5

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__linalg_hessenberg_ipu__benchmark(self):
        N = 32
        # Random symmetric matrix...
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2
        xsdiag = np.sign(np.diag(x)).astype(x.dtype)

        def hessenberg_decomposition_fn(x, xsdiag):
            # Shard inputs, and wait!
            Q, RT, sdiag = ipu_hessenberg_shard_inputs(x, xsdiag)
            Q, RT, sdiag = tile_data_barrier(Q, RT, sdiag)
            # Benchmark Hessenberg main iterations.
            Q, start = ipu_cycle_count(Q)
            Q, RT = ipu_hessenberg_iterations(Q, RT, sdiag)
            Q, end = ipu_cycle_count(Q)
            return Q, RT, start, end

        hessenberg_decomposition_fn_ipu = jax.jit(hessenberg_decomposition_fn, backend="ipu")
        _, _, start, end = hessenberg_decomposition_fn_ipu(x, xsdiag)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        hessenberg_cycle_count = end[0] - start[0]
        assert hessenberg_cycle_count <= 105000
