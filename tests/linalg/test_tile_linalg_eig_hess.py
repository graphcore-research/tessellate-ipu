# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized

from tessellate_ipu.linalg.tile_linalg_tridiagonal_eigh import ipu_hess_eigh
from tessellate_ipu.utils import IpuTargetType

# Skipping some tests if no local IPU hardware.
ipu_hw_available = len(jax.devices("ipu")) > 0 and jax.devices("ipu")[0].target_type == IpuTargetType.IPU
ipu_num_tiles = jax.devices("ipu")[0].num_tiles


@pytest.mark.ipu_hardware
class IpuTileLinalgHessEigh(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    @unittest.skipUnless(ipu_num_tiles >= 16, "Requires IPU with 16 tiles")
    @parameterized.parameters(
        {"N": 4},
        # {"N": 512},
    )
    def test__hess_eigh_raw__proper_eigh_result(self, N):
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        hess_eigh_fn = jax.jit(ipu_hess_eigh, backend="ipu")
        # Should be enough iterations...
        eigvalues, VT = hess_eigh_fn(x, num_iters=2)
        eigvalues = np.asarray(eigvalues).reshape(-1)
        VT = np.asarray(VT)
        # Expected eigen values and vectors (from Lapack?)
        expected_eigvalues, expected_eigvectors = np.linalg.eigh(x)

        # Order raw outputs.
        indices = np.argsort(eigvalues)
        eigvalues_sorted = eigvalues[indices]
        eigvectors_sorted = VT[indices].T
        npt.assert_array_almost_equal(eigvalues_sorted, expected_eigvalues, decimal=5)
        npt.assert_array_almost_equal(np.abs(eigvectors_sorted), np.abs(expected_eigvectors), decimal=5)

    # TODO: Performance test
