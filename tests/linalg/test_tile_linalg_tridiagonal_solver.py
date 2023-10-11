# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import scipy
from absl.testing import parameterized

from tessellate_ipu import ipu_cycle_count, tile_data_barrier, tile_map
from tessellate_ipu.linalg.tile_linalg_tridiagonal_solver import (
    ipu_tridiag_solve,
    ipu_tridiag_solve_shard_inputs,
    tridiagonal_solver_p,
)
from tessellate_ipu.utils import IpuTargetType

# Skipping some tests if no local IPU hardware.
ipu_hw_available = len(jax.devices("ipu")) > 0 and jax.devices("ipu")[0].target_type == IpuTargetType.IPU

jax.config.FLAGS.jax_platform_name = "cpu"
jax.config.update("jax_enable_x64", False)


@pytest.mark.ipu_hardware
class IpuTileLinalgTridiagonalSolver(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    @parameterized.parameters(
        {"N": 64},
        {"N": 1472},
    )
    def test__linalg_tridiagonal_solver_ipu__correctness(self, N):
        M = N
        # Random symmetric tridiagonal matrices...
        diag = np.random.rand(M, N).astype(jnp.float32)
        udiag = np.random.rand(M, N).astype(jnp.float32)
        ldiag = np.roll(udiag, 1, axis=1)
        rhs = np.random.rand(M, N).astype(jnp.float32)

        x_ = jax.jit(ipu_tridiag_solve, backend="ipu")(diag, udiag, ldiag, rhs)

        x = np.array(x_.array)

        deltas = []
        for i in range(M):
            data = np.vstack(
                [np.roll(udiag[i].flat, 1, axis=0), diag[i].flat, np.roll(ldiag[i].flat, -1, axis=0)],
            )
            T = scipy.sparse.spdiags(data, (1, 0, -1), N, N).toarray()

            delta = T @ x[i].reshape(N, 1) - rhs[i].reshape(N, 1)

            deltas.append(delta)

        npt.assert_almost_equal(np.array(deltas), 0, decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__linalg_tridiagonal_solver_ipu__benchmark(self):
        N = 1472
        M = 1
        # Random symmetric tridiagonal matrices...
        diag = np.random.rand(M, N).astype(jnp.float32)
        udiag = np.random.rand(M, N).astype(jnp.float32)
        rhs = np.random.rand(M, N).astype(jnp.float32)

        def tridiag_solver_fn(diag, udiag, ldiag, rhs):
            ts, tus, tls, b = ipu_tridiag_solve_shard_inputs(diag, udiag, ldiag, rhs)

            ts, tus, tls, b = tile_data_barrier(ts, tus, tls, b)
            # Benchmark the solver.
            ts, start = ipu_cycle_count(ts)
            x = tile_map(tridiagonal_solver_p, ts, tus, tls, b)
            x, end = ipu_cycle_count(x)  # type: ignore
            return x, start, end

        tridiag_solver_fn_ipu = jax.jit(tridiag_solver_fn, backend="ipu")
        _, start, end = tridiag_solver_fn_ipu(diag, udiag, np.roll(udiag, 1, axis=1), rhs)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        cycle_count = end[0] - start[0]
        assert cycle_count <= 180000
