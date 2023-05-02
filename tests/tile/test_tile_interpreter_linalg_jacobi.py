# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
import scipy.linalg
from absl.testing import parameterized

from jax_ipu_experimental_addons.tile import (
    ipu_hw_cycle_count,
    tile_data_barrier,
    tile_map_primitive,
    tile_put_replicated,
)
from jax_ipu_experimental_addons.tile.tile_interpreter_linalg_jacobi import (
    ipu_eigh,
    ipu_jacobi_eigh,
    jacobi_initial_rotation_set,
    jacobi_next_rotation_set,
    jacobi_sym_schur2_p,
    jacobi_update_eigenvectors_p,
    jacobi_update_first_step_p,
)
from jax_ipu_experimental_addons.utils import IpuTargetType

# Skipping some tests if no local IPU hardware.
ipu_hw_available = len(jax.devices("ipu")) > 0 and jax.devices("ipu")[0].target_type == IpuTargetType.IPU


class IpuTileLinalgJacobi(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    def test__jacobi_initial_rotation_set__proper_set_indexes(self):
        N = 8
        rot = jacobi_initial_rotation_set(N)
        assert rot.shape == (N // 2, 2)
        assert rot.dtype == np.uint32
        npt.assert_array_equal(rot.flatten(), np.arange(0, N))

    def test__jacobi_next_rotation_set__proper_set_indexes(self):
        N = 8
        rot = jacobi_initial_rotation_set(N)
        rot = jacobi_next_rotation_set(rot)
        npt.assert_array_equal(rot[:, 0], [0, 1, 2, 4])
        npt.assert_array_equal(rot[:, 1], [3, 5, 7, 6])

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    @parameterized.parameters(
        {"data": [[1.0, 2.0], [2.0, 3.0]]},
        {"data": [[3.0, 2.0], [2.0, 1.0]]},
        {"data": [[1.0, 0.0], [0.0, 3.0]]},
        {"data": [[1.0, 2.0], [2.0, 1.0]]},
    )
    def test__jacobi_sym_schur2_vertex__accurate_result(self, data):
        tiles = (0,)
        pq = np.array([0, 1], dtype=np.uint32)
        data = np.array(data, dtype=np.float32)

        def jacobi_sym_schur2_fn(pq, pcol, qcol):
            pq = tile_put_replicated(pq, tiles)
            pcol = tile_put_replicated(pcol, tiles)
            qcol = tile_put_replicated(qcol, tiles)
            cs = tile_map_primitive(jacobi_sym_schur2_p, pq, pcol, qcol)
            return cs

        jacobi_sym_schur2_fn = jax.jit(jacobi_sym_schur2_fn, backend="ipu")

        cs = jacobi_sym_schur2_fn(pq, data[0], data[1])
        expected_Jschur = scipy.linalg.schur(data)[1]
        npt.assert_array_almost_equal(np.abs(cs)[0], np.abs(expected_Jschur[0]))

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_sym_schur2_vertex__benchmark_performance(self):
        N = 128
        tiles = (0,)
        pq = np.array([3, N // 2], dtype=np.uint32)
        pcol = np.random.randn(N).astype(np.float32)

        def jacobi_sym_schur2_fn(pq, pcol, qcol):
            pq = tile_put_replicated(pq, tiles)
            pcol = tile_put_replicated(pcol, tiles)
            qcol = tile_put_replicated(qcol, tiles)
            # Force synchronization at this point, before cycle count.
            pq, pcol, qcol = tile_data_barrier(pq, pcol, qcol)
            pcol, start = ipu_hw_cycle_count(pcol)
            cs = tile_map_primitive(jacobi_sym_schur2_p, pq, pcol, qcol)
            cs, end = ipu_hw_cycle_count(cs)  # type:ignore
            return cs, start, end

        jacobi_sym_schur2_fn = jax.jit(jacobi_sym_schur2_fn, backend="ipu")
        _, start, end = jacobi_sym_schur2_fn(pq, pcol, pcol)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        qr_correction_cycle_count = end[0] - start[0]
        assert qr_correction_cycle_count <= 310
        # print("CYCLE count:", qr_correction_cycle_count)
        # assert False

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_update_first_step_vertex__benchmark_performance(self):
        N = 128
        tiles = (0,)
        pq = np.array([3, N // 2], dtype=np.uint32)
        pcol = np.random.randn(N).astype(np.float32)
        qcol = np.random.randn(N).astype(np.float32)

        def jacobi_update_first_step_fn(pq, pcol, qcol):
            pq = tile_put_replicated(pq, tiles)
            pcol = tile_put_replicated(pcol, tiles)
            qcol = tile_put_replicated(qcol, tiles)
            # Force synchronization at this point, before cycle count.
            pq, pcol, qcol = tile_data_barrier(pq, pcol, qcol)
            pcol, start = ipu_hw_cycle_count(pcol)
            cs, _, _ = tile_map_primitive(  # type:ignore
                jacobi_update_first_step_p, pq, pcol, qcol, N=N
            )
            cs, end = ipu_hw_cycle_count(cs)
            return cs, start, end

        jacobi_update_first_step_fn = jax.jit(jacobi_update_first_step_fn, backend="ipu")
        _, start, end = jacobi_update_first_step_fn(pq, pcol, qcol)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        qr_correction_cycle_count = end[0] - start[0]
        assert qr_correction_cycle_count <= 1600
        # print("CYCLE count:", qr_correction_cycle_count)
        # assert False

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_update_eigenvectors_vertex__benchmark_performance(self):
        N = 256
        tiles = (0,)
        cs = np.array([0.2, 0.5], dtype=np.float32)
        pcol = np.random.randn(N).astype(np.float32)
        qcol = np.random.randn(N).astype(np.float32)

        def jacobi_update_eigenvectors_fn(cs, pcol, qcol):
            cs = tile_put_replicated(cs, tiles)
            pcol = tile_put_replicated(pcol, tiles)
            qcol = tile_put_replicated(qcol, tiles)
            # Force synchronization at this point, before cycle count.
            cs, pcol, qcol = tile_data_barrier(cs, pcol, qcol)
            pcol, start = ipu_hw_cycle_count(pcol)
            pcol, qcol = tile_map_primitive(  # type:ignore
                jacobi_update_eigenvectors_p, cs, pcol, qcol
            )
            pcol, end = ipu_hw_cycle_count(pcol)
            return pcol, qcol, start, end

        jacobi_update_eigenvectors_fn = jax.jit(jacobi_update_eigenvectors_fn, backend="ipu")
        pcol_updated, qcol_updated, start, end = jacobi_update_eigenvectors_fn(cs, pcol, qcol)
        pcol_updated = np.asarray(pcol_updated)
        qcol_updated = np.asarray(qcol_updated)

        # Make sure we have the right result!
        npt.assert_array_almost_equal(pcol_updated[0], pcol * cs[0] - qcol * cs[1])
        npt.assert_array_almost_equal(qcol_updated[0], pcol * cs[1] + qcol * cs[0])

        # Cycle count reference for scale_add: 64(375), 128(467), 256(665), 512(1043)
        start, end = np.asarray(start)[0], np.asarray(end)[0]
        qr_correction_cycle_count = end[0] - start[0]
        assert qr_correction_cycle_count <= 2000
        # print("CYCLE count:", qr_correction_cycle_count)
        # assert False

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_eigh__single_iteration(self):
        N = 32
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        jacobi_eigh_fn = jax.jit(ipu_jacobi_eigh, backend="ipu", static_argnums=(1,))
        A, _ = jacobi_eigh_fn(x, num_iters=1)
        A = np.asarray(A)
        # Should be still symmetric.
        npt.assert_array_almost_equal(A.T, A)
        # Same eigenvalues.
        npt.assert_array_almost_equal(np.linalg.eigh(A)[0], np.linalg.eigh(x)[0], decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_eigh_raw__proper_eigh_result(self):
        N = 8
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        jacobi_eigh_fn = jax.jit(ipu_jacobi_eigh, backend="ipu", static_argnums=(1,))
        # Should be enough iterations...
        A, VT = jacobi_eigh_fn(x, num_iters=10)
        A = np.asarray(A)
        VT = np.asarray(VT)
        eigvalues = np.diag(A)
        # Expected eigen values and vectors (from Lapack?)
        expected_eigvalues, expected_eigvectors = np.linalg.eigh(x)

        # Order raw outputs.
        indices = np.argsort(eigvalues)
        eigvalues_sorted = eigvalues[indices]
        eigvectors_sorted = VT[indices].T
        npt.assert_array_almost_equal(eigvalues_sorted, expected_eigvalues, decimal=5)
        npt.assert_array_almost_equal(np.abs(eigvectors_sorted), np.abs(expected_eigvectors), decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_eigh__not_sorting(self):
        N = 8
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        ipu_eigh_fn = jax.jit(lambda x: ipu_eigh(x, sort_eigenvalues=False, num_iters=5), backend="ipu")
        # Should be enough iterations...
        eigvectors, eigvalues = ipu_eigh_fn(x)
        eigvalues = np.asarray(eigvalues)
        eigvectors = np.asarray(eigvectors)
        # Expected eigen values and vectors (from Lapack?)
        expected_eigvalues, expected_eigvectors = np.linalg.eigh(x)

        # Order raw outputs.
        indices = np.argsort(eigvalues)
        eigvalues_sorted = eigvalues[indices]
        eigvectors_sorted = eigvectors.T[indices].T
        npt.assert_array_almost_equal(eigvalues_sorted, expected_eigvalues, decimal=5)
        npt.assert_array_almost_equal(np.abs(eigvectors_sorted), np.abs(expected_eigvectors), decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_eigh__sorting(self):
        N = 8
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        ipu_eigh_fn = jax.jit(lambda x: ipu_eigh(x, sort_eigenvalues=True, num_iters=5), backend="ipu")
        # Should be enough iterations...
        eigvectors, eigvalues = ipu_eigh_fn(x)
        eigvalues = np.asarray(eigvalues)
        eigvectors = np.asarray(eigvectors)
        # Expected eigen values and vectors (from Lapack?)
        expected_eigvalues, expected_eigvectors = np.linalg.eigh(x)

        npt.assert_array_almost_equal(eigvalues, expected_eigvalues, decimal=5)
        npt.assert_array_almost_equal(np.abs(eigvectors), np.abs(expected_eigvectors), decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_eigh__sorting_failure_case(self):
        # Trivial diagonalization, but with multiple identical eigen values.
        x = np.diag([1.0, 2.0, 3.0, 2.0]).astype(np.float32)

        ipu_eigh_fn = jax.jit(lambda x: ipu_eigh(x, sort_eigenvalues=True, num_iters=5), backend="ipu")
        # Should be enough iterations...
        eigvectors, eigvalues = ipu_eigh_fn(x)
        eigvalues = np.asarray(eigvalues)
        eigvectors = np.asarray(eigvectors)
        # Expected eigen values and vectors (from Lapack?)
        expected_eigvalues, expected_eigvectors = np.linalg.eigh(x)

        npt.assert_array_almost_equal(eigvalues, expected_eigvalues, decimal=5)
        npt.assert_array_almost_equal(np.abs(eigvectors), np.abs(expected_eigvectors), decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__jacobi_eigh__jit_multi_calls__reused_buffer_bug(self):
        N = 4
        x = np.random.randn(N, N).astype(np.float32)
        x0 = (x + x.T) / 2.0
        x1 = x0 @ x0.T

        @partial(jax.jit, backend="ipu")
        def compute_fn(x0, x1):
            # Two calls: make sure we don't reuse the same buffers.
            eigvecs0, _ = ipu_eigh(x0, sort_eigenvalues=True, num_iters=4)
            eigvecs1, _ = ipu_eigh(x1, sort_eigenvalues=True, num_iters=4)
            return eigvecs0, eigvecs1

        eigvecs0, eigvecs1 = compute_fn(x0, x1)
        # Expected eigen values and vectors (from Lapack?)
        _, expected_eigvectors0 = np.linalg.eigh(x0)
        _, expected_eigvectors1 = np.linalg.eigh(x1)
        # Both eigenvectors should be accurate.
        npt.assert_array_almost_equal(np.abs(eigvecs0), np.abs(expected_eigvectors0), decimal=4)
        npt.assert_array_almost_equal(np.abs(eigvecs1), np.abs(expected_eigvectors1), decimal=4)
