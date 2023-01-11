# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import chex
import jax
import numpy as np
import numpy.testing as npt
import scipy.linalg
from absl.testing import parameterized

from jax_ipu_research.tile import (
    ipu_hw_cycle_count,
    tile_data_barrier,
    tile_map_primitive,
    tile_put_replicated,
    tile_put_sharded,
)
from jax_ipu_research.tile.tile_interpreter_linalg_jacobi import (
    ipu_jacobi_eigh,
    jacobi_initial_rotation_set,
    jacobi_next_rotation_set,
    jacobi_rotate_columns,
    jacobi_sort_columns,
    jacobi_sym_schur2_p,
    jacobi_update_first_step_p,
)


class IpuTileLinalgJacobi(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    def test__jacobi_initial_rotation_set__proper_set_indexes(self):
        N = 8
        rot = jacobi_initial_rotation_set(N)
        assert rot.shape == (N // 2, 2)
        assert rot.dtype == np.int32
        npt.assert_array_equal(rot.flatten(), np.arange(0, N))

    def test__jacobi_next_rotation_set__proper_set_indexes(self):
        N = 8
        rot = jacobi_initial_rotation_set(N)
        rot = jacobi_next_rotation_set(rot)
        npt.assert_array_equal(rot[:, 0], [0, 1, 2, 4])
        npt.assert_array_equal(rot[:, 1], [3, 5, 7, 6])

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
        assert qr_correction_cycle_count <= 306
        # print("CYCLE count:", qr_correction_cycle_count)
        # assert False

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

    def test__jacobi_rotate_columns__proper_result(self):
        N = 8
        rot = jacobi_initial_rotation_set(N)
        rot = jacobi_next_rotation_set(rot)

        # Just need to test on CPU? TODO: IPU as well?
        jacobi_rotate_fn = jax.jit(jacobi_rotate_columns, backend="cpu")
        rot0, rot1 = jacobi_rotate_fn(rot[:, 0], rot[:, 1])
        expected_rot = jacobi_next_rotation_set(rot)

        npt.assert_array_equal(rot0, expected_rot[:, 0])
        npt.assert_array_equal(rot1, expected_rot[:, 1])

    def test__jacobi_sort_columns__proper_result(self):
        N = 8
        tiles = (0, 1, 2, 3)
        # Find interesting rotation set!
        rot = jacobi_initial_rotation_set(N)
        for _ in range(4):
            rot = jacobi_next_rotation_set(rot)

        def jacobi_sort_fn(pcols, qcols):
            pcols = tile_put_sharded(pcols, tiles)
            qcols = tile_put_sharded(qcols, tiles)
            return jacobi_sort_columns(rot, pcols, qcols)

        # Just need to test on CPU? TODO: IPU as well?
        jacobi_sort_fn = jax.jit(jacobi_sort_fn, backend="cpu")
        # Dummy pcols and qcols, corresponding to rotation data.
        rot_sorted, pcols_sorted, qcols_sorted = jacobi_sort_fn(rot[:, 0], rot[:, 1])
        rot_sorted = np.asarray(rot_sorted)

        # Proper sorting of columns.
        npt.assert_array_equal(rot_sorted[:, 0], np.min(rot, axis=1))
        npt.assert_array_equal(rot_sorted[:, 1], np.max(rot, axis=1))
        npt.assert_array_equal(pcols_sorted, rot_sorted[:, 0])
        npt.assert_array_equal(qcols_sorted, rot_sorted[:, 1])

    def test__jacobi_eigh__single_iteration(self):
        N = 16
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        jacobi_eigh_fn = jax.jit(ipu_jacobi_eigh, backend="ipu", static_argnums=(1,))
        A, _ = jacobi_eigh_fn(x, num_iters=1)
        # Should be still symmetric.
        npt.assert_array_almost_equal(A.T, A)
        # Same eigenvalues.
        npt.assert_array_almost_equal(np.linalg.eigh(A)[0], np.linalg.eigh(x)[0])

    def test__jacobi_eigh__proper_eigh_result(self):
        N = 8
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2.0

        jacobi_eigh_fn = jax.jit(ipu_jacobi_eigh, backend="ipu", static_argnums=(1,))
        # Should be enough iterations...
        A, _ = jacobi_eigh_fn(x, num_iters=5)

        eigvalues = np.diag(A)
        expected_eigvalues, _ = np.linalg.eigh(x)
        npt.assert_array_almost_equal(np.sort(eigvalues), expected_eigvalues, decimal=5)
