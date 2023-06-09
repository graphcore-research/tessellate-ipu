# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import unittest

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized

from jax_ipu_experimental_addons.tile import (
    TileShardedArray,
    ipu_cycle_count,
    tile_data_barrier,
    tile_map_primitive,
    tile_put_replicated,
    tile_put_sharded,
)
from jax_ipu_experimental_addons.tile.tile_interpreter_linalg_qr import (
    dot_product1d_p,
    ipu_qr,
    ipu_qr_iterations,
    ipu_qr_shard_inputs,
    make_ipu_vector1d_worker_offsets,
    qr_correction_vector_p,
    qr_householder_row_update_p,
)
from jax_ipu_experimental_addons.utils import IpuTargetType

# Skipping some tests if no local IPU hardware.
ipu_hw_available = len(jax.devices("ipu")) > 0 and jax.devices("ipu")[0].target_type == IpuTargetType.IPU


def qr_correction_vector_impl(rcol, sdiag, rcol_idx):
    """Reference implementation of QR correction vector."""
    N = len(rcol)
    mask = (np.arange(0, N) >= rcol_idx).astype(rcol.dtype)
    v = rcol * mask
    norm = np.linalg.norm(v)

    v[rcol_idx] -= norm * sdiag[rcol_idx]
    norm = np.linalg.norm(v)

    scale = 2.0 / (norm * norm)
    return v, scale


@pytest.mark.ipu_hardware
class IpuTileLinalgQR(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    @parameterized.parameters(
        {"N": 4, "expected_offsets": [0, 1, 2, 2, 2, 2, 2], "expected_stride": 1},
        {"N": 16, "expected_offsets": [0, 2, 4, 6, 8, 8, 8], "expected_stride": 1},
        {"N": 36, "expected_offsets": [0, 3, 6, 9, 12, 15, 18], "expected_stride": 1},
        {"N": 128, "expected_offsets": [0, 11, 22, 33, 44, 55, 64], "expected_stride": 1},
    )
    def test__tile_linalg__make_ipu_vector1d_worker_offsets(self, N, expected_offsets, expected_stride):
        vector_size = 2
        num_workers = 6
        woffsets = make_ipu_vector1d_worker_offsets(N, vector_size, num_workers=num_workers, wdtype=np.int16)
        assert woffsets.shape == (num_workers + 1,)
        assert sum(woffsets[1:] - woffsets[:-1]) * vector_size == N
        npt.assert_array_equal(woffsets, expected_offsets)

    # def test__tile_linalg__qr__small_matrix__proper_result(self):
    #     N = 8
    #     tiles = (0,)
    #     # Random symmetric matrix...
    #     x = np.random.randn(N, N).astype(np.float32)
    #     x = (x + x.T) / 2
    #     x = np.expand_dims(x, axis=0)

    #     def qr_decomposition_fn(in0):
    #         input0 = tile_put_sharded(in0, tiles)
    #         return tile_map_primitive(qr_p, input0, full_matrices=True)

    #     qr_decomposition_fn_ipu = jax.jit(qr_decomposition_fn, backend="ipu")
    #     qr_decomposition_fn_cpu = jax.jit(qr_decomposition_fn, backend="cpu")

    #     Q_ipu, R_ipu = qr_decomposition_fn_ipu(x)
    #     Q_cpu, R_cpu = qr_decomposition_fn_cpu(x)

    #     assert isinstance(Q_ipu, TileShardedArray)
    #     assert isinstance(R_ipu, TileShardedArray)
    #     npt.assert_array_almost_equal(np.abs(Q_ipu), np.abs(Q_cpu))
    #     npt.assert_array_almost_equal(np.abs(R_ipu), np.abs(R_cpu))

    @parameterized.parameters(
        {"N": 16, "M": 16},
        {"N": 16, "M": 12},
        {"N": 48, "M": 36},
    )
    def test__qr_householder_row_update_p__proper_result(self, N, M):
        tiles = (0,)
        x = np.random.randn(N).astype(np.float32)
        v = np.random.randn(M).astype(np.float32)
        w = 0.5 + np.random.rand(1).astype(np.float32)

        start_idx = N - M

        def qr_householder_update_fn(x, v, w):
            x = tile_put_replicated(x, tiles)
            v = tile_put_replicated(v, tiles)
            w = tile_put_replicated(w, tiles)
            # Two scaling factors passed.
            return tile_map_primitive(qr_householder_row_update_p, x, v, w, w, start_idx=start_idx)

        qr_householder_update_fn_ipu = jax.jit(qr_householder_update_fn, backend="ipu")
        ret_ipu = qr_householder_update_fn_ipu(x, v, w)

        ret_cpu = x.copy()
        ret_cpu[start_idx:] = x[start_idx:] - w[0] * w[0] * v

        assert isinstance(ret_ipu, TileShardedArray)
        assert ret_ipu.tile_shape == ret_cpu.shape
        npt.assert_array_almost_equal(ret_ipu.array[0, :start_idx], ret_cpu[:start_idx])
        npt.assert_array_almost_equal(ret_ipu.array[0, start_idx:], ret_cpu[start_idx:])

    def test__qr_householder_row_update_p__benchmark_performance(self):
        N = 32 * 8
        M = N
        tiles = (0,)
        x = np.random.randn(N).astype(np.float32)
        v = np.random.randn(M).astype(np.float32)
        w = np.random.randn(1).astype(np.float32)

        def qr_householder_update_fn(x, v, w):
            x = tile_put_replicated(x, tiles)
            v = tile_put_replicated(v, tiles)
            w = tile_put_replicated(w, tiles)
            # Need a first call to force all data transfers to tile.
            x = tile_map_primitive(qr_householder_row_update_p, x, v, w, w, start_idx=N - M)
            x, start = ipu_cycle_count(x)
            x = tile_map_primitive(qr_householder_row_update_p, x, v, w, w, start_idx=N - M)
            # x = tile_map_primitive(scaled_sub_p, x, v, w) # Comparison point.
            x, end = ipu_cycle_count(x)
            return x, start, end

        qr_householder_update_fn_ipu = jax.jit(qr_householder_update_fn, backend="ipu")
        _, start, end = qr_householder_update_fn_ipu(x, v, w)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        qr_correction_cycle_count = end[0] - start[0]
        # Observe on IPU Mk2 hw ~683 cycles.
        assert qr_correction_cycle_count <= 700
        # print("SIZE / CYCLE COUNT: ", N, qr_correction_cycle_count)
        # assert False

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    @parameterized.parameters(
        {"N": 16, "col_idx": 0},
        {"N": 16, "col_idx": 13},
        {"N": 16, "col_idx": 15},
    )
    def test__qr_correction_vector_vertex__proper_result(self, N, col_idx):
        tiles = (0,)
        Rcol = np.random.randn(N).astype(np.float32)
        sdiag = np.random.randn(N).astype(np.float32)

        def qr_correction_vector_fn(Rcol, sdiag):
            Rcol = tile_put_replicated(Rcol, tiles)
            sdiag = tile_put_replicated(sdiag, tiles)
            v_ipu, v_rescale = tile_map_primitive(qr_correction_vector_p, Rcol, sdiag, col_idx=col_idx)  # type:ignore
            # TODO: understand the issue when returning both?
            return (v_ipu,)

        qr_correction_vector_fn_ipu = jax.jit(qr_correction_vector_fn, backend="ipu")
        (v_ipu,) = qr_correction_vector_fn_ipu(Rcol, sdiag)
        v_exp, _ = qr_correction_vector_impl(Rcol, sdiag, col_idx)

        assert isinstance(v_ipu, TileShardedArray)
        npt.assert_array_equal(v_ipu.array[0][:col_idx], 0)
        npt.assert_array_almost_equal(v_ipu.array[0], v_exp)

    def test__qr_correction_vector_vertex__benchmark_performance(self):
        N = 128
        tiles = (0,)
        col_idx = 16
        Rcol = np.random.randn(N).astype(np.float32)

        def qr_correction_vector_fn(Rcol):
            Rcol = tile_put_replicated(Rcol, tiles)
            Rcol, start = ipu_cycle_count(Rcol)
            # FIXME: having to pass the same data to get accurate cycle count.
            r, _ = tile_map_primitive(qr_correction_vector_p, Rcol, Rcol, col_idx=col_idx)  # type:ignore
            r, end = ipu_cycle_count(r)
            return r, start, end

        qr_correction_vector_fn = jax.jit(qr_correction_vector_fn, backend="ipu")
        _, start, end = qr_correction_vector_fn(Rcol)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        qr_correction_cycle_count = end[0] - start[0]
        assert qr_correction_cycle_count <= 950
        # print(qr_correction_cycle_count)
        # assert False

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    @parameterized.parameters(
        {"N": 16},
        {"N": 64},
    )
    def test__linalg_qr_ipu__result_close_to_numpy(self, N):
        # Random symmetric matrix...
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2
        xsdiag = np.sign(np.diag(x)).astype(x.dtype)

        def qr_decomposition_fn(x, xsdiag):
            return ipu_qr(x, xsdiag)

        qr_decomposition_fn_ipu = jax.jit(qr_decomposition_fn, backend="ipu")
        Q, RT = qr_decomposition_fn_ipu(x, xsdiag)
        # NumPy as reference point!
        Qexp, Rexp = np.linalg.qr(x)

        npt.assert_array_almost_equal(np.abs(Q.array), np.abs(Qexp), decimal=5)
        npt.assert_array_almost_equal(np.abs(RT.array), np.abs(Rexp.T), decimal=5)

    @unittest.skipUnless(ipu_hw_available, "Requires IPU hardware")
    def test__linalg_qr_ipu__benchmark(self):
        N = 32
        # Random symmetric matrix...
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2
        xsdiag = np.sign(np.diag(x)).astype(x.dtype)

        def qr_decomposition_fn(x, xsdiag):
            # Shard inputs, and wait!
            Q, RT, sdiag = ipu_qr_shard_inputs(x, xsdiag)
            Q, RT, sdiag = tile_data_barrier(Q, RT, sdiag)
            # Benchmark QR main iterations.
            Q, start = ipu_cycle_count(Q)
            Q, RT = ipu_qr_iterations(Q, RT, sdiag)
            Q, end = ipu_cycle_count(Q)
            return Q, RT, start, end

        qr_decomposition_fn_ipu = jax.jit(qr_decomposition_fn, backend="ipu")
        _, _, start, end = qr_decomposition_fn_ipu(x, xsdiag)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        qr_cycle_count = end[0] - start[0]
        # TODO: understand why it used to be 60k cycles?
        # assert qr_cycle_count <= 60000
        assert qr_cycle_count <= 75000

        # ipu_frequency = 1.8 * 1e9
        # timing = qr_cycle_count / ipu_frequency
        # print("CYCLE COUNT, TIMING:", qr_cycle_count, timing * 1000)
        # assert False

    @parameterized.parameters({"dtype": np.float32})
    def test__dot_product1d__proper_result(self, dtype):
        IPU_NUM_THREADS = 6
        IPU_SIMD_WIDTH = 8 // dtype(0).nbytes
        WORKER_SIZE = IPU_NUM_THREADS * IPU_SIMD_WIDTH
        # Array size doesn't need to be an exact multiple of WORKER_SIZE
        # but making it so to simplify numpy comparison
        N = 16 * WORKER_SIZE
        # N = 128 * 6
        tiles = (0, 1, 2)
        T = len(tiles)
        x = np.random.randn(T, N).astype(dtype)
        y = np.random.randn(T, N).astype(dtype)

        def dot_product1d_fn(x, y):
            # Shard inputs, and wait!
            x = tile_put_sharded(x, tiles)
            y = tile_put_sharded(y, tiles)
            x, y = tile_data_barrier(x, y)
            r = tile_map_primitive(dot_product1d_p, x, y)
            return r

        dot_product1d_fn = jax.jit(dot_product1d_fn, backend="ipu")
        res = dot_product1d_fn(x, y)
        res = np.asarray(res)
        assert res.shape == (T, WORKER_SIZE)
        np_res = [np.dot(x[i], y[i]) for i in range(T)]
        ipu_res = np.sum(res, 1)
        # NOTE: accumulation accuracy changing a bit depending on the strategy.
        npt.assert_array_almost_equal(ipu_res, np_res, decimal=5)

    def test__dot_product1d__benchmark(self):
        N = 512
        # N = 128 * 6
        tiles = (0,)
        x = np.random.randn(N).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)

        def dot_product1d_fn(x, y):
            # Shard inputs, and wait!
            x = tile_put_replicated(x, tiles)
            y = tile_put_replicated(y, tiles)
            x, y = tile_data_barrier(x, y)
            # Benchmark dot product1d vertex.
            x, start = ipu_cycle_count(x)
            r = tile_map_primitive(dot_product1d_p, x, y)
            # r = tile_map_primitive(reduce_sum_p, r, axes=(0,)) # Optional final reduction.
            r, end = ipu_cycle_count(r)  # type:ignore
            return r, start, end

        dot_product1d_fn = jax.jit(dot_product1d_fn, backend="ipu")
        res, start, end = dot_product1d_fn(x, y)
        res = np.asarray(res)
        # NOTE: accumulation accuracy changing a bit depending of the strategy.
        assert res.shape == (1, 12)
        npt.assert_array_almost_equal(np.sum(res), np.dot(x, y), decimal=5)

        start, end = np.asarray(start)[0], np.asarray(end)[0]
        cycle_count = end[0] - start[0]
        assert cycle_count <= 800

        # ipu_frequency = 1.8 * 1e9
        # timing = cycle_count / ipu_frequency
        # print("CYCLE COUNT:", cycle_count, " / TIMING (ms):", timing * 1000)
        # assert False
