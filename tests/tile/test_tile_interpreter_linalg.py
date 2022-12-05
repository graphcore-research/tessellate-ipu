# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax.lax.linalg import qr_p

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_replicated, tile_put_sharded
from jax_ipu_research.tile.tile_interpreter_linalg import ipu_qr, qr_correction_vector_p, qr_householder_update_p


class IpuTileLinalgQR(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    def test__tile_linalg__qr__small_matrix__proper_result(self):
        N = 8
        tiles = (0,)
        # Random symmetric matrix...
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2
        x = np.expand_dims(x, axis=0)

        def qr_decomposition_fn(in0):
            input0 = tile_put_sharded(in0, tiles)
            return tile_map_primitive(qr_p, input0, full_matrices=True)

        qr_decomposition_fn_ipu = jax.jit(qr_decomposition_fn, backend="ipu")
        qr_decomposition_fn_cpu = jax.jit(qr_decomposition_fn, backend="cpu")

        Q_ipu, R_ipu = qr_decomposition_fn_ipu(x)
        Q_cpu, R_cpu = qr_decomposition_fn_cpu(x)

        assert isinstance(Q_ipu, TileShardedArray)
        assert isinstance(R_ipu, TileShardedArray)
        npt.assert_array_almost_equal(np.abs(Q_ipu), np.abs(Q_cpu))
        npt.assert_array_almost_equal(np.abs(R_ipu), np.abs(R_cpu))

    def test__qr_householder_update__proper_result(self):
        N = 8
        tiles = (0,)
        x = np.random.randn(N, N).astype(np.float32)
        v = np.random.randn(
            N,
        ).astype(np.float32)
        w = np.random.randn(
            N,
        ).astype(np.float32)

        def qr_householder_update_fn(x, v, w):
            x = tile_put_replicated(x, tiles)
            v = tile_put_replicated(v, tiles)
            w = tile_put_replicated(w, tiles)
            return tile_map_primitive(qr_householder_update_p, x, v, w)

        qr_householder_update_fn_ipu = jax.jit(qr_householder_update_fn, backend="ipu")
        x_ipu = qr_householder_update_fn_ipu(x, v, w)

        assert isinstance(x_ipu, TileShardedArray)
        npt.assert_array_almost_equal(x_ipu.array[0], x - np.outer(w, v))

    def test__qr_correction_vector__proper_result(self):
        N = 8
        tiles = (0,)
        col_idx = 4
        Rcol = np.random.randn(N).astype(np.float32)
        sdiag = np.random.randn(N).astype(np.float32)

        def qr_correction_vector_fn(Rcol, sdiag):
            Rcol = tile_put_replicated(Rcol, tiles)
            sdiag = tile_put_replicated(sdiag, tiles)
            return tile_map_primitive(qr_correction_vector_p, Rcol, sdiag, col_idx=col_idx)

        qr_correction_vector_fn_ipu = jax.jit(qr_correction_vector_fn, backend="ipu")
        v_ipu = qr_correction_vector_fn_ipu(Rcol, sdiag)

        assert isinstance(v_ipu, TileShardedArray)
        npt.assert_array_equal(v_ipu.array[0][:col_idx], 0)
        npt.assert_almost_equal(np.linalg.norm(v_ipu.array[0]), 1.0 * np.sqrt(2))
        # TODO: additional testing?

    def test__linalg_qr_ipu__result_close_to_numpy(self):
        N = 32
        # Random symmetric matrix...
        x = np.random.randn(N, N).astype(np.float32)
        x = (x + x.T) / 2
        xsdiag = np.sign(np.diag(x)).astype(x.dtype)

        def qr_decomposition_fn(x, xsdiag):
            return ipu_qr(x, xsdiag)

        qr_decomposition_fn_ipu = jax.jit(qr_decomposition_fn, backend="ipu")
        Q, RT = qr_decomposition_fn_ipu(x, xsdiag)
        # Numpy as reference point!
        Qexp, Rexp = np.linalg.qr(x)

        npt.assert_array_almost_equal(np.abs(Q.array), np.abs(Qexp), decimal=5)
        npt.assert_array_almost_equal(np.abs(RT.array), np.abs(Rexp.T), decimal=5)
