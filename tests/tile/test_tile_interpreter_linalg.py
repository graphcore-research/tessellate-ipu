# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax.lax.linalg import qr_p

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_sharded


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
