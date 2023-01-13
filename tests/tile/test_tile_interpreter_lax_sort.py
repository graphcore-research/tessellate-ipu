# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_ipu_research.tile.tile_interpreter_lax_sort import ipu_argsort_quadratic_unique


class IpuTileLinalgJacobi(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        np.random.seed(42)

    def test__ipu_argsort_quadratic_unique__proper_result(self):
        N = 1024
        values = np.random.rand(N).astype(np.float32)

        ipu_argsort_quadratic_fn = jax.jit(ipu_argsort_quadratic_unique, backend="ipu")
        indices = ipu_argsort_quadratic_fn(values)
        expected_indices = np.argsort(values)

        npt.assert_array_equal(indices, expected_indices)
