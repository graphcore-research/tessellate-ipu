# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from tessellate_ipu import tile_map, tile_put_replicated
from tessellate_ipu.lax import cummax_p, cummin_p, cumprod_p, cumsum_p


class IpuTilePrimitivesLaxCumsumTests(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        # Not very clean, but for better reproducibility.
        np.random.seed(123)

    @parameterized.parameters(
        {"N": 16, "dtype": np.float32, "cumop": cumsum_p},
        {"N": 16, "dtype": np.int32, "cumop": cumsum_p},
        {"N": 16, "dtype": np.float32, "cumop": cummax_p},
        {"N": 16, "dtype": np.int32, "cumop": cummax_p},
        {"N": 16, "dtype": np.float32, "cumop": cummin_p},
        {"N": 16, "dtype": np.int32, "cumop": cummin_p},
        {"N": 16, "dtype": np.float32, "cumop": cumprod_p},
        {"N": 16, "dtype": np.int32, "cumop": cumprod_p},
    )
    def test__tile_map__cumulative_op__jitting__proper_result(self, N, dtype, cumop):
        tiles = (0,)
        data = (np.random.randn(N)).astype(dtype)

        def compute_fn(data):
            data = tile_put_replicated(data, tiles)
            return tile_map(cumop, data, axis=0, reverse=False)

        cpu_compute_fn = partial(jax.jit, backend="cpu")(compute_fn)
        ipu_compute_fn = partial(jax.jit, backend="ipu")(compute_fn)

        cpu_output = cpu_compute_fn(data)
        ipu_output = ipu_compute_fn(data)
        assert ipu_output.tiles == tiles
        assert ipu_output.dtype == data.dtype
        npt.assert_array_almost_equal(ipu_output, cpu_output, decimal=5)
