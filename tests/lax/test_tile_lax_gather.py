# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized

from tessellate_ipu import tile_map, tile_put_replicated
from tessellate_ipu.lax import gather_p


@pytest.mark.ipu_hardware
class IpuTilePrimitivesLaxGatherHwTests(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        # Not very clean, but for better reproducibility.
        np.random.seed(123)

    @parameterized.parameters(
        {"data_shape": (8,), "num_indices": 3},
        {"data_shape": (8,), "num_indices": 12},
        {"data_shape": (256,), "num_indices": 512},
        {"data_shape": (256, 17), "num_indices": 123},
        {"data_shape": (256, 5, 3), "num_indices": 373},
    )
    def test__tile_map__gather__first_axis_cases__jitting__proper_result(self, data_shape, num_indices):
        tiles = (0,)
        data = np.random.randn(*data_shape).astype(np.float32)
        indices = np.random.randint(low=0, high=data_shape[0], size=num_indices)
        indices = indices.reshape(-1, 1).astype(np.uint32)

        # First axis gather only supported configuration!
        dim_numbers = jax.lax.GatherDimensionNumbers(
            offset_dims=tuple(range(1, len(data_shape))), collapsed_slice_dims=(0,), start_index_map=(0,)
        )

        def gather_fn(data, indices):
            data = tile_put_replicated(data, tiles)
            indices = tile_put_replicated(indices, tiles)
            return tile_map(
                gather_p,
                data,
                indices,
                dimension_numbers=dim_numbers,
                slice_sizes=(1, *data_shape[1:]),
                mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                unique_indices=False,
                indices_are_sorted=False,
                fill_value=None,
            )

        cpu_gather_fn = partial(jax.jit, backend="cpu")(gather_fn)
        ipu_gather_fn = partial(jax.jit, backend="ipu")(gather_fn)

        cpu_output = cpu_gather_fn(data, indices)
        ipu_output = ipu_gather_fn(data, indices)

        assert ipu_output.tiles == tiles
        assert ipu_output.dtype == data.dtype
        npt.assert_array_equal(ipu_output, cpu_output)
