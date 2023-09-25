# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from tessellate_ipu import tile_map, tile_put_replicated
from tessellate_ipu.lax import scatter_add_p, scatter_max_p, scatter_mul_p, scatter_p


class IpuTilePrimitivesLaxScater(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        # Not very clean, but for better reproducibility.
        np.random.seed(123)

    @parameterized.parameters(
        {"num_elements": 8, "num_indices": 3, "scatter_prim": scatter_p},
        {"num_elements": 8, "num_indices": 16, "scatter_prim": scatter_add_p},
        {"num_elements": 8, "num_indices": 16, "scatter_prim": scatter_max_p},
        {"num_elements": 8, "num_indices": 16, "scatter_prim": scatter_mul_p},
        {"num_elements": 8, "num_indices": 3, "scatter_prim": scatter_add_p},
        {"num_elements": 8, "num_indices": 12, "scatter_prim": scatter_add_p},
        {"num_elements": 256, "num_indices": 512, "scatter_prim": scatter_add_p},
    )
    def test__tile_map__scatter__jitting__multi_sizes__proper_result(self, num_elements, num_indices, scatter_prim):
        tiles = (0,)
        data = np.random.randn(num_elements).astype(np.float32)
        indices = np.random.randint(low=0, high=num_elements, size=num_indices)
        indices = indices.reshape(-1, 1).astype(np.uint32)
        updates = np.random.randn(indices.size).astype(np.float32)

        # Only supported configuration!
        scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
        )

        def scatter_add_fn(data, indices, updates):
            data = tile_put_replicated(data, tiles)
            indices = tile_put_replicated(indices, tiles)
            updates = tile_put_replicated(updates, tiles)
            return tile_map(
                scatter_prim,
                data,
                indices,
                updates,
                dimension_numbers=scatter_dnums,
                indices_are_sorted=False,
                unique_indices=False,
                mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                update_jaxpr=None,
                update_consts=None,
            )

        cpu_scatter_add_fn = partial(jax.jit, backend="cpu")(scatter_add_fn)
        ipu_scatter_add_fn = partial(jax.jit, backend="ipu")(scatter_add_fn)

        cpu_output = cpu_scatter_add_fn(data, indices, updates)
        ipu_output = ipu_scatter_add_fn(data, indices, updates)

        assert ipu_output.tiles == tiles
        assert ipu_output.dtype == data.dtype
        npt.assert_array_almost_equal(ipu_output, cpu_output)
