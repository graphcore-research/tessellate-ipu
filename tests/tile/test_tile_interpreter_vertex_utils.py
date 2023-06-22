# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from tessellate_ipu.tile.tile_interpreter_vertex_utils import (
    make_ipu_vector1d_worker_offsets,
    make_num_elements_per_worker,
)


class IpuTileVertexUtils(chex.TestCase, parameterized.TestCase):
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
    def test__tile_vertex_utils__make_ipu_vector1d_worker_offsets(self, N, expected_offsets, expected_stride):
        vector_size = 2
        num_workers = 6
        woffsets = make_ipu_vector1d_worker_offsets(N, vector_size, num_workers=num_workers, wdtype=np.int16)
        assert woffsets.shape == (num_workers + 1,)
        assert sum(woffsets[1:] - woffsets[:-1]) * vector_size == N
        npt.assert_array_equal(woffsets, expected_offsets)

    @parameterized.parameters(
        {"N": 0, "expected_num_elements": [0, 0, 0, 0, 0, 0]},
        {"N": 1, "expected_num_elements": [1, 0, 0, 0, 0, 0]},
        {"N": 5, "expected_num_elements": [1, 1, 1, 1, 1, 0]},
        {"N": 6, "expected_num_elements": [1, 1, 1, 1, 1, 1]},
        {"N": 16, "expected_num_elements": [3, 3, 3, 3, 2, 2]},
        {"N": 36, "expected_num_elements": [6, 6, 6, 6, 6, 6]},
        {"N": 128, "expected_num_elements": [22, 22, 21, 21, 21, 21]},
    )
    def test__tile_vertex_utils__make_num_elements_per_worker(self, N, expected_num_elements):
        num_workers = 6
        num_elements = make_num_elements_per_worker(N, num_workers)
        assert np.sum(num_elements) == N
        npt.assert_array_equal(num_elements, expected_num_elements)
