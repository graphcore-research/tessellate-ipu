from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_ipu_research import is_ipu_model
from jax_ipu_research.tile import (
    TileShardedArray,
    ipu_get_hw_seeds_tmap,
    ipu_random_normal_tmap,
    ipu_random_uniform_tmap,
    ipu_set_hw_seeds_tmap,
    tile_put_sharded,
)


class IpuTilePrimitivesRandomSeeds(chex.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        self.num_worker_contexts = self.device.num_worker_contexts

    def test__ipu_get_hw_seeds_tmap__proper_seed_array(self):
        tiles = (1, 2, self.num_tiles - 1)

        @partial(jax.jit, backend="ipu")
        def compute_fn():
            return ipu_get_hw_seeds_tmap(tiles)

        ipu_seeds = compute_fn()
        assert isinstance(ipu_seeds, TileShardedArray)
        assert ipu_seeds.shape == (len(tiles), self.num_worker_contexts, 4)
        assert ipu_seeds.dtype == np.uint32

    def test__ipu_set_get_hw_seeds_tmap__proper_round_trip(self):
        tiles = (1, 2, self.num_tiles - 1)
        ipu_seeds_in = np.random.randint(0, 256, size=(len(tiles), self.num_worker_contexts, 4)).astype(np.uint32)

        @partial(jax.jit, backend="ipu")
        def compute_fn(seeds):
            ipu_seeds_in = tile_put_sharded(seeds, tiles)
            ipu_seeds_in = ipu_set_hw_seeds_tmap(ipu_seeds_in)
            ipu_seeds_out = ipu_get_hw_seeds_tmap(tiles)
            return ipu_seeds_in, ipu_seeds_out

        ipu_seeds_in, ipu_seeds_out = compute_fn(ipu_seeds_in)

        assert ipu_seeds_out.dtype == ipu_seeds_in.dtype
        assert ipu_seeds_out.shape == ipu_seeds_in.shape
        # Random only properly implemented on IPU hardware
        if not is_ipu_model(self.device):
            npt.assert_array_equal(ipu_seeds_out.array, ipu_seeds_in.array)


class IpuTilePrimitivesRandomUniform(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles
        self.num_worker_contexts = self.device.num_worker_contexts

    @parameterized.parameters([np.float32, np.float16])
    def test__ipu_random_uniform_tmap__float__proper_random_array(self, dtype):
        tiles = (1, 2, 3)
        size = 1000

        @partial(jax.jit, backend="ipu")
        def compute_fn():
            return ipu_random_uniform_tmap(tiles, size=size, dtype=dtype, offset=1.0, scale=2.0)

        ipu_uniform_array = compute_fn()
        assert isinstance(ipu_uniform_array, TileShardedArray)
        assert ipu_uniform_array.shape == (len(tiles), size)
        assert ipu_uniform_array.dtype == dtype

        assert np.min(ipu_uniform_array) >= 0.0
        assert np.max(ipu_uniform_array) <= 2.0

    @parameterized.parameters([np.int32])
    def test__ipu_random_uniform_tmap__int__proper_random_array(self, dtype):
        tiles = (1, 2, 3)
        size = 1000
        offset = 10
        scale = 256

        @partial(jax.jit, backend="ipu")
        def compute_fn():
            return ipu_random_uniform_tmap(tiles, size=size, dtype=dtype, offset=offset, scale=scale)

        ipu_uniform_array = compute_fn()
        assert isinstance(ipu_uniform_array, TileShardedArray)
        assert ipu_uniform_array.shape == (len(tiles), size)
        assert ipu_uniform_array.dtype == dtype

        # TODO: understand why the range is not proper??? IPU model issue?
        if not is_ipu_model(self.device):
            assert np.min(ipu_uniform_array) >= offset
            assert np.max(ipu_uniform_array) < scale + offset


class IpuTilePrimitivesRandomNormal(chex.TestCase, parameterized.TestCase):
    @parameterized.parameters([np.float32, np.float16])
    def test__ipu_random_normal_tmap__float__proper_random_array(self, dtype):
        tiles = (1, 2, 3)
        size = 10000
        mean = 1.0
        stddev = 2.0

        @partial(jax.jit, backend="ipu")
        def compute_fn():
            return ipu_random_normal_tmap(tiles, size=size, dtype=dtype, mean=mean, stddev=stddev)

        ipu_normal_array = compute_fn()
        assert isinstance(ipu_normal_array, TileShardedArray)
        assert ipu_normal_array.shape == (len(tiles), size)
        assert ipu_normal_array.dtype == dtype

        npt.assert_almost_equal(np.mean(ipu_normal_array), mean, decimal=2)
        npt.assert_almost_equal(np.std(np.asarray(ipu_normal_array, dtype=np.float32) - mean), stddev, decimal=1)
