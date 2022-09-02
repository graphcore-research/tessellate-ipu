import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized

from jax_ipu_research.tile.tile_array_primitives import tile_put_replicated_prim, tile_put_sharded_prim


class TilePutShardedPrimTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test__tile_put_sharded_prim__invalid_number_tiles(self):
        @self.variant
        def tile_put_sharded_fn(x):
            return tile_put_sharded_prim(x, (3, 4, 5, 6))

        input = np.asarray([1, 2, 3], np.float32)
        with pytest.raises(AssertionError):
            tile_put_sharded_fn(input)

    def test__tile_put_sharded_prim__no_jitting(self):
        input = np.asarray([1, 2, 3], np.float32)
        tiles = (3, 4, 5)
        output = tile_put_sharded_prim(input, tiles)
        assert output is input

    @parameterized.parameters(["cpu", "ipu"])
    def test__tile_put_sharded_prim__device_jitting(self, backend):
        input = np.asarray([1, 2, 3], np.float32)
        tiles = (3, 4, 5)
        output = jax.jit(tile_put_sharded_prim, static_argnums=1, backend=backend)(input, tiles)
        assert output.shape == input.shape
        npt.assert_array_equal(output, input)


def test__tile_put_replicated_prim__no_jitting():
    input = np.asarray([[1, 2]], np.float32)
    tiles = (3, 4, 5)
    output = tile_put_replicated_prim(input, tiles)
    assert output.shape == (len(tiles), *input.shape)
    npt.assert_array_equal(output, np.stack([input for _ in range(len(tiles))]))


@pytest.mark.parametrize("backend", ["ipu"])
def test__tile_put_replicated_prim__device_jitting(backend):
    input = np.asarray([[1, 2]], np.float32)
    tiles = (3, 4, 5)
    output = jax.jit(tile_put_replicated_prim, static_argnums=1, backend=backend)(input, tiles)
    assert output.shape == (len(tiles), *input.shape)
    npt.assert_array_equal(output, tile_put_replicated_prim(input, tiles))
