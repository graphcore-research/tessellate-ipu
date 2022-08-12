import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest

from jax_ipu_research.tile import TileShardedArray, tile_put_replicated, tile_put_sharded


class TileShardedArrayTests(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test__tile_sharded_array__shape_dtype(self):
        @self.variant
        def tile_put_sharded_fn(arr) -> TileShardedArray:
            return tile_put_sharded(arr, (3, 4, 5))

        input = np.asarray([1, 2, 3], np.float32)
        output = tile_put_sharded_fn(input)
        assert output.aval.dtype == input.dtype
        assert output.aval.shape == input.shape

    @chex.variants(with_jit=True, without_jit=True)
    def test__tile_sharded_array__getitem__slicing(self):
        @self.variant
        def tile_array_slicing(arr) -> TileShardedArray:
            arr = tile_put_sharded(arr, (3, 4, 5))
            return arr[1:3]

        input = np.asarray([1, 2, 3], np.float32)
        output = tile_array_slicing(input)
        npt.assert_array_equal(output.array, input[1:3])
        assert output.tiles == (4, 5)


@pytest.mark.parametrize("backend", ["ipu"])
def test__tile_put_sharded__ipu_jitting(backend):
    input = np.asarray([1, 2, 3], np.float32)
    tiles = (3, 4, 5)
    output = jax.jit(tile_put_sharded, static_argnums=1, backend=backend)(input, tiles)

    assert isinstance(output, TileShardedArray)
    assert output.tiles == tiles
    assert output.aval.shape == input.shape
    npt.assert_array_equal(output.array, input)


@pytest.mark.parametrize("backend", ["ipu"])
def test__tile_put_replicated__ipu_jitting(backend):
    input = np.asarray([1, 2, 3], np.float32)
    tiles = (3, 4, 5)
    output = jax.jit(tile_put_replicated, static_argnums=1, backend=backend)(input, tiles)

    assert isinstance(output, TileShardedArray)
    assert output.tiles == tiles
    assert output.aval.shape == (len(tiles), *input.shape)
    npt.assert_array_equal(output.array, np.stack([input for _ in range(len(tiles))]))
