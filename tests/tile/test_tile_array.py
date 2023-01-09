from functools import partial
from typing import Tuple

import chex
import jax
import numpy as np
import numpy.testing as npt
import pytest
from absl.testing import parameterized

from jax_ipu_research.tile import TileShardedArray, tile_data_barrier, tile_put_replicated, tile_put_sharded
from jax_ipu_research.tile.tile_array import check_tile_array_multi_slice


class TileShardedArrayTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test__tile_sharded_array__shape_dtype(self):
        @self.variant
        def tile_put_sharded_fn(arr) -> TileShardedArray:
            return tile_put_sharded(arr, (3, 4, 5))

        input = np.asarray([1, 2, 3], np.float32)
        output = tile_put_sharded_fn(input)
        assert output.aval.dtype == input.dtype
        assert output.aval.shape == input.shape

    @parameterized.parameters(
        [
            {"keys": (0, 0), "shape": (1, 1)},
            {"keys": (slice(None), 1), "shape": (3, 10)},
            {"keys": (slice(1, 3, 2), 5, 1), "shape": (3, 7, 10)},
            {"keys": (slice(1, 3, 2), 3, slice(3, 8)), "shape": (3, 7, 10)},
            {"keys": (slice(None), slice(0, 2), slice(None)), "shape": (3, 7, 10)},
            {"keys": (slice(None), 3, slice(0, 2), slice(None)), "shape": (3, 5, 7, 10)},
        ]
    )
    def test__check_tile_array_multi_slice__valid_slices(self, keys, shape):
        assert check_tile_array_multi_slice(keys, shape)

    @parameterized.parameters(
        [
            {"keys": (0, 1, 2), "shape": (3, 10)},
            {"keys": (0, slice(0, None, 2)), "shape": (3, 10)},  # strided slicing
            {"keys": (0, slice(0, 2), slice(1, 3)), "shape": (3, 7, 10)},
        ]
    )
    def test__check_tile_array_multi_slice__invalid_slices(self, keys, shape):
        with self.assertRaises(ValueError):
            check_tile_array_multi_slice(keys, shape)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters([1, slice(1, 3), slice(None, None), slice(0, None, 2)])
    def test__tile_sharded_array__getitem__tile_axis_slicing(self, key):
        tiles = (3, 4, 5)
        data = np.random.randn(len(tiles), 7, 11)

        @self.variant
        def tile_array_slicing(arr) -> TileShardedArray:
            arr = tile_put_sharded(arr, tiles)
            return arr[key]

        outdata = tile_array_slicing(data)
        # Always keeping the first axis corresponding to tile sharding.
        npykey = slice(key, key + 1) if isinstance(key, int) else key
        assert outdata.tiles == tiles[npykey]
        npt.assert_array_almost_equal(outdata.array, data[npykey])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        [
            {"keys": (slice(None), 1), "tile_shape": (11,)},
            {"keys": (slice(1, 3, 2), slice(1, 3)), "tile_shape": (11,)},
            {"keys": (slice(1, 3, 2), 4, slice(1, 3)), "tile_shape": (5, 11)},
        ]
    )
    def test__tile_sharded_array__getitem__tile_data_slicing(self, keys, tile_shape):
        tiles = (3, 4, 5)
        data = np.random.randn(len(tiles), *tile_shape)

        @self.variant
        def tile_array_slicing(arr) -> TileShardedArray:
            arr = tile_put_sharded(arr, tiles)
            return arr[keys]

        outdata = tile_array_slicing(data)
        assert outdata.tiles == tiles[keys[0]]
        npt.assert_array_almost_equal(outdata.array, data[keys])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        [
            {"shape": (3, 4, 5), "reshape": (3, 20)},
            {"shape": (3, 4, 5), "reshape": (-1, 20)},
            {"shape": (3, 4, 5), "reshape": (-1, 1, 20)},
        ]
    )
    def test__tile_sharded_array__reshape__tile_data_reshaping(self, shape, reshape):
        tiles = (3, 4, 5)
        data = np.random.randn(*shape)

        @self.variant
        def tile_array_reshaping(arr) -> TileShardedArray:
            arr = tile_put_sharded(arr, tiles)
            return arr.reshape(reshape)

        outdata = tile_array_reshaping(data)
        assert outdata.tiles == tiles
        npt.assert_array_almost_equal(outdata.array, data.reshape(reshape))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        [
            {"shape": (3, 1, 5, 1), "exp_shape": (3, 5)},
            {"shape": (1, 1, 5, 1), "exp_shape": (1, 5)},
        ]
    )
    def test__tile_sharded_array__squeeze__proper_shape(self, shape, exp_shape):
        data = np.random.randn(*shape)
        tiles = tuple(range(shape[0]))

        @self.variant
        def tile_array_squeezing(arr) -> TileShardedArray:
            arr = tile_put_sharded(arr, tiles)
            return arr.squeeze()

        outdata = tile_array_squeezing(data)
        assert outdata.tiles == tiles
        assert outdata.shape == exp_shape


@pytest.mark.parametrize("backend", ["ipu"])
def test__tile_put_sharded__ipu_jitting(backend):
    # TODO: more complex 3 x 64 x 64 array examples.
    input = np.asarray([1, 2, 3], np.float32)
    tiles = (3, 4, 5)
    output = jax.jit(tile_put_sharded, static_argnums=1, backend=backend)(input, tiles)

    assert isinstance(output, TileShardedArray)
    assert output.tiles == tiles
    assert output.aval.shape == input.shape
    assert output.tile_aval.shape == ()
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


class TileDataBarrierTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test__tile_data_barrier__jitting_test(self):
        # Set of random tiles mapping.
        inputs_tiles = [[0, 1], [2, 3]]

        @self.variant
        def tile_data_barrier_fn(data) -> Tuple[TileShardedArray, ...]:
            inputs = [tile_put_replicated(data, tiles) for tiles in inputs_tiles]
            outputs = tile_data_barrier(*inputs)
            return outputs

        data = np.asarray([1, 2, 5], np.float32)
        tile_data_barrier_fn(data)

    def test__tile_data_barrier__single_input__noop(self):
        tiles = [0, 1]
        data = np.asarray([1, 2, 5], np.float32)
        t0 = tile_put_replicated(data, tiles)
        t1 = tile_data_barrier(t0)
        assert t1 is t0

    def test__tile_data_barrier__not_supporting_different_size_dtypes(self):
        tiles = [0, 1]
        data = np.asarray([1, 2, 5], np.float32)

        @partial(jax.jit, backend="ipu")
        def tile_data_barrier_fn(data) -> Tuple[TileShardedArray, ...]:
            t0 = tile_put_replicated(data, tiles)
            t1 = tile_put_replicated(data.astype(np.float16), tiles)
            return tile_data_barrier(t0, t1)

        with self.assertRaises(TypeError):
            tile_data_barrier_fn(data)

    def test__tile_data_barrier__supporting_same_size_dtypes(self):
        tiles = [0, 1]
        data = np.asarray([1, 2, 5], np.float32)

        @partial(jax.jit, backend="ipu")
        def tile_data_barrier_fn(data) -> Tuple[TileShardedArray, ...]:
            t0 = tile_put_replicated(data, tiles)
            t1 = tile_put_replicated(data.astype(np.uint32), tiles)
            t2 = tile_put_replicated(data.astype(np.int32), tiles)
            return tile_data_barrier(t0, t1, t2)

        tile_data_barrier_fn(data)

    @parameterized.parameters([np.float16, np.float32, np.int32])
    def test__tile_data_barrier__dtypes__ipu_jitting(self, dtype):
        # Set of random tiles mapping.
        inputs_tiles = [[0, 1], [2, 3], [0, 1], [1, 4, 5]]

        @partial(jax.jit, backend="ipu")
        def tile_data_barrier_fn(data) -> Tuple[TileShardedArray, ...]:
            inputs = [tile_put_replicated(data, tiles) for tiles in inputs_tiles]
            outputs = tile_data_barrier(*inputs)
            return outputs

        data = np.asarray([1, 2, 5], dtype)
        outputs = tile_data_barrier_fn(data)

        assert len(outputs) == len(inputs_tiles)
        outputs = [np.asarray(v) for v in outputs]
        for idx in range(len(inputs_tiles)):
            npt.assert_array_equal(outputs[idx][0], data)
