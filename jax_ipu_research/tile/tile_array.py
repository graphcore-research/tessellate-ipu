# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Sequence, Tuple, Union

import chex
import numpy as np
from jax.interpreters.xla import DeviceArray, ShapedArray

from jax_ipu_research.utils import DTypeLike, Shape

from .tile_array_primitives import (
    tile_data_barrier_prim,
    tile_gather_prim,
    tile_put_replicated_prim,
    tile_put_sharded_prim,
)

SliceType = Union[int, slice]
MultiSliceType = Tuple[SliceType, ...]


def check_tile_array_multi_slice(slices: MultiSliceType, shape: Shape) -> bool:
    """Check if a tile array multi-slice is valid, i.e.
    it will keep memort contiguity of the underlying IPU array.

    Args:
        slices: Tuple of slices.
        shape: (full) Shape of the array to slice.
    """
    assert isinstance(slices, tuple)
    # TODO: support `newaxis`
    if len(slices) > len(shape):
        raise ValueError(f"Unsupported slicing `{slices}` on IPU tile array of shape `{shape}`.")
    if len(slices) < len(shape):
        # Complete with full slices.
        full_slices = [slice(None)] * (len(shape) - len(slices))
        slices = (*slices, *full_slices)

    # Check there is no strided slice.
    for s in slices[1:]:
        if isinstance(s, slice) and s.step not in {None, 1}:
            raise ValueError(f"Unsupported strided slicing `{slices}` on IPU tile array of shape `{shape}`.")

    # Last axis with non trivial stride
    non_trivial_slice_axes = [idx for idx in range(len(shape)) if (shape[idx] > 1 and slices[idx] != slice(None))]
    last_non_trivial_slice_axis = max(non_trivial_slice_axes) if len(non_trivial_slice_axes) > 0 else 0

    # Check only axis slicing in-between.
    for idx in range(1, last_non_trivial_slice_axis):
        s = slices[idx]
        valid_slice = isinstance(s, int)
        if not valid_slice:
            raise ValueError(f"Unsupported slicing `{slices}` on IPU tile array of shape `{shape}`.")

    # Should be good!
    return True


@chex.dataclass(frozen=True, mappable_dataclass=False)
class TileShardedArray:
    """JAX array sharded over (IPU) tiles.

    An IPU tile sharded array should satisfy the following assumptions:
        - Sharded over the first axis on a given collection of tiles;
        - Each shard is contiguous in memory on every tile;

    On CPU, GPU, devices ... a tile sharded array will just be a normal array, with
    no particular assumption on memory layout.

    The constructor is assuming already proper tile mapping. Please use `tile_put_sharded`
    and `tile_put_replicated` to build a proper tile sharded array.

    Args:
        array: Underlying JAX array.
        tiles: List of tiles on which the array is sharded.
    """

    array: chex.ArrayDevice
    tiles: Tuple[int, ...]

    def __post_init__(self):
        # Check consistent array and collection of tiles.
        if len(self.tiles) != self.array.shape[0]:
            raise ValueError(
                f"Inconsistent IPU sharded array shape '{self.array.shape}' and number of tiles {len(self.tiles)}."
            )

    @property
    def dtype(self) -> Any:
        return self.array.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def size(self) -> int:
        return self.array.size

    @property
    def aval(self) -> ShapedArray:
        if isinstance(self.array, np.ndarray):
            return ShapedArray(self.array.shape, self.array.dtype)
        return self.array.aval

    @property
    def num_tiles(self) -> int:
        return len(self.tiles)

    @property
    def tile_aval(self) -> ShapedArray:
        """Abstract val, at the tile level."""
        aval = self.aval
        return ShapedArray(aval.shape[1:], aval.dtype)

    @property
    def tile_shape(self) -> Shape:
        return self.tile_aval.shape

    @property
    def device_array(self) -> DeviceArray:
        return self.array

    def reshape(self, shape: Shape) -> "TileShardedArray":
        d0 = shape[0]
        if d0 != -1 and d0 != self.num_tiles:
            raise ValueError(f"Can not reshape '{shape}' the tile sharding axis in a TileShardedArray.")
        shape = (self.num_tiles, *shape[1:])
        return TileShardedArray(array=self.array.reshape(shape), tiles=self.tiles)  # type:ignore

    def tile_reshape(self, shape: Shape) -> "TileShardedArray":
        return self.reshape((self.num_tiles, *shape))

    def squeeze(self):
        squeezed_array = self.array.squeeze()
        has_single_tile = self.num_tiles == 1
        if has_single_tile:
            squeezed_array = squeezed_array.reshape((1, *squeezed_array.shape))
        return TileShardedArray(array=squeezed_array, tiles=self.tiles)  # type:ignore

    def __len__(self) -> int:
        return len(self.array)

    def __array__(self, dtype: DTypeLike = None):
        # Force converting to Numpy array.
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key: Union[SliceType, MultiSliceType]) -> "TileShardedArray":
        """Slice over the tile axis."""
        # Make sure we have a tuple of slices.
        if isinstance(key, (int, slice)):
            return self.__getitem__((key,))
        if not isinstance(key, tuple):
            raise ValueError(f"Unsupported tile sharded array slicing key: {key}.")

        # First key => always a slice so we keep the tile axis.
        k0 = key[0]
        if isinstance(k0, int):
            key = (slice(k0, k0 + 1), *key[1:])

        # Check we have a valid slice (keep memory contiguity).
        check_tile_array_multi_slice(key, self.array.shape)
        return TileShardedArray(array=self.array[key], tiles=self.tiles[key[0]])  # type:ignore


def tile_put_sharded(array: DeviceArray, tiles: Sequence[int]) -> TileShardedArray:
    """Shard a JAX array over tiles on the first axis.

    Args:
        array: Array to shard on the first axis.
        tiles: Collection of tiles ids to shard the array on.
    Returns:
        Tile sharded array.
    """
    # TODO: support JAX pytrees.
    return TileShardedArray(array=tile_put_sharded_prim(array, tiles), tiles=tiles)  # type:ignore


def tile_put_replicated(array: DeviceArray, tiles: Sequence[int]) -> TileShardedArray:
    """Replicate a JAX array over tiles on the first axis.

    Args:
        array: Array to replicate on tiles
        tiles: Collection of tiles ids to shard the array on.
    Returns:
        Tile sharded array.
    """
    # TODO: support JAX pytrees.
    return TileShardedArray(array=tile_put_replicated_prim(array, tiles), tiles=tiles)  # type:ignore


def tile_data_barrier(*args: TileShardedArray) -> Tuple[TileShardedArray, ...]:
    """Tile sharded arrays data barrier: force aligning between tiles in the Poplar program.

    Args:
        *args: Input tile sharded arrays.
    Returns:
        Output tile arrays.
    """
    assert all([isinstance(v, TileShardedArray) for v in args])
    # No need for a barrier when it is a single array.
    if len(args) == 1:
        return args[0]  # type:ignore

    inputs_tiles = [v.tiles for v in args]
    raw_inputs = [v.array for v in args]
    raw_outputs = tile_data_barrier_prim(raw_inputs, inputs_tiles)
    return tuple([TileShardedArray(output, input.tiles) for output, input in zip(raw_outputs, args)])  # type:ignore


tile_barrier = tile_data_barrier


def tile_gather(
    arr: Union[DeviceArray, TileShardedArray], indices: Sequence[int], tiles: Sequence[int], copy: bool = False
) -> TileShardedArray:
    """Gather a JAX array over tiles on the first axis.

    By default, if a slice of an input sharded array is already located on the proper tile,
    data will not be copied (no `Memcpy` vertex inserted).

    Args:
        arr: Array. Generic, or already tile sharded.
        indices: Gather (static) indices.
        tiles: IPU tiles sharding.
        copy: If True, data is always copied, even when already properly tile mapped.
    Returns:
        Sharded array over IPU tiles.
    """
    assert len(indices) == len(tiles)
    assert min(indices) >= 0
    assert max(indices) <= len(arr) - 1
    # Existing tile mapping? -1 by default when none.
    previous_tiles = tuple([-1] * len(arr))
    if isinstance(arr, TileShardedArray):
        previous_tiles = arr.tiles
    # Force copy => act like there is no pre-existing tile mapping.
    if copy:
        previous_tiles = tuple([-1] * len(previous_tiles))

    data_arr = arr.array if isinstance(arr, TileShardedArray) else arr
    gather_arr = tile_gather_prim(data_arr, previous_tiles, indices, tiles)
    return TileShardedArray(array=gather_arr, tiles=tiles)  # type:ignore
