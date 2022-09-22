# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Tuple, Union

import chex
import numpy as np
from jax.interpreters.xla import DeviceArray, ShapedArray

from .tile_array_primitives import tile_put_replicated_prim, tile_put_sharded_prim


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
    def aval(self) -> ShapedArray:
        if isinstance(self.array, np.ndarray):
            return ShapedArray(self.array.shape, self.array.dtype)
        return self.array.aval

    @property
    def tile_aval(self) -> ShapedArray:
        """Abstract val, at the tile level."""
        aval = self.aval
        return ShapedArray(aval.shape[1:], aval.dtype)

    @property
    def device_array(self) -> DeviceArray:
        return self.array

    def __array__(self, dtype: Any = None):
        # Force converting to Numpy array.
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key: Union[int, slice]) -> "TileShardedArray":
        """Slice over the tile axis."""
        if not isinstance(key, (int, slice)):
            raise ValueError(f"Unsupported tile sharded array slicing key: {key}.")
        if isinstance(key, int):
            # Integer get converted into slicing, for consistency.
            key = slice(key, key + 1)
        return TileShardedArray(array=self.array[key], tiles=self.tiles[key])  # type:ignore


def tile_put_sharded(array: DeviceArray, tiles: Tuple[int, ...]) -> TileShardedArray:
    """Shard a JAX array over tiles on the first axis.

    Args:
        array: Array to shard on the first axis.
        tiles: Collection of tiles ids to shard the array on.
    Returns:
        Tile sharded array.
    """
    return TileShardedArray(array=tile_put_sharded_prim(array, tiles), tiles=tiles)  # type:ignore


def tile_put_replicated(array: DeviceArray, tiles: Tuple[int, ...]) -> TileShardedArray:
    """Replicate a JAX array over tiles on the first axis.

    Args:
        array: Array to replicate on tiles
        tiles: Collection of tiles ids to shard the array on.
    Returns:
        Tile sharded array.
    """
    return TileShardedArray(array=tile_put_replicated_prim(array, tiles), tiles=tiles)  # type:ignore
