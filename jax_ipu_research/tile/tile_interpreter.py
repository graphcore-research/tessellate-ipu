# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Building IPU tile MPMD programming as a custom JAX interpreter (https://github.com/google/jax/tree/main/jax/interpreters).

In particular, we need a registry mapping JAX primitives to IPU vertex (and additionally support custom IPU vertex).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from jax.core import Primitive
from jax.interpreters.xla import ShapedArray

from .tile_array import TileShardedArray
from .tile_interpreter_primitives import IpuTileMapEquation, tile_map_equation_call

IpuVertexTranslation = Callable[
    [Primitive, Tuple[int, ...], List[ShapedArray], Optional[Dict[str, Any]]], IpuTileMapEquation
]
"""Ipu vertex translation: callable translating a JAX primitive (with inputs/outputs) into a full
vertex info data structure.
"""

_ipu_tile_primitive_registry: Dict[str, Tuple[Primitive, IpuVertexTranslation]] = {}
"""Global registry mapping JAX primitives to IPU vertex translation rules.

The registry is indexed by the primitive name.
"""


def tile_map_primitive(
    primitive: Primitive,
    inputs: List[TileShardedArray],
    attributes: Dict[str, Any] = None,
    tiles: Optional[Tuple[int, ...]] = None,
) -> List[TileShardedArray]:
    """Map a JAX primitive over tiles.

    Args:
        primitive: JAX primitive to map.
        inputs: List of input sharded arrays.
        attributes: Attributes to pass to the JAX primitive (and translation rule).
        tiles: Optional tile mapping, provided when there is no input.
    Returns:
        List of output sharded arrays.
    """
    if primitive is None:
        # No primitive: by default a no-op.
        return inputs
    if primitive.name not in _ipu_tile_primitive_registry:
        raise KeyError(f"The JAX primitive `{primitive}` is not supported for tile mapping on the IPU.")
    if not all([isinstance(v, TileShardedArray) for v in inputs]):
        raise TypeError("Tile map inputs must be `TileShardedArray` instances.")

    # TODO: check tile mapping consistency.
    tiles = tiles or inputs[0].tiles
    attributes = attributes or {}

    # Get the IPU tile map equation corresponding.
    _, ipu_prim_translation = _ipu_tile_primitive_registry[primitive.name]
    tile_map_eqn: IpuTileMapEquation = ipu_prim_translation(primitive, tiles, [v.tile_aval for v in inputs], attributes)
    tile_map_eqn_json: str = tile_map_eqn.to_json_str()
    # Call JAX tile custom primitive, dispatching properly the equation call.
    outputs = tile_map_equation_call(
        [v.device_array for v in inputs],
        pname=primitive.name,
        tiles=tiles,
        tile_map_eqn_json=tile_map_eqn_json,
        **attributes,
    )
    # Convert back to TileShardedArray.
    if not primitive.multiple_results:
        return TileShardedArray(outputs, tiles)  # type:ignore
    return [TileShardedArray(v, tiles) for v in outputs]  # type:ignore


def register_ipu_tile_primitive(primitive: Primitive, translation: IpuVertexTranslation):
    """Register an IPU tile vertex translation from JAX primitive.

    Args:
        primitive: JAX primitive.
        translation: IPU vertex translation rule.
    """
    global _ipu_tile_primitive_registry
    _ipu_tile_primitive_registry[primitive.name] = (primitive, translation)


def get_ipu_tile_primitive_translation(pname: str) -> Tuple[Primitive, IpuVertexTranslation]:
    """Ge the primitive and IPU translation corresponding to a primitive name."""
    return _ipu_tile_primitive_registry[pname]
