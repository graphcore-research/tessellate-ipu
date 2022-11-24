# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

import jax
import numpy as np
from jax.core import Primitive, ShapedArray

from jax_ipu_research.utils import DTypeLike

from .tile_array import TileShardedArray
from .tile_interpreter import register_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    make_ipu_vertex_attributes,
    make_ipu_vertex_inputs,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_outputs,
)

"""Bridging to the low level interface allowing to control IPU random seed for every tile.

Equivalent to Poplar APIs `getHwSeeds` and `setHwSeeds`
"""
ipu_set_hw_seeds_p = Primitive("ipu_set_hw_seeds")


def get_ipu_num_worker_contexts() -> int:
    """Get the IPU `num_worker_contexts` property."""
    return jax.devices("ipu")[0].num_worker_contexts


def ipu_set_hw_seeds_abstract_eval(seeds: ShapedArray) -> ShapedArray:
    # Poplar tensor: poplar::UNSIGNED_INT, {numTiles, numWorkerContexts, 4}
    assert seeds.shape[0] == get_ipu_num_worker_contexts()
    assert seeds.shape[1] == 4
    assert seeds.dtype == np.uint32
    return seeds


def ipu_set_hw_seeds_translation_ipu(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    assert len(inavals) == 1
    inavals_dict = {"seeds": inavals[0]}
    inout_names = {"seeds"}
    # Translation rule to IPU vertex.
    ipu_prim_info = IpuTileMapEquation(
        vname="poplar_rt::SetHwSeedsSupervisor",
        pname=p.name,
        tiles=tiles,
        inputs_info=make_ipu_vertex_inputs(inavals_dict, inout_names=inout_names),
        outputs_info=make_ipu_vertex_outputs(inavals_dict, inout_names=inout_names),
        # Optional GP filename and perf. estimate.
        perf_estimate=14 + 10 * get_ipu_num_worker_contexts(),
    )
    return ipu_prim_info


def ipu_set_hw_seeds_tmap(seeds: TileShardedArray) -> TileShardedArray:
    """Set IPU hardware seeds on a collection of tiles."""
    return tile_map_primitive(ipu_set_hw_seeds_p, seeds)  # type:ignore


ipu_set_hw_seeds_p.def_abstract_eval(ipu_set_hw_seeds_abstract_eval)
register_ipu_tile_primitive(ipu_set_hw_seeds_p, ipu_set_hw_seeds_translation_ipu)


ipu_get_hw_seeds_p = Primitive("ipu_get_hw_seeds")


def ipu_get_hw_seeds_abstract_eval() -> ShapedArray:
    # Poplar tensor: poplar::UNSIGNED_INT, {numTiles, numWorkerContexts, 4}
    return ShapedArray((get_ipu_num_worker_contexts(), 4), np.uint32)


def ipu_get_hw_seeds_translation_ipu(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    outavals_dict = {"seeds": ipu_get_hw_seeds_abstract_eval()}
    # Translation rule to IPU vertex.
    ipu_prim_info = IpuTileMapEquation(
        vname="poplar_rt::GetHwSeedsSupervisor",
        pname=p.name,
        tiles=tiles,
        inputs_info=[],
        outputs_info=make_ipu_vertex_outputs(outavals_dict),
        # Optional GP filename and perf. estimate.
        perf_estimate=14 + 7 * get_ipu_num_worker_contexts(),
    )
    return ipu_prim_info


def ipu_get_hw_seeds_tmap(tiles: Tuple[int, ...]) -> TileShardedArray:
    """Get IPU hardware seeds from a collection of tiles."""
    return tile_map_primitive(ipu_get_hw_seeds_p, tiles=tiles)  # type:ignore


ipu_get_hw_seeds_p.def_abstract_eval(ipu_get_hw_seeds_abstract_eval)
register_ipu_tile_primitive(ipu_get_hw_seeds_p, ipu_get_hw_seeds_translation_ipu)


ipu_random_uniform_p = Primitive("ipu_random_uniform")


def ipu_random_uniform_abstract_eval(size: int, dtype: DTypeLike, offset: Any, scale: Any) -> ShapedArray:
    dtype = np.dtype(dtype)
    # Type supported in the IPU vertex.
    assert dtype in {np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.int32)}
    return ShapedArray((size,), dtype)


def ipu_random_uniform_translation_ipu(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    assert attributes is not None

    size = int(attributes["size"])
    dtype = np.dtype(attributes["dtype"])
    offset = attributes["offset"]
    scale = attributes["scale"]
    vname = make_ipu_vertex_name_templated("poprand::Uniform", dtype)

    outavals_dict = {"out": ipu_random_uniform_abstract_eval(size, dtype, offset, scale)}
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(offset=offset, scale=scale)
    # Translation rule to IPU vertex.
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[],
        outputs_info=make_ipu_vertex_outputs(outavals_dict),
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


ipu_random_uniform_p.def_abstract_eval(ipu_random_uniform_abstract_eval)
register_ipu_tile_primitive(ipu_random_uniform_p, ipu_random_uniform_translation_ipu)


def ipu_random_uniform_tmap(
    tiles: Tuple[int, ...], size: int, dtype: DTypeLike, offset: float = 0.0, scale: float = 1.0
) -> TileShardedArray:
    """IPU Uniform sampling on a collection of tiles."""
    return tile_map_primitive(  # type:ignore
        ipu_random_uniform_p, size=size, dtype=dtype, offset=offset, scale=scale, tiles=tiles
    )


ipu_random_normal_p = Primitive("ipu_random_normal")


def ipu_random_normal_abstract_eval(size: int, dtype: DTypeLike, mean: float, stddev: float) -> ShapedArray:
    dtype = np.dtype(dtype)
    # Type supported in the IPU vertex.
    assert dtype in {np.dtype(np.float16), np.dtype(np.float32)}
    return ShapedArray((size,), dtype)


def ipu_random_normal_translation_ipu(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    assert attributes is not None

    size = int(attributes["size"])
    dtype = np.dtype(attributes["dtype"])
    mean = float(attributes["mean"])
    stddev = float(attributes["stddev"])
    vname = make_ipu_vertex_name_templated("poprand::Normal", dtype)

    outavals_dict = {"out": ipu_random_normal_abstract_eval(size, dtype, mean, stddev)}
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(mean=mean, stdDev=stddev)
    # Translation rule to IPU vertex.
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[],
        outputs_info=make_ipu_vertex_outputs(outavals_dict),
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


ipu_random_normal_p.def_abstract_eval(ipu_random_normal_abstract_eval)
register_ipu_tile_primitive(ipu_random_normal_p, ipu_random_normal_translation_ipu)


def ipu_random_normal_tmap(
    tiles: Tuple[int, ...], size: int, dtype: DTypeLike, mean: float = 0.0, stddev: float = 1.0
) -> TileShardedArray:
    """IPU Normal sampling on a collection of tiles."""
    return tile_map_primitive(  # type:ignore
        ipu_random_normal_p, size=size, dtype=dtype, mean=mean, stddev=stddev, tiles=tiles
    )
