# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from jax import core

from .tile_interpreter import TileShardedArray, register_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
)

# from jax._src.lax.control_flow.remat_impl import _optimization_barrier as optimization_barrier


hw_cycle_count_p = core.Primitive("hw_cycle_count")
hw_cycle_count_dtype = np.uint32


def hw_cycle_count(arg):
    return hw_cycle_count_p.bind(arg)


def hw_cycle_count_numpy_impl(arg):
    # Unsupported: zero cycle count.
    return arg, np.zeros((2,), dtype=hw_cycle_count_dtype)


def hw_cycle_count_abstract_eval(arg):
    assert isinstance(arg, core.ShapedArray)
    return arg, core.ShapedArray((2,), hw_cycle_count_dtype)


def hw_cycle_count_tile_translation_ipu(
    p: core.Primitive,
    tiles: Tuple[int, ...],
    inavals: List[core.ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU tile translation for custom arange vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: Op attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    inaval = inavals[0]
    _, outaval = hw_cycle_count_abstract_eval(inaval)

    vertex_name = make_ipu_vertex_name_templated("CycleCountBarrier", inaval.dtype)
    gp_filename = os.path.join(os.path.dirname(__file__), "vertex", "hw_vertex.cpp")

    # Translation rule to IPU vertex
    ipu_prim_info = IpuTileMapEquation(
        vname=vertex_name,
        pname=p.name,
        tiles=tiles,
        # IO vertex infos.
        inputs_info=[make_ipu_vertex_inout_info("data", inaval)],
        outputs_info=[make_ipu_vertex_inout_info("data", inaval), make_ipu_vertex_out_info("out", outaval)],
        # Perf. estimate from Poplar code.
        gp_filename=gp_filename,
        perf_estimate=50,
    )
    return ipu_prim_info


hw_cycle_count_p.map_primitive = False
hw_cycle_count_p.multiple_results = True
# Register the primal implementation with JAX
hw_cycle_count_p.def_impl(hw_cycle_count_numpy_impl)
# Register the abstract evaluation with JAX
hw_cycle_count_p.def_abstract_eval(hw_cycle_count_abstract_eval)
# Register tile IPU translation.
register_ipu_tile_primitive(hw_cycle_count_p, hw_cycle_count_tile_translation_ipu)


def ipu_hw_cycle_count(arg: TileShardedArray, sync: bool = False) -> Tuple[TileShardedArray, TileShardedArray]:
    """Get IPU hardware cycle count, with a given argument acting as barrier.

    See XLA optimization barrier for more information on the expected behaviour of a barrier.
    """
    assert isinstance(arg, TileShardedArray)
    return tile_map_primitive(hw_cycle_count_p, arg, sync=sync)  # type:ignore
