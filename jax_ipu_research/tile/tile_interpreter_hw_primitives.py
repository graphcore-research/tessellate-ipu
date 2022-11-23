# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

import jax
import numpy as np
from jax import core

from jax_ipu_research.utils import is_ipu_model

from .tile_interpreter import register_ipu_tile_primitive
from .tile_interpreter_primitives import IpuTileMapEquation, make_ipu_vertex_constant_info, make_ipu_vertex_out_info

hw_cycle_count_p = core.Primitive("hw_cycle_count")
hw_cycle_count_dtype = np.uint32


def hw_cycle_count():
    return hw_cycle_count_p.bind()


def hw_cycle_count_numpy_impl():
    # Unsupported: zero cycle count.
    return np.zeros((2,), dtype=hw_cycle_count_dtype)


def hw_cycle_count_abstract_eval():
    return core.ShapedArray((2,), hw_cycle_count_dtype)


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
    assert len(inavals) == 0
    outaval = hw_cycle_count_abstract_eval()
    is_model = is_ipu_model(jax.devices("ipu")[0])

    vertex_name = "poplar_rt::TimeItStart"
    inputs_info = []
    if is_model:
        # IPU model: no vertex to call, just forward zero constant tensor.
        vertex_name = ""
        inputs_info.append(make_ipu_vertex_constant_info("in", np.zeros((2,), dtype=np.uint32)))

    # Translation rule to IPU vertex
    ipu_prim_info = IpuTileMapEquation(
        vname=vertex_name,
        pname=p.name,
        tiles=tiles,
        # IO vertex infos.
        inputs_info=inputs_info,
        outputs_info=[make_ipu_vertex_out_info("out", outaval)],
        # Perf. estimate from Poplar code.
        perf_estimate=39,
    )
    return ipu_prim_info


hw_cycle_count_p.map_primitive = False
# Register the primal implementation with JAX
hw_cycle_count_p.def_impl(hw_cycle_count_numpy_impl)
# Register the abstract evaluation with JAX
hw_cycle_count_p.def_abstract_eval(hw_cycle_count_abstract_eval)
# Register tile IPU translation.
register_ipu_tile_primitive(hw_cycle_count_p, hw_cycle_count_tile_translation_ipu)
