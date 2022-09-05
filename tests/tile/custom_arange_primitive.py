# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from jax import core

from jax_ipu_research.tile import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_io_info,
    register_ipu_tile_primitive,
)

custom_arange_p = core.Primitive("custom_arange")


def custom_arange(size: int, dtype: Any):
    return custom_arange_p.bind(size=size, dtype=dtype)


def custom_arange_numpy_impl(size: int, dtype: Any):
    return np.arange(size).astype(dtype)


def custom_arange_abstract_eval(size: int, dtype: Any):
    return core.ShapedArray((size,), dtype)


def custom_arange_tile_translation_ipu(
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
    assert attributes is not None
    # Output shape.
    outshape = (attributes["size"],)
    outdtype = attributes["dtype"]
    outaval = core.ShapedArray(outshape, outdtype)
    gp_filename = os.path.join(os.path.dirname(__file__), "custom_arange_vertex.cpp")

    ipu_dtype = from_numpy_dtype_to_ipu_type(outdtype)
    vertex_name = f"CustomArangeVertex<{ipu_dtype.name.lower()}>"
    ipu_prim_info = IpuTileMapEquation(
        vname=vertex_name,
        pname=p.name,
        tiles=tiles,
        inputs_info=[],
        outputs_info=[make_ipu_vertex_io_info("out", IpuVertexIOType.Out, outaval)],
        attributes_u32=[],
        attributes_f32=[],
        gp_filename=gp_filename,
        perf_estimate=outshape[0] + 5,
    )
    return ipu_prim_info


custom_arange_p.map_primitive = False
# Register the primal implementation with JAX
custom_arange_p.def_impl(custom_arange_numpy_impl)
# Register the abstract evaluation with JAX
custom_arange_p.def_abstract_eval(custom_arange_abstract_eval)
# Register tile IPU translation.
register_ipu_tile_primitive(custom_arange_p, custom_arange_tile_translation_ipu)
