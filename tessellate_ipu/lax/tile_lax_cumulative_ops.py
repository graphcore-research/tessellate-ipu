# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

from jax.core import Primitive, ShapedArray
from jax.lax import cummax_p, cummin_p, cumprod_p, cumsum_p

from tessellate_ipu.core import (
    IpuTileMapEquation,
    make_ipu_vertex_in_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
    register_ipu_tile_primitive,
)

_cumop_primitive_to_opcode: Dict[Primitive, int] = {
    cumsum_p: 0,
    cummin_p: 1,
    cummax_p: 2,
    cumprod_p: 3,
}


def get_cumulative_ops_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "../core", "vertex", "tile_cumulative_ops_vertex.cpp")


def ipu_cumop_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU cumulative ops LAX primitive translation rule to IPU vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    assert attributes is not None
    inaval = inavals[0]
    axis = attributes.get("axis", 0)
    reverse = attributes.get("reverse", False)

    # Subset of configuration supported.
    assert axis == 0
    assert not reverse
    assert len(inaval.shape) == 1
    outaval = inaval
    # TessellateIPU custom cumulative vertices.
    opcode = _cumop_primitive_to_opcode[p]
    vname = make_ipu_vertex_name_templated("tl::CumulativeOp", inaval.dtype, opcode)
    inputs_info = [make_ipu_vertex_in_info("in", inaval)]
    outputs_info = [make_ipu_vertex_out_info("out", outaval)]
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
        attributes_i32=[],
        attributes_f32=[],
        gp_filename=get_cumulative_ops_gp_filename(),
        perf_estimate=inaval.size * 6,
    )
    return ipu_prim_info


# Register JAX LAX cumulative-ops primitive.
register_ipu_tile_primitive(cumsum_p, ipu_cumop_primitive_translation)
register_ipu_tile_primitive(cummax_p, ipu_cumop_primitive_translation)
register_ipu_tile_primitive(cummin_p, ipu_cumop_primitive_translation)
register_ipu_tile_primitive(cumprod_p, ipu_cumop_primitive_translation)
