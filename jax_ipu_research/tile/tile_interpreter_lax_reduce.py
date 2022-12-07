# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

import numpy as np
from jax.core import Primitive, ShapedArray
from jax.lax import reduce_and_p, reduce_max_p, reduce_min_p, reduce_or_p, reduce_prod_p, reduce_sum_p
from numpy.typing import DTypeLike

from .tile_interpreter import register_ipu_tile_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    make_ipu_vertex_attributes,
    make_ipu_vertex_io_info,
    make_ipu_vertex_name_templated,
)

_reduce_primitive_to_basename: Dict[Primitive, str] = {
    reduce_sum_p: "ReduceAdd",
    reduce_max_p: "ReduceMax",
    reduce_min_p: "ReduceMin",
    reduce_prod_p: "ReduceMul",
    reduce_or_p: "ReduceOr",
    reduce_and_p: "ReduceAnd",
}


def make_continuous_reduce_vertex_fullname(
    reduce_p: Primitive, partial_dtype: DTypeLike, out_dtype: DTypeLike, is_update: bool = False
) -> str:
    """Generate a (continous) popops reduce vertex name.

    Args:
        reduce_p: Reduce primitive.
        partial_dtype: Dtype used for partials.
        out_dtype: Output dtype.
        is_update: Updating the output tensor inplace.
    Returns:
        Full popops continuous reduce vertex name.
    """
    basename = _reduce_primitive_to_basename[reduce_p]
    basename = f"popops::{basename}"
    return make_ipu_vertex_name_templated("popops::ContinuousReduce", basename, partial_dtype, out_dtype, is_update)


def ipu_reduce_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `reduce` primitive translation rule to IPU vertex.

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
    # TODO: understand convention between axes/axis/dimensions!
    axes = tuple(sorted(attributes["axes"]))
    assert len(axes) > 0
    first_axis = axes[0]
    # Only supporting reduction on the last axes for now (i.e. no striding).
    if axes != tuple(range(axes[0], inaval.ndim)):
        raise NotImplementedError(
            f"IPU tile mapped `{p.name}` only supporting (partial or full) reduction on the last axes."
        )
    outaval = p.abstract_eval(*inavals, axes=axes)[0]

    # Supporting partial reduce (i.e. last dimensions only).
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(
        numOutputsM1=np.prod(inaval.shape[:first_axis]) - 1, numPartials=np.prod(inaval.shape[first_axis:])
    )
    vname = make_continuous_reduce_vertex_fullname(p, inaval.dtype, inaval.dtype, False)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_io_info("partials", IpuVertexIOType.In, inaval)],
        outputs_info=[make_ipu_vertex_io_info("out", IpuVertexIOType.Out, outaval)],
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


# Register all supported JAX reduce ops.
for p in _reduce_primitive_to_basename.keys():
    register_ipu_tile_primitive(p, ipu_reduce_primitive_translation)
