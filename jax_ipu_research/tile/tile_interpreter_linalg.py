# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

from jax.core import Primitive, ShapedArray
from jax.lax.linalg import qr_p
from numpy.typing import DTypeLike

from .tile_interpreter import register_ipu_tile_primitive
from .tile_interpreter_primitives import (  # make_ipu_vertex_attributes,
    IpuTileMapEquation,
    from_numpy_dtype_to_ipu_type,
    make_ipu_shaped_array,
    make_ipu_vertex_in_info,
    make_ipu_vertex_out_info,
)


def make_qr_fullname_vertex(dtype: DTypeLike) -> str:
    """qr custom vertex name."""
    dtype_ipu = from_numpy_dtype_to_ipu_type(dtype).name.lower()
    return f"LinalgQRVertex<{dtype_ipu}>"


def ipu_qr_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU unary primitive translation rule to IPU vertex.

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
    # Only supporting square matrices.
    assert inaval.ndim == 2
    assert inaval.shape[0] == inaval.shape[1]

    size = inaval.shape[0]
    qaval, raval = qr_p.abstract_eval(inaval, full_matrices=True)[0]

    gp_filename = os.path.join(os.path.dirname(__file__), "vertex", "tile_qr_vertex.cpp")
    vname = make_qr_fullname_vertex(inaval.dtype)
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_in_info("x", inaval)],
        outputs_info=[
            make_ipu_vertex_out_info("Q", qaval),
            make_ipu_vertex_out_info("R", raval),
        ],
        attributes_i32=[],
        attributes_f32=[],
        # Temporary scratch space to use by the vertex (zero=unused).
        tmp_space_aval=make_ipu_shaped_array((size, 3), inaval.dtype),
        # Optional GP filename and perf. estimate.
        gp_filename=gp_filename,
        perf_estimate=size**2,
    )
    return ipu_prim_info


# Register QR tile IPU translation.
register_ipu_tile_primitive(qr_p, ipu_qr_primitive_translation)
