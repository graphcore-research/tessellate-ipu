# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from jax.core import Primitive, ShapedArray
from jax.lax import dot_general_p
from jax.lax.linalg import qr_p
from numpy.typing import DTypeLike

from jax_ipu_research.utils import Array

from .tile_array import TileShardedArray, tile_put_replicated, tile_put_sharded
from .tile_interpreter import create_simple_tile_primitive, register_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_lax_binary import scaled_sub_p
from .tile_interpreter_lax_dot import IpuConvVertexType
from .tile_interpreter_primitives import (  # make_ipu_vertex_attributes,
    IpuTileMapEquation,
    from_numpy_dtype_to_ipu_type,
    make_ipu_shaped_array,
    make_ipu_vertex_in_info,
    make_ipu_vertex_out_info,
)


def get_qr_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "vertex", "tile_qr_vertex.cpp")


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
        gp_filename=get_qr_vertex_gp_filename(),
        perf_estimate=size**2,
    )
    return ipu_prim_info


# Register QR tile IPU translation.
register_ipu_tile_primitive(qr_p, ipu_qr_primitive_translation)

# Vertex performing inplace update: x -= 2 * w @ v.T
qr_householder_update_p = create_simple_tile_primitive(
    "qr_householder_update",
    "QRHouseholderUpdateVertex<{x}>",
    ["x", "v", "w"],
    {"x": 0},
    gp_filename=get_qr_vertex_gp_filename(),
    perf_estimate=1000,
)

# Vertex computing QR correction vector
qr_correction_vector_p = create_simple_tile_primitive(
    "qr_correction_vector",
    "QRCorrectionVectorVertex<{Rcol}>",
    ["Rcol", "sdiag"],
    {"v": 0},
    gp_filename=get_qr_vertex_gp_filename(),
    perf_estimate=1000,
)


def ipu_qr(x: Array, xsdiag: Array) -> Tuple[Array, Array]:
    """IPU implementation of the QR algorithm.

    This implementation is returing R^T instead of R, as it is more
    efficient to store the former while iterating.

    Args:
        x: Symmetric matrix.
    Returns:
        Q, R^T matrices.
    """
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]

    # Sharding R and Q
    Q_tiles = tuple(range(0, N))
    R_tiles = tuple(range(N, 2 * N))

    # TODO: on-device construction of identity
    Q = tile_put_sharded(np.identity(N, dtype=x.dtype), Q_tiles)
    # TODO: shard R and Q on different tiles.
    RT = tile_put_sharded(x.T, R_tiles)
    # Replicate once on all tiles. Faster then for the looping.
    sdiag_full = tile_put_replicated(xsdiag, R_tiles)

    for cidx in range(N - 1):
        # Extract the proper R column (no tile copy, pure view).
        Rcol = RT[cidx]
        sdiag = sdiag_full[cidx]
        # Correction vector. NOTE: computed on a single tile, changing at every loop.
        v: TileShardedArray = tile_map_primitive(qr_correction_vector_p, Rcol, sdiag, col_idx=cidx)  # type:ignore

        # Replicate to all Q and R tiles.
        vQ = tile_put_replicated(v.array[0], Q_tiles)
        vR = tile_put_replicated(v.array[0], R_tiles)

        # w = R^T @ v
        w: TileShardedArray = tile_map_primitive(  # type:ignore
            dot_general_p,
            vR,
            RT,
            dimension_numbers=(([0], [0]), ([], [])),
            precision=None,
            preferred_element_type=None,
            ipu_vertex_type=IpuConvVertexType.ConvPartialHMAC,
        )
        # Inplace update of R.
        RT = tile_map_primitive(scaled_sub_p, RT, vR, w)  # type:ignore

        # w = Q @ v
        w: TileShardedArray = tile_map_primitive(  # type:ignore
            dot_general_p,
            vQ,
            Q,
            dimension_numbers=(([0], [0]), ([], [])),
            precision=None,
            preferred_element_type=None,
            ipu_vertex_type=IpuConvVertexType.ConvPartialHMAC,
        )
        # Inplace update of Q.
        Q = tile_map_primitive(scaled_sub_p, Q, vQ, w)  # type:ignore

    # Return JAX arrays directly?
    return (Q, RT)
