# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Tuple

import jax.lax
import numpy as np
from jax.core import ShapedArray

from .tile_array import TileShardedArray, tile_put_replicated, tile_put_sharded
from .tile_interpreter import create_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_vertex_utils import make_ipu_vector1d_worker_offsets

Array = Any


def get_qr_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "vertex", "tile_qr_vertex.cpp")


dot_product1d_p = create_ipu_tile_primitive(
    "dot_product1d",
    "DotProduct1dVertex",
    inputs=["x", "y"],
    outputs={"partials": ShapedArray((12,), dtype=np.float32)},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[0].size, vector_size=2, num_workers=6, wdtype=np.uint16
        )
    },
    # tmp_space=ShapedArray((12,), dtype=np.float32),
    gp_filename=get_qr_vertex_gp_filename(),
    perf_estimate=1000,
)

"""Vertex computing QR correction vector.
"""
qr_correction_vector_p = create_ipu_tile_primitive(
    "qr_correction_vector",
    "QRCorrectionVectorVertex",
    inputs=["Rcol", "sdiag"],
    outputs={"v": 0, "vrescale": ShapedArray((1,), dtype=np.float32)},
    gp_filename=get_qr_vertex_gp_filename(),
    perf_estimate=1000,
)

"""Vertex QR HouseHolder performing row inplace update: x -= scale1[0] * scale2[0] * v
"""
qr_householder_row_update_p = create_ipu_tile_primitive(
    "qr_householder_row_update",
    "QRHouseholderRowUpdateVertex",
    inputs=["x", "v", "scale1", "scale2"],
    outputs={"x": 0},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[1].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_qr_vertex_gp_filename(),
    perf_estimate=1000,
)


def ipu_qr_shard_inputs(x: Array, xsdiag: Array) -> Tuple[TileShardedArray, TileShardedArray, TileShardedArray]:
    """IPU QR initial sharding of input arrays across IPU tiles.

    Args:
        x: X array.
        sdiag: X diagonal sign.
    Returns:
        Tile sharded Q, RT, sdiag.
    """
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    # Sharding R and Q
    Q_tiles = tuple(range(0, N))
    R_tiles = tuple(range(N, 2 * N))

    # TODO: on-device construction of identity
    Q = tile_put_sharded(np.identity(N, dtype=x.dtype), Q_tiles)
    RT = tile_put_sharded(x.T, R_tiles)
    # Replicate once on all tiles. Faster then for the looping.
    sdiag_full = tile_put_replicated(xsdiag, R_tiles)
    return Q, RT, sdiag_full


def ipu_qr_iterations(
    Q: TileShardedArray, RT: TileShardedArray, sdiag_full: TileShardedArray
) -> Tuple[TileShardedArray, TileShardedArray]:
    """IPU QR algorithm iterations.

    Args:
        Q: Initial Q sharded array.
        RT: Initial R.T sharded array.
        sdiag_full: Diagonal sign (replicated).
    Returns:
        (Q, RT) after N-1 iterations.
    """
    assert len(Q) == len(RT)
    N = len(Q)
    # Sharding of R and Q on tiles.
    Q_tiles = Q.tiles
    R_tiles = RT.tiles

    for cidx in range(N - 1):
        # From which column to start computation: skipping zeros. Must be a multiple of 2 for proper vectorization.
        start_idx = (cidx // 2) * 2
        # Extract the proper R column (no tile copy, pure view).
        Rcol = RT[cidx]
        sdiag = sdiag_full[cidx]
        # Correction vector. NOTE: computed on a single tile, changing at every loop.
        v, vrescale = tile_map_primitive(qr_correction_vector_p, Rcol, sdiag, col_idx=cidx)  # type:ignore

        # Replicate to all Q and R tiles.
        vQ = tile_put_replicated(v.array[0], Q_tiles)
        vR = tile_put_replicated(v.array[0], R_tiles)
        # v normalization factor to pass to householder update.
        vrescaleQ = tile_put_replicated(vrescale.array[0], Q_tiles)
        vrescaleR = tile_put_replicated(vrescale.array[0], R_tiles)

        # Using "smart" slicing to reduce compute to do.
        # w = R^T @ v
        w = tile_map_primitive(dot_product1d_p, vR[:, start_idx:], RT[:, start_idx:])
        w = tile_map_primitive(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
        # Inplace update of R.
        RT = tile_map_primitive(  # type:ignore
            qr_householder_row_update_p, RT, vR[:, start_idx:], w, vrescaleR, start_idx=start_idx  # type:ignore
        )

        # w = Q @ v
        w = tile_map_primitive(dot_product1d_p, vQ[:, start_idx:], Q[:, start_idx:])
        w = tile_map_primitive(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
        # Inplace update of Q.
        Q = tile_map_primitive(
            qr_householder_row_update_p, Q, vQ[:, start_idx:], w, vrescaleQ, start_idx=start_idx  # type:ignore
        )

    return (Q, RT)


def ipu_qr(x: Array, xsdiag: Array) -> Tuple[Array, Array]:
    """IPU implementation of the QR algorithm.

    This implementation is returing R^T instead of R, as it is more
    efficient to store the former while iterating.

    Args:
        x: Symmetric matrix.
    Returns:
        Q, R^T matrices (as tile sharded arrays).
    """
    # Initialize Q, RT, sdiag.
    Q, RT, sdiag_full = ipu_qr_shard_inputs(x, xsdiag)
    # IPU QR iterations.
    return ipu_qr_iterations(Q, RT, sdiag_full)
