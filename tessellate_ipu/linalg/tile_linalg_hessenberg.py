# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import math
import os
from typing import Any, Tuple

import jax.lax
import numpy as np  # used for np.float32, shouldn't we use jax types?
from jax.core import ShapedArray

from tessellate_ipu import (
    TileShardedArray,
    create_ipu_tile_primitive,
    tile_data_barrier,
    tile_map,
    tile_put_replicated,
    tile_put_sharded,
)
from tessellate_ipu.core.tile_interpreter_vertex_utils import make_ipu_vector1d_worker_offsets

from .tile_linalg_qr import dot_product1d_p

Array = Any


def get_hessenberg_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "../core", "vertex", "tile_hessenberg_vertex.cpp")


dot_product1d_indexed_p = create_ipu_tile_primitive(
    "dot_product1d_indexed",
    "DotProduct1dIndexedVertex",
    inputs=["x", "y", "start_idx"],
    outputs={"partials": ShapedArray((12,), dtype=np.float32)},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[0].size, vector_size=2, num_workers=6, wdtype=np.uint16
        )
    },
    # tmp_space=ShapedArray((12,), dtype=np.float32),
    gp_filename=get_hessenberg_vertex_gp_filename(),
    perf_estimate=1000,
)

"""Vertex computing Hessenberg correction vector.
"""
hessenberg_correction_vector_p = create_ipu_tile_primitive(
    "hessenberg_correction_vector",
    "HessenbergCorrectionVectorVertex",
    inputs=["Rcol", "sdiag", "cidx"],
    outputs={"v": 0, "vrescale": ShapedArray((1,), dtype=np.float32)},
    gp_filename=get_hessenberg_vertex_gp_filename(),
    perf_estimate=1000,
)

"""Vertex Hessenberg HouseHolder performing row inplace update: x -= scale1[0] * scale2[0] * v
"""
hessenberg_householder_row_update_p = create_ipu_tile_primitive(
    "hessenberg_householder_row_update",
    "HessenbergHouseholderRowUpdateVertex",
    inputs=["x", "v", "scale1", "scale2", "start_idx_"],
    outputs={"x": 0},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[1].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_hessenberg_vertex_gp_filename(),
    perf_estimate=1000,
)


def ipu_hessenberg_shard_inputs(x: Array, xsdiag: Array) -> Tuple[TileShardedArray, TileShardedArray, TileShardedArray]:
    """IPU QR initial sharding of input arrays across IPU tiles.

    Args:
        x: X array.
        sdiag: X diagonal sign.
    Returns:
        Tile sharded Q, R, sdiag.
    """
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    n_tiles = 1472
    # Sharding R and Q

    n_per_tile = math.ceil(N / float(n_tiles))
    full_tiles = N % n_tiles
    if full_tiles == 0:
        full_tiles = n_tiles

    Q_tiles = [i for i in range(full_tiles) for _ in range(n_per_tile)] + [
        i for i in range(full_tiles, n_tiles) for _ in range(n_per_tile - 1)
    ]
    R_tiles = Q_tiles

    # TODO: on-device construction of identity
    Q = tile_put_sharded(np.identity(N, dtype=x.dtype), Q_tiles)
    R = tile_put_sharded(x, R_tiles)
    # Replicate once on all tiles. Faster then for the looping.
    sdiag_full = tile_put_replicated(xsdiag.T, R_tiles)
    return Q, R, sdiag_full


# Heavily based on ipu_qr_iterations in tile_linalg_qr.py
# The body of the for-loop computes
# v = Householder(R[i])         # v is chosen to annihilate the elements below the first lower diagonal
# R = R - 2 *  v.reshape(-1, 1) @ (v.reshape(1, -1)  @ R)
# R = R - 2 * (R @ v.reshape(-1, 1)) @ v.reshape(1, -1)  # Not present in QR algorithm
# Q = Q - 2 * (Q @ v.reshape(-1, 1)) @ v.reshape(1, -1)


def ipu_hessenberg_body(
    i: int, carry: Tuple[TileShardedArray, TileShardedArray, TileShardedArray]
) -> Tuple[TileShardedArray, TileShardedArray, TileShardedArray]:

    Q, R, sdiag_full = carry

    dim_numbers = jax.lax.GatherDimensionNumbers(offset_dims=tuple(), collapsed_slice_dims=(0,), start_index_map=(0,))

    i_rep = tile_put_replicated(jax.numpy.array([[i]], dtype=np.uint32), R.tiles)

    Rcol = tile_map(
        jax.lax.gather_p,
        R,
        i_rep,
        dimension_numbers=dim_numbers,
        slice_sizes=(1,),
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    )  # => TileShardedArray() (Num_tiles, 1)

    Rcol_replicated = tile_put_replicated(Rcol.array, tiles=[736])  # type:ignore

    sdiag = tile_map(
        jax.lax.gather_p,
        sdiag_full,
        i_rep,
        dimension_numbers=dim_numbers,
        slice_sizes=(1,),
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    )  # => TileShardedArray() (Num_tiles, 1)

    sdiag_rep = tile_put_replicated(sdiag.array, Rcol_replicated.tiles)  # type:ignore

    # start_idx = (i // 2) * 2
    start_idx = 0

    start_idxQ = tile_put_replicated(start_idx, Q.tiles)
    start_idxR = tile_put_replicated(start_idx, R.tiles)

    # Alternative: we pass the whole RT and sdiag; then we extract the result from the i-th tile

    # Correction vector. Computed
    v, vrescale = tile_map(
        hessenberg_correction_vector_p, Rcol_replicated, sdiag_rep, tile_put_replicated(i + 1, Rcol_replicated.tiles)
    )  # type:ignore
    # v, vrescale = tile_map(
    #     hessenberg_correction_vector_p, RT, sdiag_full, tile_put_replicated(i + 1, RT.tiles)
    # )  # type:ignore

    # This compiles
    # vi = tile_gather(v.array[i], [0], [0]

    # Replicate to all Q and R tiles.
    vQ = tile_put_replicated(v.array, Q.tiles)  # 0
    vR = tile_put_replicated(v.array, R.tiles)  # 0
    # v normalization factor to pass to householder update.
    vrescaleQ = tile_put_replicated(vrescale.array, Q.tiles)  # 0
    vrescaleR = tile_put_replicated(vrescale.array, R.tiles)  # 0

    # Alternative using tile_gather
    # vQ = tile_gather(v, [i]*len(Q.tiles), list(Q.tiles), copy=False)    # 0
    # vR = tile_gather(v, [i]*len(RT.tiles), RT.tiles)    # 0
    # # v normalization factor to pass to householder update.
    # vrescaleQ = tile_gather(vrescale, [i]*len(Q.tiles), Q.tiles)    # 0
    # vrescaleR = tile_gather(vrescale, [i]*len(RT.tiles), RT.tiles)    #

    RT = tile_put_sharded(R.array.T, R.tiles)

    # w = R^T @ v
    w = tile_map(
        # dot_product1d_indexed_p, vR, RT, start_idxR
        dot_product1d_p,
        vR,
        RT,
    )  # this returns size 12 array (6 worker threads)
    w = tile_map(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
    # Inplace update of R.
    RT = tile_map(  # type:ignore
        hessenberg_householder_row_update_p, RT, vR, w, vrescaleR, start_idxR  # type:ignore
    )

    # w = Q @ v
    # w = tile_map(dot_product1d_indexed_p, vQ, Q, start_idxQ)
    w = tile_map(dot_product1d_p, vQ, Q)
    w = tile_map(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
    # Inplace update of Q.
    Q = tile_map(
        hessenberg_householder_row_update_p, Q, vQ, w, vrescaleQ, start_idxQ  # type:ignore
    )
    RT, Q = tile_data_barrier(RT, Q)

    # Transpose the RT matrix so that we can do the right product
    R = tile_put_sharded(RT.array.T, RT.tiles)

    # w = R^T @ v
    w = tile_map(
        # dot_product1d_indexed_p, vR, R, start_idxR
        dot_product1d_p,
        vR,
        R,
    )  # this returns size 12 array (6 worker threads)
    w = tile_map(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
    # Inplace update of R.
    R = tile_map(  # type:ignore
        hessenberg_householder_row_update_p, R, vR, w, vrescaleR, start_idxR  # type:ignore
    )

    return (Q, R, sdiag_full)


def ipu_hessenberg_iterations(
    Q: TileShardedArray, R: TileShardedArray, sdiag_full: TileShardedArray
) -> Tuple[TileShardedArray, TileShardedArray]:
    """IPU Hessenberg algorithm iterations.

    Args:
        Q: Initial Q sharded array.
        RT: Initial R.T sharded array.
        sdiag_full: Diagonal sign (replicated).
    Returns:
        (Q, RT) after N-2 iterations.
    """
    assert len(Q) == len(R)
    N = len(Q)

    Q, R, sdiag_full = jax.lax.fori_loop(0, N - 2, ipu_hessenberg_body, (Q, R, sdiag_full))

    return (Q, R)


def ipu_hessenberg(x: Array) -> Tuple[Array, Array]:
    """IPU implementation of the Hessenberg decomposition via Householder reflections.

    This implementation is returing R^T instead of R, as it is more
    efficient to store the former while iterating.

    Args:
        x: Symmetric matrix.
    Returns:
        Q, R^T matrices (as tile sharded arrays).
    """
    # Initialize Q, RT, sdiag.
    Q, R, sdiag_full = ipu_hessenberg_shard_inputs(x, jax.numpy.sign(jax.numpy.diag(x)))
    # IPU QR iterations.
    return ipu_hessenberg_iterations(Q, R, sdiag_full)
