# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Tuple

import jax.lax
import jax.numpy as jnp
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
from .tile_linalg_qr import ipu_qr_shard_inputs as ipu_hessenberg_shard_inputs

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


# Heavily based on ipu_qr_iterations in tile_linalg_qr.py
# The body of the for-loop computes
# v = Householder(R[i])         # v is chosen to annihilate the elements below the first lower diagonal
# R = R - 2 *  v.reshape(-1, 1) @ (v.reshape(1, -1)  @ R)
# R = R - 2 * (R @ v.reshape(-1, 1)) @ v.reshape(1, -1)  # Not present in QR algorithm
# Q = Q - 2 * (Q @ v.reshape(-1, 1)) @ v.reshape(1, -1)


def roll_matrices(
    Q: TileShardedArray, RT: TileShardedArray, sdiag_full: TileShardedArray
) -> Tuple[TileShardedArray, TileShardedArray, TileShardedArray]:

    # nextRT = jnp.roll(RT.array, shift=(-1,0), axis=(0,1))
    nextRT = jnp.concatenate([RT.array[1:], RT.array[:1]])
    RT = tile_put_sharded(nextRT, RT.tiles)

    # nextQ = jnp.roll(Q.array, shift=(-1,0), axis=(0,1))
    nextQ = jnp.concatenate([Q.array[1:], Q.array[:1]])
    Q = tile_put_sharded(nextQ, Q.tiles)

    nextsdiag_full = jnp.roll(sdiag_full.array, shift=-1, axis=0)
    sdiag_full = tile_put_sharded(nextsdiag_full, sdiag_full.tiles)

    return Q, RT, sdiag_full


def ipu_hessenberg_body(
    i: int, carry: Tuple[TileShardedArray, TileShardedArray, TileShardedArray]
) -> Tuple[TileShardedArray, TileShardedArray, TileShardedArray]:

    Q, RT, sdiag_full = carry

    Rcol_array = RT.array[i].reshape(1, -1)
    sdiag_array = sdiag_full.array[i].reshape(1, -1)

    # These too are very expensive
    # Rcol_array = jnp.take_along_axis(RT.array, jnp.array([[i]],dtype=np.int32), axis=0)
    # sdiag_array = jnp.take_along_axis(sdiag_full.array, jnp.array([[i]],dtype=np.int32), axis=0)

    # Rcol_array = jnp.take(RT.array, i, axis=0).reshape(1,-1)
    # sdiag_array = jnp.take(sdiag_full.array, i, axis=0).reshape(1,-1)

    Rcol = tile_put_sharded(Rcol_array, [736])
    sdiag = tile_put_sharded(sdiag_array, [736])

    # start_idx = (i // 2) * 2
    start_idx = 0

    start_idxQ = tile_put_replicated(start_idx, Q.tiles)
    start_idxR = tile_put_replicated(start_idx, RT.tiles)

    # Correction vector. NOTE: computed on a single tile, changing at every loop.
    v, vrescale = tile_map(
        hessenberg_correction_vector_p, Rcol, sdiag, tile_put_replicated(i + 1, Rcol.tiles)
    )  # type:ignore

    # Replicate to all Q and R tiles.
    vQ = tile_put_replicated(v.array[0], Q.tiles)
    vR = tile_put_replicated(v.array[0], RT.tiles)
    # v normalization factor to pass to householder update.
    vrescaleQ = tile_put_replicated(vrescale.array[0], Q.tiles)
    vrescaleR = tile_put_replicated(vrescale.array[0], RT.tiles)

    # Using "smart" slicing to reduce compute to do.
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
    # RT_rolled = jnp.roll(RT.array, shift=i, axis=0)
    # RT_rolled_back = jnp.roll(RT_rolled.T, shift=-i, axis=0)
    # R = tile_put_sharded(RT_rolled_back, RT.tiles)
    R = tile_put_sharded(RT.array.T, RT.tiles)

    # Using "smart" slicing to reduce compute to do.
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

    # RT = tile_put_sharded(R.array.T, R.tiles)

    # R_rolled = jnp.roll(R.array, shift=i, axis=0)
    # R_rolled_back = jnp.roll(R_rolled.T, shift=-i, axis=0)
    # RT = tile_put_sharded(R_rolled_back, R.tiles)

    RT = tile_put_sharded(R.array.T, R.tiles)

    # rotate everything !!!!!
    # Q, RT, sdiag_full = roll_matrices(Q, RT, sdiag_full)

    return (Q, RT, sdiag_full)


def ipu_hessenberg_iterations(
    Q: TileShardedArray, RT: TileShardedArray, sdiag_full: TileShardedArray
) -> Tuple[TileShardedArray, TileShardedArray]:
    """IPU Hessenberg algorithm iterations.

    Args:
        Q: Initial Q sharded array.
        RT: Initial R.T sharded array.
        sdiag_full: Diagonal sign (replicated).
    Returns:
        (Q, RT) after N-2 iterations.
    """
    assert len(Q) == len(RT)
    N = len(Q)

    Q, RT, sdiag_full = jax.lax.fori_loop(0, N - 2, ipu_hessenberg_body, (Q, RT, sdiag_full))

    return (Q, RT)


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
    Q, RT, sdiag_full = ipu_hessenberg_shard_inputs(x, jax.numpy.sign(jax.numpy.diag(x)))
    # IPU QR iterations.
    return ipu_hessenberg_iterations(Q, RT, sdiag_full)
