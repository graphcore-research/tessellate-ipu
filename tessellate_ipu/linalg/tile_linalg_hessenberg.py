# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Tuple

import jax.lax

from tessellate_ipu import TileShardedArray, tile_data_barrier, tile_map, tile_put_replicated, tile_put_sharded

from .tile_linalg_qr import dot_product1d_p
from .tile_linalg_qr import ipu_qr_shard_inputs as ipu_hessenberg_shard_inputs
from .tile_linalg_qr import qr_correction_vector_p, qr_householder_row_update_p

Array = Any

# Heavily based on ipu_qr_iterations in tile_linalg_qr.py
# The body of the for-loop computes
# v = Householder(R[i])         # v is chosen to annihilate the elements below the first lower diagonal
# R = R - 2 *  v.reshape(-1, 1) @ (v.reshape(1, -1)  @ R)
# R = R - 2 * (R @ v.reshape(-1, 1)) @ v.reshape(1, -1)  # Not present in QR algorithm
# Q = Q - 2 * (Q @ v.reshape(-1, 1)) @ v.reshape(1, -1)


def ipu_hessenberg_iterations(
    Q: TileShardedArray, RT: TileShardedArray, sdiag_full: TileShardedArray
) -> Tuple[TileShardedArray, TileShardedArray]:
    """IPU Hessenberg algorithm iterations.

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

    for cidx in range(N - 2):
        # From which column to start computation: skipping zeros. Must be a multiple of 2 for proper vectorization.
        start_idx = (cidx // 2) * 2
        # Extract the proper R column (no tile copy, pure view).
        Rcol = RT[cidx]
        sdiag = sdiag_full[cidx]
        # Correction vector. NOTE: computed on a single tile, changing at every loop.
        v, vrescale = tile_map(qr_correction_vector_p, Rcol, sdiag, col_idx=cidx + 1)  # type:ignore

        # Replicate to all Q and R tiles.
        vQ = tile_put_replicated(v.array[0], Q_tiles)
        vR = tile_put_replicated(v.array[0], R_tiles)
        # v normalization factor to pass to householder update.
        vrescaleQ = tile_put_replicated(vrescale.array[0], Q_tiles)
        vrescaleR = tile_put_replicated(vrescale.array[0], R_tiles)

        # Using "smart" slicing to reduce compute to do.
        # w = R^T @ v
        w = tile_map(
            dot_product1d_p, vR[:, start_idx:], RT[:, start_idx:]
        )  # this returns size 12 array (6 worker threads)
        w = tile_map(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
        # Inplace update of R.
        RT = tile_map(  # type:ignore
            qr_householder_row_update_p, RT, vR[:, start_idx:], w, vrescaleR, start_idx=start_idx  # type:ignore
        )

        # w = Q @ v
        w = tile_map(dot_product1d_p, vQ[:, start_idx:], Q[:, start_idx:])
        w = tile_map(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
        # Inplace update of Q.
        Q = tile_map(
            qr_householder_row_update_p, Q, vQ[:, start_idx:], w, vrescaleQ, start_idx=start_idx  # type:ignore
        )
        RT, Q = tile_data_barrier(RT, Q)

        R = tile_put_sharded(RT.array.T, RT.tiles)
        # Using "smart" slicing to reduce compute to do.
        # w = R^T @ v
        w = tile_map(
            dot_product1d_p, vR[:, start_idx:], R[:, start_idx:]
        )  # this returns size 12 array (6 worker threads)
        w = tile_map(jax.lax.reduce_sum_p, w, axes=(0,))  # type:ignore
        # Inplace update of R.
        R = tile_map(  # type:ignore
            qr_householder_row_update_p, R, vR[:, start_idx:], w, vrescaleR, start_idx=start_idx  # type:ignore
        )

        RT = tile_put_sharded(R.array.T, R.tiles)

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
    Q, RT, sdiag_full = ipu_hessenberg_shard_inputs(x, jax.numpy.sign(jax.numpy.diag(x)))
    # IPU QR iterations.
    return ipu_hessenberg_iterations(Q, RT, sdiag_full)
