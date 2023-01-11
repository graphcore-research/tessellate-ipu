# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Tuple

import jax.lax
import numpy as np
from jax.core import ShapedArray

from jax_ipu_research.utils import Array

from .tile_array import TileShardedArray, tile_put_replicated, tile_put_sharded
from .tile_interpreter import create_ipu_tile_primitive, tile_map_primitive


def get_jacobi_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "vertex", "tile_jacobi_vertex.cpp")


# Jacobi symmetric Schur2
jacobi_sym_schur2_p = create_ipu_tile_primitive(
    "jacobi_sym_schur2",
    "JacobiSymSchur2",
    inputs=["pq", "pcol", "qcol"],
    outputs={"cs": ShapedArray((2,), dtype=np.float32)},
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)

jacobi_update_first_step_p = create_ipu_tile_primitive(
    "jacobi_update_first_step",
    "JacobiUpdateFirstStep",
    inputs=["rotset", "pcol", "qcol"],
    outputs={"cs": ShapedArray((2,), dtype=np.float32), "pcol_updated": 1, "qcol_updated": 2},
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


jacobi_update_second_step_p = create_ipu_tile_primitive(
    "jacobi_update_second_step",
    "JacobiUpdateSecondStep",
    inputs=["cs_arr", "rotset_arr", "rotset_idx_ignored", "pcol", "qcol"],
    outputs={"cs_arr": 0, "pcol_updated": 3, "qcol_updated": 4},
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


def jacobi_initial_rotation_set(N: int) -> np.ndarray:
    """Jacobi initial rotation array/set (N/2, 2)."""
    rot = np.arange(0, N).astype(np.int32).reshape((-1, 2))
    return rot


def jacobi_next_rotation_set(rot: np.ndarray) -> np.ndarray:
    """Jacobi next rotation set (N/2, 2)."""
    next_rot = np.copy(rot)
    # Translate columns.
    next_rot[2:, 0] = rot[1:-1, 0]
    next_rot[0:-1, 1] = rot[1:, 1]
    # Manage corners!
    next_rot[0, 1] = rot[1, 1]
    next_rot[1, 0] = rot[0, 1]
    next_rot[-1, 1] = rot[-1, 0]
    return next_rot


def jacobi_rotate_columns(pcols: Array, qcols: Array) -> Tuple[Array, Array]:
    """Jacobi rotation matrix columns between processors (tiles).

    This op is a pure view operation (slicing + concat), to comms between IPU tiles.

    NOTE: we use two arrays for storage of P and Q columns, instead of a single
    (N//2, 2, N) array, in order to have more flexible memory mapping on every tile
    (no requirement on contiguity between p and q. Potentially on different memory bank).

    See for the rotation description:
        Gene H. Golub, Charles F. Van Loan, MATRIX COMPUTATIONS, 3rd edition, Johns Hopkins Chapter 8.

    Args:
        pcols: P-index columns (N//2, N).
        qcols: Q-index columns (N//2, N).
    """
    assert pcols.shape == qcols.shape
    halfN = len(pcols)
    pcols_rotated = jax.lax.concatenate(
        [
            jax.lax.slice_in_dim(pcols, start_index=0, limit_index=1),
            jax.lax.slice_in_dim(qcols, start_index=0, limit_index=1),
            jax.lax.slice_in_dim(pcols, start_index=1, limit_index=halfN - 1),
        ],
        dimension=0,
    )
    qcols_rotated = jax.lax.concatenate(
        [
            jax.lax.slice_in_dim(qcols, start_index=1, limit_index=halfN),
            jax.lax.slice_in_dim(pcols, start_index=halfN - 1, limit_index=halfN),
        ],
        dimension=0,
    )
    return pcols_rotated, qcols_rotated


def jacobi_sort_columns(
    rot: np.ndarray, pcols: TileShardedArray, qcols: TileShardedArray
) -> Tuple[np.ndarray, TileShardedArray, TileShardedArray]:
    """Jacobi sorting of columns on every processor, ensuring that we always have p < q.

    This op is a pure view operation (slicing + concat). Just re-assigning columns within tiles.

    Args:
        rot: Rotation set, assigning columns to processors (static argnum).
        pcols: P-columns.
        qcols: Q-columns.
    Returns:
        Equivalent (rot, pcols, qcols) such that p < q on every processor/tile.
    """
    halfN = len(rot)
    assert len(pcols.array) == halfN
    assert len(qcols.array) == halfN
    assert pcols.tiles == qcols.tiles

    rot_sorted = []
    pcols_sorted = []
    qcols_sorted = []
    for idx in range(halfN):
        p, q = rot[idx]
        assert p != q
        if p < q:
            # Keep same ordering.
            rot_sorted.append((p, q))
            pcols_sorted.append(jax.lax.slice_in_dim(pcols.array, idx, idx + 1))
            qcols_sorted.append(jax.lax.slice_in_dim(qcols.array, idx, idx + 1))
        else:
            # Swap p and q (on the same tile).
            rot_sorted.append((q, p))
            pcols_sorted.append(jax.lax.slice_in_dim(qcols.array, idx, idx + 1))
            qcols_sorted.append(jax.lax.slice_in_dim(pcols.array, idx, idx + 1))

    rot_sorted_arr = np.array(rot_sorted, np.uint32)
    pcols_sorted_arr = TileShardedArray(jax.lax.concatenate(pcols_sorted, dimension=0), pcols.tiles)  # type:ignore
    qcols_sorted_arr = TileShardedArray(jax.lax.concatenate(qcols_sorted, dimension=0), qcols.tiles)  # type:ignore
    return (rot_sorted_arr, pcols_sorted_arr, qcols_sorted_arr)


def ipu_jacobi_eigh(x: Array, num_iters: int = 1) -> Tuple[Array, Array]:
    """IPU Eigen decomposition, implemented using Jacobi algorithm.

    Args:
        x: Symmetric matrix.
    Returns:
        (eigenvectors (N, N), eigenvalues (N,))
    """
    assert x.ndim == 2
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    assert N % 2 == 0
    assert N <= 1024
    halfN = N // 2

    tiles = tuple(range(0, halfN))
    # Initial rotation (i.e. allocation of columns per tiles).
    rotset = jacobi_initial_rotation_set(N)
    pcols = tile_put_sharded(jax.lax.slice_in_dim(x, 0, N, stride=2), tiles=tiles)
    qcols = tile_put_sharded(jax.lax.slice_in_dim(x, 1, N, stride=2), tiles=tiles)
    # Constant tensor of index to ignored at every iteration.
    rotset_index_ignored = tile_put_sharded(np.arange(0, halfN, dtype=np.uint32), tiles=tiles)

    for _ in range(num_iters):
        # All different size 2 partitions on columns.
        for _ in range(1, N):
            # Sorted rotation + columns. No copy, just on tile slicing + concat.
            rotset_sorted, pcols_sorted, qcols_sorted = jacobi_sort_columns(rotset, pcols, qcols)
            # TODO: on device create? or constant replicated?
            rotset_replicated = tile_put_replicated(rotset_sorted, tiles=tiles)
            rotset_sharded = tile_put_sharded(rotset_sorted, tiles=tiles)

            # Compute Schur decomposition + on-tile update of columns.
            cs_per_tile, pcols_sorted, qcols_sorted = tile_map_primitive(  # type:ignore
                jacobi_update_first_step_p, rotset_sharded, pcols_sorted, qcols_sorted, N=N
            )
            # Replicate Schur decomposition across all tiles: (2*N//2) comms.
            cs_replicated = tile_put_replicated(cs_per_tile.array, tiles=tiles)
            # Second Jacobi update step.
            cs_replicated, pcols_sorted, qcols_sorted = tile_map_primitive(  # type:ignore
                jacobi_update_second_step_p,
                cs_replicated,
                rotset_replicated,
                rotset_index_ignored,
                pcols_sorted,
                qcols_sorted,
                halfN=halfN,
            )

            # Unsort to keep the pure functional flow. No copy.
            _, pcols, qcols = jacobi_sort_columns(rotset, pcols_sorted, qcols_sorted)

            # Update rotation and columns.
            rotset = jacobi_next_rotation_set(rotset)
            # Move columns between tiles. 2*N commns per tile.
            pcols_array, qcols_array = jacobi_rotate_columns(pcols.array, qcols.array)
            pcols = tile_put_sharded(pcols_array, tiles=tiles)
            qcols = tile_put_sharded(qcols_array, tiles=tiles)

    # Re-organize pcols and qcols into the result matrix.
    result_rows = [None] * N
    for idx, (p, q) in enumerate(rotset):
        result_rows[p] = jax.lax.slice_in_dim(pcols.array, start_index=idx, limit_index=idx + 1)
        result_rows[q] = jax.lax.slice_in_dim(qcols.array, start_index=idx, limit_index=idx + 1)
    A = jax.lax.concatenate(result_rows, dimension=0)
    return A, None
