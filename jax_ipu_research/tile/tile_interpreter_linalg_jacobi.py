# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Tuple

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from jax_ipu_research.utils import Array

from .tile_array import TileShardedArray, tile_data_barrier, tile_gather, tile_put_replicated, tile_put_sharded
from .tile_interpreter import create_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_lax_sort import ipu_argsort_quadratic_unique
from .tile_interpreter_linalg import make_ipu_vector1d_worker_offsets


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
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[1].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


jacobi_update_second_step_p = create_ipu_tile_primitive(
    "jacobi_update_second_step",
    "JacobiUpdateSecondStep",
    inputs=["cs_arr", "rotset_arr", "rotset_idx_ignored", "pcol", "qcol"],
    outputs={"cs_arr": 0, "pcol_updated": 3, "qcol_updated": 4},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[3].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)

jacobi_update_eigenvectors_p = create_ipu_tile_primitive(
    "jacobi_update_eigenvectors",
    "JacobiUpdateEigenvectors",
    inputs=["cs", "vpcol", "vqcol"],
    outputs={"vpcol": 1, "vqcol": 2},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[1].size, vector_size=2, wdtype=np.uint16
        )
    },
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
            # pcols_sorted.append(jax.lax.slice_in_dim(pcols.array, idx, idx + 1))
            # qcols_sorted.append(jax.lax.slice_in_dim(qcols.array, idx, idx + 1))
            pcols_sorted.append(pcols.array[idx])
            qcols_sorted.append(qcols.array[idx])
        else:
            # Swap p and q (on the same tile).
            rot_sorted.append((q, p))
            # pcols_sorted.append(jax.lax.slice_in_dim(qcols.array, idx, idx + 1))
            # qcols_sorted.append(jax.lax.slice_in_dim(pcols.array, idx, idx + 1))
            pcols_sorted.append(qcols.array[idx])
            qcols_sorted.append(pcols.array[idx])

    pcols_sorted = [jax.lax.expand_dims(t, (0,)) for t in pcols_sorted]
    qcols_sorted = [jax.lax.expand_dims(t, (0,)) for t in qcols_sorted]

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

    Atiles = tuple(range(0, halfN))
    Vtiles = tuple(range(halfN, 2 * halfN))
    # Initial rotation (i.e. allocation of columns per tiles).
    rotset = jacobi_initial_rotation_set(N)
    Apcols = tile_put_sharded(jax.lax.slice_in_dim(x, 0, N, stride=2), tiles=Atiles)
    Aqcols = tile_put_sharded(jax.lax.slice_in_dim(x, 1, N, stride=2), tiles=Atiles)
    # Initial eigenvectors (identity matrix).
    Vpcols = tile_put_sharded(np.identity(N)[0::2], tiles=Vtiles)
    Vqcols = tile_put_sharded(np.identity(N)[1::2], tiles=Vtiles)

    # Constant tensor of index to ignored at every iteration.
    rotset_index_ignored = tile_put_sharded(np.arange(0, halfN, dtype=np.uint32), tiles=Atiles)

    for _ in range(num_iters):
        # All different size 2 partitions on columns.
        for _ in range(1, N):
            # Sorted rotation + columns. No copy, just on tile slicing + concat.
            rotset_sorted, Apcols_sorted, Aqcols_sorted = jacobi_sort_columns(rotset, Apcols, Aqcols)
            _, Vpcols_sorted, Vqcols_sorted = jacobi_sort_columns(rotset, Vpcols, Vqcols)

            # TODO: on device create? or constant replicated?
            rotset_replicated = tile_put_replicated(rotset_sorted, tiles=Atiles)
            rotset_sharded = tile_put_sharded(rotset_sorted, tiles=Atiles)

            # Compute Schur decomposition + on-tile update of columns.
            cs_per_tile, Apcols_sorted, Aqcols_sorted = tile_map_primitive(  # type:ignore
                jacobi_update_first_step_p, rotset_sharded, Apcols_sorted, Aqcols_sorted, N=N
            )
            # Replicate Schur decomposition across all A tiles: (2*N//2) comms.
            cs_replicated = tile_put_replicated(cs_per_tile.array, tiles=Atiles)
            # Just copy Schur decomposition to associated V tiles.
            cs_Vtiles = tile_put_sharded(cs_per_tile.array, tiles=Vtiles)

            # Second Jacobi update step.
            cs_replicated, Apcols_sorted, Aqcols_sorted = tile_map_primitive(  # type:ignore
                jacobi_update_second_step_p,
                cs_replicated,
                rotset_replicated,
                rotset_index_ignored,
                Apcols_sorted,
                Aqcols_sorted,
                halfN=halfN,
            )
            # Jacobi eigenvectors update step.
            Vpcols_sorted, Vqcols_sorted = tile_map_primitive(  # type:ignore
                jacobi_update_eigenvectors_p,
                cs_Vtiles,
                Vpcols_sorted,
                Vqcols_sorted,
            )

            # Unsort to keep the pure functional flow. No copy.
            _, Apcols, Aqcols = jacobi_sort_columns(rotset, Apcols_sorted, Aqcols_sorted)
            _, Vpcols, Vqcols = jacobi_sort_columns(rotset, Vpcols_sorted, Vqcols_sorted)
            # Barrier, to make we sync. both set of tiles A and V
            Apcols, Aqcols, Vpcols, Vqcols = tile_data_barrier(Apcols, Aqcols, Vpcols, Vqcols)

            # Update rotation and columns.
            rotset = jacobi_next_rotation_set(rotset)
            # Move columns between tiles. 2*N commns per tile.
            Apcols, Aqcols = tile_rotate_columns(Apcols, Aqcols)
            Vpcols, Vqcols = tile_rotate_columns(Vpcols, Vqcols)

    # Re-organize pcols and qcols into the result matrix.
    Aresult_rows = [None] * N
    Vresult_cols = [None] * N
    for idx, (p, q) in enumerate(rotset):
        Aresult_rows[p] = jax.lax.slice_in_dim(Apcols.array, start_index=idx, limit_index=idx + 1)
        Aresult_rows[q] = jax.lax.slice_in_dim(Aqcols.array, start_index=idx, limit_index=idx + 1)

        Vresult_cols[p] = jax.lax.slice_in_dim(Vpcols.array, start_index=idx, limit_index=idx + 1)
        Vresult_cols[q] = jax.lax.slice_in_dim(Vqcols.array, start_index=idx, limit_index=idx + 1)

    A = jax.lax.concatenate(Aresult_rows, dimension=0)
    VT = jax.lax.concatenate(Vresult_cols, dimension=0)
    return A, VT


def tile_rotate_columns(pcols: TileShardedArray, qcols: TileShardedArray) -> Tuple[TileShardedArray, TileShardedArray]:
    """Rotate columns between tiles using a static `tile_gather`."""
    assert pcols.shape == qcols.shape
    assert pcols.tiles == qcols.tiles
    halfN = pcols.shape[0]
    # Concat all columns, in order to perform a single gather.
    all_cols = TileShardedArray(  # type:ignore
        jax.lax.concatenate([pcols.array, qcols.array], dimension=0), (*pcols.tiles, *qcols.tiles)
    )
    # Express rotation as indices for a gather op.
    pcols_indices = (0, halfN, *range(1, halfN - 1))
    qcols_indices = (*range(halfN + 1, 2 * halfN), halfN - 1)
    all_indices = (*pcols_indices, *qcols_indices)
    # Move columns around + re-split between pcols and qcols.
    all_cols_updated = tile_gather(all_cols, all_indices, all_cols.tiles)
    return all_cols_updated[:halfN], all_cols_updated[halfN:]


def ipu_eigh(
    x: Array, *, lower: bool = True, symmetrize_input: bool = False, sort_eigenvalues: bool = True, num_iters: int = 1
) -> Tuple[Array, Array]:
    """IPU (optimized) eigh implementation.

    Args:
        x: Input matrix (N,N) (Nd not supported).
        lower: Not supported.
        symmetrize_input: Not supported, must be false.
        sort_eigenvalues: Sort in ascending order.
    Returns:
        Tuple of eigenvectors (N, N), eigenvalues (N,)
    """
    assert x.ndim == 2
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    assert N % 2 == 0
    assert N <= 1024
    assert not symmetrize_input

    A, VT = ipu_jacobi_eigh(x, num_iters=num_iters)
    eigvalues = jnp.diag(A)
    eigvectors_tr = VT
    # Sorting eigen values, assuming uniqueness!
    if sort_eigenvalues:
        indices = ipu_argsort_quadratic_unique(eigvalues)
        eigvalues = eigvalues[indices]
        eigvectors_tr = eigvectors_tr[indices]
    return eigvectors_tr.T, eigvalues
