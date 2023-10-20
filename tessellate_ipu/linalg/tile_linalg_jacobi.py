# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Tuple

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

# import tessellate_ipu
from tessellate_ipu import (
    TileShardedArray,
    create_ipu_tile_primitive,
    tile_constant_sharded,
    tile_data_barrier,
    tile_gather,
    tile_map,
    tile_put_replicated,
    tile_put_sharded,
)
from tessellate_ipu.core import make_ipu_vector1d_worker_offsets, make_ipu_vector1d_worker_offsets_and_sizes
from tessellate_ipu.lax import tile_fill
from tessellate_ipu.utils import NDArray

Array = Any


INDEX_PREFIX = 2
"""Index prefix size in p/q columns.
"""


def get_jacobi_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "../core", "vertex", "tile_jacobi_vertex.cpp")


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
    inputs=["pcol", "qcol"],
    outputs={
        "rotset_sorted": ShapedArray((2,), dtype=np.uint32),
        "cs": ShapedArray((2,), dtype=np.float32),
        "pcol_updated": 0,
        "qcol_updated": 1,
    },
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[0].size - INDEX_PREFIX, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


jacobi_update_second_step_p = create_ipu_tile_primitive(
    "jacobi_update_second_step",
    "JacobiUpdateSecondStep",
    inputs=["cs_arr", "rotset_sorted_arr", "rotset_idx_ignored", "pcol", "qcol"],
    outputs={"cs_arr": 0, "pcol_updated": 3, "qcol_updated": 4},
    constants={
        # NOTE: using grain_size=4 because of partial loop unrolling
        # Rescale the size to be directly in grain size unit.
        "worker_offsets_sizes": lambda inavals, *_: make_ipu_vector1d_worker_offsets_and_sizes(
            inavals[3].size - INDEX_PREFIX, vector_size=2, grain_size=4, wdtype=np.uint16, allow_overlap=True
        )
        // np.array([[1, 2]], dtype=np.uint16)
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)

jacobi_update_eigenvectors_p = create_ipu_tile_primitive(
    "jacobi_update_eigenvectors",
    "JacobiUpdateEigenvectors",
    inputs=["cs", "vpcol", "vqcol"],
    outputs={"vpcol_out": 1, "vqcol_out": 2},  # Bug when inplace update?
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            # Remove 2 for pq indices prefix.
            inavals[1].size - INDEX_PREFIX,
            vector_size=2,
            wdtype=np.uint16,
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


def jacobi_initial_rotation_set(N: int) -> NDArray[np.uint32]:
    """Jacobi initial rotation array/set (N/2, 2)."""
    rot = np.arange(0, N).astype(np.uint32).reshape((-1, 2))
    return rot


def jacobi_initial_pqindices(N: int) -> Tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Jacobi initial p/q indices arrays.
    Padded to (N/2, 2) for 64bits alignment.

    Returns:
        A tuple of p/q indices arrays.
    """
    rotset = jacobi_initial_rotation_set(N)
    pindices = rotset[:, :1]
    qindices = rotset[:, 1:]
    pindices = np.concatenate([pindices, pindices], axis=1)
    qindices = np.concatenate([qindices, qindices], axis=1)
    return (pindices, qindices)


def tile_sharded_pq_columns(
    pcols: Array, qcols: Array, tiles: Tuple[int, ...]
) -> Tuple[TileShardedArray, TileShardedArray]:
    """Tile sharding of p/q columns arrays + adding indexing prefix.

    Args:
        pcols/qcols: (M, N) arrays.
        tiles: Collection of tiles to shard on.
    Returns:
        Pair of tile sharded array (M, N+2), with indexing prefix.
    """
    assert pcols.shape == qcols.shape
    assert len(pcols.shape) == 2
    N = pcols.shape[0] * 2
    # N = pcols.shape[-1]

    pindices, qindices = jacobi_initial_pqindices(N)
    pindices_prefix = tile_constant_sharded(pindices.view(np.float32), tiles=tiles)
    qindices_prefix = tile_constant_sharded(qindices.view(np.float32), tiles=tiles)
    # Prepend the p/q indices. Note: keeping 64bits alignment with 2 uint32s.
    pcols = jax.lax.concatenate([pindices_prefix.array, pcols], dimension=1)
    qcols = jax.lax.concatenate([qindices_prefix.array, qcols], dimension=1)
    # Shard between tiles. TODO: single call with tuple.
    pcols = tile_put_sharded(pcols, tiles=tiles)
    qcols = tile_put_sharded(qcols, tiles=tiles)
    return pcols, qcols


def jacobi_next_rotation_set(rot: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Jacobi next rotation set (N/2, 2).

    In short: moving p columns to the right, q columns to the left, with
        p[0] not moving.
    """
    next_rot = np.copy(rot)
    # Translate columns.
    next_rot[2:, 0] = rot[1:-1, 0]
    next_rot[0:-1, 1] = rot[1:, 1]
    # Manage corners!
    next_rot[0, 1] = rot[1, 1]
    next_rot[1, 0] = rot[0, 1]
    next_rot[-1, 1] = rot[-1, 0]
    return next_rot


def jacobi_sort_rotation_set(rotset: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Sort the p, q indices in the Jacobi rotation set, such p < q."""
    pindices, qindices = rotset[:, 0], rotset[:, 1]
    pindices, qindices = np.minimum(pindices, qindices), np.maximum(pindices, qindices)
    return np.stack([pindices, qindices], axis=-1)


def tile_rotate_columns(pcols: TileShardedArray, qcols: TileShardedArray) -> Tuple[TileShardedArray, TileShardedArray]:
    """Rotate columns between tiles using a static `tile_gather`.

    We follow the Jacobi rotation patterns between tiles. In short
        - moving `pcols` to the "left"
        - moving `qcols` to the "right"
    """
    assert pcols.shape == qcols.shape
    assert pcols.tiles == qcols.tiles
    halfN = pcols.shape[0]
    N = halfN * 2
    # Concat all columns, in order to perform a single gather.
    all_cols = TileShardedArray(
        jax.lax.concatenate([pcols.array, qcols.array], dimension=0), (*pcols.tiles, *qcols.tiles)
    )

    pcols_indices = np.arange(0, halfN, dtype=np.int32)
    qcols_indices = np.arange(halfN, N, dtype=np.int32)
    # Rotation of columns between tiles (see Jacobi alg.)
    # Roughtly: pcols move to the right, qcols to the left.
    pcols_indices_new = np.concatenate([pcols_indices[0:1], qcols_indices[0:1], pcols_indices[1:-1]])
    qcols_indices_new = np.concatenate([qcols_indices[1:], pcols_indices[-1:]])

    # Move columns around!
    pcols_updated = tile_gather(all_cols, pcols_indices_new.tolist(), pcols.tiles)
    qcols_updated = tile_gather(all_cols, qcols_indices_new.tolist(), qcols.tiles)
    return pcols_updated, qcols_updated

    # FIXME: understand why Poplar add a copy with the following code.
    # all_indices = np.concatenate([pcols_indices_new, qcols_indices_new])
    # all_cols_updated = tile_gather(all_cols, all_indices.tolist(), all_cols.tiles)
    # return all_cols_updated[:halfN], all_cols_updated[halfN:]


def ipu_jacobi_eigh_body(idx: Array, inputs: Tuple[TileShardedArray, ...]) -> Tuple[TileShardedArray, ...]:
    """IPU Jacobi eigen-decomposition algorithm main body.

    Args:
        idx: Loop index.
        inputs: Tile sharded Apcols, Aqcols, Vpcols, Vqcols
    Returns:
        Apcols, Aqcols, Vpcols, Vqcols after a main Jacobi update.
    """
    Apcols, Aqcols, Vpcols, Vqcols = inputs
    Atiles = Apcols.tiles
    Vtiles = Vpcols.tiles
    halfN = Apcols.shape[0]

    with jax.named_scope("jacobi_eigh"):
        with jax.named_scope("Apqcols_rotation"):
            Apcols, Aqcols = tile_rotate_columns(Apcols, Aqcols)
        with jax.named_scope("Vpqcols_rotation"):
            Vpcols, Vqcols = tile_rotate_columns(Vpcols, Vqcols)
        # Barrier, to make we sync. both set of tiles A and V and force fused comms.
        Apcols, Aqcols, Vpcols, Vqcols = tile_data_barrier(Apcols, Aqcols, Vpcols, Vqcols)

        # Sharded constant with p/q indices to ignore in second update stage.
        with jax.named_scope("rotset_index_ignored"):
            rotset_index_ignored = tile_constant_sharded(np.arange(0, halfN, dtype=np.uint32), tiles=Atiles)

        # Compute Schur decomposition + on-tile update of columns.
        # Note: not expecting p < q. Input pcols/qcols sorted inside the vertex.
        rotset_sorted_sharded, cs_per_tile, Apcols, Aqcols = tile_map(  # type:ignore
            jacobi_update_first_step_p, Apcols, Aqcols
        )
        # Append zero indices to the rotset, for loop unrolling in `jacobi_update_second_step`
        rotset_zeros = tile_fill((2,), 0, dtype=rotset_sorted_sharded.dtype, tiles=(0,))
        # Barrier to make sure communication gets fused into a single block.
        rotset_zeros, rotset_sorted_sharded, cs_per_tile = tile_data_barrier(
            rotset_zeros, rotset_sorted_sharded, cs_per_tile
        )
        rotset_sorted_sharded = TileShardedArray.concatenate([rotset_sorted_sharded, rotset_zeros])

        # Replicate Schur decomposition + rotset across all A tiles: (2*N//2) comms.
        with jax.named_scope("rotset_sorted_replicated"):
            rotset_sorted_replicated = tile_put_replicated(rotset_sorted_sharded.array, tiles=Atiles)
        with jax.named_scope("cs_replicated_sharded"):
            cs_replicated = tile_put_replicated(cs_per_tile.array, tiles=Atiles)
            # Just copy Schur decomposition to associated V tiles (no need to replicate).
            cs_sharded_Vtiles = tile_put_sharded(cs_per_tile.array, tiles=Vtiles)

        # Barrier to force all communications to be fused.
        cs_replicated, cs_sharded_Vtiles, rotset_sorted_replicated = tile_data_barrier(
            cs_replicated, cs_sharded_Vtiles, rotset_sorted_replicated
        )
        # Second Jacobi update step.
        # Note: does not require sorting of pcols and qcols.
        cs_replicated, Apcols, Aqcols = tile_map(  # type:ignore
            jacobi_update_second_step_p,
            cs_replicated,
            rotset_sorted_replicated,
            rotset_index_ignored,
            Apcols,
            Aqcols,
        )
        # Jacobi eigenvectors update step.
        Vpcols, Vqcols = tile_map(  # type:ignore
            jacobi_update_eigenvectors_p,
            cs_sharded_Vtiles,
            Vpcols,
            Vqcols,
        )

        # Apcols, Aqcols, Vpcols, Vqcols = tile_data_barrier(Apcols, Aqcols, Vpcols, Vqcols)
        # # Move columns between tiles following Jacobi rotation pattern. 2*N commns per tile.
        # with jax.named_scope("Apqcols_rotation"):
        #     Apcols, Aqcols = tile_rotate_columns(Apcols, Aqcols)
        # with jax.named_scope("Vpqcols_rotation"):
        #     Vpcols, Vqcols = tile_rotate_columns(Vpcols, Vqcols)
        return Apcols, Aqcols, Vpcols, Vqcols


def ipu_jacobi_eigh_iteration(idx: Array, all_AV_cols: Tuple[TileShardedArray, ...]) -> Tuple[TileShardedArray, ...]:
    """IPU Eigen decomposition: single iteration of the Jacobi algorithm.

    NOTE: the goal is to have a function which can be easily combined with `fori_loop`.

    Args:
        all_AV_cols: A and V sharded p/q columns + index prefixing..
    Returns:
        Tuple of updated A and V matrices p/q columns.
    """
    Apcols, Aqcols, Vpcols, Vqcols = all_AV_cols
    shape = Apcols.shape
    assert len(shape) == 2
    assert shape[0] * 2 + INDEX_PREFIX == shape[1]
    N = shape[0] * 2
    # Jacobi eigh iteration as a single fori_loop.
    Apcols, Aqcols, Vpcols, Vqcols = jax.lax.fori_loop(1, N, ipu_jacobi_eigh_body, (Apcols, Aqcols, Vpcols, Vqcols))
    return (Apcols, Aqcols, Vpcols, Vqcols)


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

    tile_offset = 1
    Atiles = tuple(range(tile_offset, halfN + tile_offset))
    Vtiles = tuple(range(halfN + tile_offset, 2 * halfN + tile_offset))
    # Initial "eigenvalues" matrix.
    Apcols_init = jax.lax.slice_in_dim(x, 0, N, stride=2)
    Aqcols_init = jax.lax.slice_in_dim(x, 1, N, stride=2)
    # Initial eigenvectors (identity matrix).
    Vpcols_init = np.identity(N)[0::2]
    Vqcols_init = np.identity(N)[1::2]

    # Shard p/q columns + adding index prefix.
    Apcols, Aqcols = tile_sharded_pq_columns(Apcols_init, Aqcols_init, tiles=Atiles)
    Vpcols, Vqcols = tile_sharded_pq_columns(Vpcols_init, Vqcols_init, tiles=Vtiles)
    # JAX fori_loop => no Python unrolling and code bloating!
    Apcols, Aqcols, Vpcols, Vqcols = jax.lax.fori_loop(
        0, num_iters, ipu_jacobi_eigh_iteration, (Apcols, Aqcols, Vpcols, Vqcols)
    )

    # Back to JAX arrays, removing indexing prefix.
    (Apcols, Aqcols, Vpcols, Vqcols) = map(lambda x: x.array[:, INDEX_PREFIX:], (Apcols, Aqcols, Vpcols, Vqcols))

    # Expect the output to follow the initial rotation set columns split.
    rotset = jacobi_initial_rotation_set(N)
    # Re-organize pcols and qcols into the result matrix.
    Aresult_rows = [None] * N
    Vresult_cols = [None] * N
    for idx, (p, q) in enumerate(rotset):
        Aresult_rows[p] = jax.lax.slice_in_dim(Apcols, start_index=idx, limit_index=idx + 1)
        Aresult_rows[q] = jax.lax.slice_in_dim(Aqcols, start_index=idx, limit_index=idx + 1)

        Vresult_cols[p] = jax.lax.slice_in_dim(Vpcols, start_index=idx, limit_index=idx + 1)
        Vresult_cols[q] = jax.lax.slice_in_dim(Vqcols, start_index=idx, limit_index=idx + 1)

    A = jax.lax.concatenate(Aresult_rows, dimension=0)
    VT = jax.lax.concatenate(Vresult_cols, dimension=0)
    return A, VT


def permute_pq_indices(
    pindices: NDArray[np.int32], qindices: NDArray[np.int32], rotset_permute_mask: NDArray[np.bool_]
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Permute p,q indices based on a mask.

    Args, Returns: (N//2,) shaped arrays.
    """
    return (np.where(rotset_permute_mask, pindices, qindices), np.where(rotset_permute_mask, qindices, pindices))


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
    # Sorting eigen values.
    if sort_eigenvalues:
        indices = jax.lax.iota(np.int32, len(eigvalues))
        eigvalues, indices = jax.lax.sort_key_val(eigvalues, indices)
        eigvectors_tr = eigvectors_tr[indices]

    # TODO: understand memory layout bug when not forcing the data to be re-organized.
    # Is it related to host rearrangement?
    eigvectors = tile_put_sharded(eigvectors_tr.T, tiles=tuple(range(N)))
    return eigvectors.array, eigvalues
