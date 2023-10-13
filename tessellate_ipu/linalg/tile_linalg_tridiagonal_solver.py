import os.path as osp
from typing import Any, Tuple

import jax

from tessellate_ipu import TileShardedArray, create_ipu_tile_primitive, tile_map, tile_put_sharded

jax.config.FLAGS.jax_platform_name = "cpu"

Array = Any


vertex_filename = osp.join(osp.dirname(__file__), "../core", "vertex", "tile_tridiagonal_solver_vertex.cpp")

tridiagonal_solver_p = create_ipu_tile_primitive(
    "tridiagonal_solver",
    "TridiagonalSolverVertex",
    inputs=["d", "u", "l", "b"],
    outputs={"d": 0, "u": 1, "l": 2, "b": 3},
    gp_filename=vertex_filename,
    perf_estimate=100,
)


def ipu_tridiag_solve_shard_inputs(
    diag: Array, udiag: Array, ldiag: Array, rhs: Array
) -> Tuple[TileShardedArray, TileShardedArray, TileShardedArray, TileShardedArray]:
    """
    diag: main diagonal, (1,N)
    udiag: upper diagonal, (1,N), the last element is not used
    ldiag: lower diagonal, (1,N), the first element is not used, i.e. A[1,0] == ldiag[1]
    rhs: right hand side, (1,N)
    Note the logic is different from that of scipy.sparse.spdiags()
    """

    # TODO: check all inputs have same shape

    N = diag.shape[0]

    threads_per_tile = 6

    required_tiles = N // threads_per_tile
    n_on_last_tile = N % threads_per_tile

    tiles = [i for i in range(required_tiles) for _ in range(threads_per_tile)]
    if n_on_last_tile:
        tiles += [required_tiles for _ in range(n_on_last_tile)]

    d = tile_put_sharded(diag, tiles=tiles)

    ell = tile_put_sharded(ldiag, tiles=tiles)  # Flake8 complains about 'l'

    u = tile_put_sharded(udiag, tiles=tiles)

    b = tile_put_sharded(rhs, tiles=tiles)

    return d, u, ell, b


def ipu_tridiag_solve(diag: Array, udiag: Array, ldiag: Array, rhs: Array):
    """
    diag: main diagonal, (1,N)
    udiag: upper diagonal, (1,N), the last element is not used
    ldiag: lower diagonal, (1,N), the first element is not used, i.e. A[1,0] == ldiag[1]
    rhs: right hand side, (1,N)
    Note the logic is different from that of scipy.sparse.spdiags()
    """

    # TODO: check all inputs have same shape

    d, u, l, b = ipu_tridiag_solve_shard_inputs(diag, udiag, ldiag, rhs)

    x, _, _, _ = tile_map(tridiagonal_solver_p, d, u, l, b)  # type: ignore
    return x
