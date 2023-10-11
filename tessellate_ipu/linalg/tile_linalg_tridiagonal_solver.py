import os.path as osp
from typing import Any

import jax

from tessellate_ipu import create_ipu_tile_primitive, tile_map, tile_put_replicated

jax.config.FLAGS.jax_platform_name = "cpu"

Array = Any


vertex_filename = osp.join(osp.dirname(__file__), "../core", "vertex", "tile_tridiagonal_solver_vertex.cpp")

tridiagonal_solver_p = create_ipu_tile_primitive(
    "tridiagonal_solver",
    "TridiagonalSolverVertex",
    inputs=["ts", "tus", "tls", "b"],
    outputs={"ts": 0},
    gp_filename=vertex_filename,
    perf_estimate=100,
)


def ipu_tridiag_solve(diag: Array, ldiag: Array, rhs: Array):

    tiles = [100]

    ts = tile_put_replicated(diag, tiles=tiles)

    tls = tile_put_replicated(ldiag, tiles=tiles)

    tus = tls

    b = tile_put_replicated(rhs, tiles=tiles)

    x_ = tile_map(tridiagonal_solver_p, ts, tus, tls, b)
    return x_
