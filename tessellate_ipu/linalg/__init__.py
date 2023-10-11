# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register JAX primitives for tile interpreter.
from . import tile_linalg_jacobi, tile_linalg_qr
from .tile_linalg_hessenberg import ipu_hessenberg
from .tile_linalg_jacobi import ipu_eigh
from .tile_linalg_qr import ipu_qr
from .tile_linalg_tridiagonal_solver import ipu_tridiag_solve
