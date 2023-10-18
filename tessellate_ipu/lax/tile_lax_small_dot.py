# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict

import numpy as np
from jax.core import ShapedArray

from tessellate_ipu.core import declare_ipu_tile_primitive, make_ipu_vector1d_worker_offsets


def get_small_dot_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "../core", "vertex", "tile_small_dot.cpp")


@declare_ipu_tile_primitive("Rotation2dVertex", gp_filename=get_small_dot_vertex_gp_filename())
def rotation2d_p(cs: ShapedArray, inrow0: ShapedArray, inrow1: ShapedArray):
    """2d rotation apply primitive.

    Specific optimization on IPU backend compared to `dot_general_p` primitive.
    In particular, allows passing the 2 rows of the (2, N) input as separate arrays (in some
    applications, contiguous storage may not be possible).

    Args:
        cs: Cos/sin 2d rotation entries.
        inrow0: First row (N,)
        inrow1: Second row (N,)
    Returns:
        outrow0: First output row (N,)
        outrow1: Second output row (N,)
    """
    N = inrow0.size
    assert N % 2 == 0
    assert inrow0 == inrow1
    assert cs.dtype == inrow0.dtype
    assert cs.dtype == inrow1.dtype
    assert inrow0.dtype == np.float32

    outputs = {
        "outrow0": inrow0,
        "outrow1": inrow1,
    }
    constants = {"worker_offsets": make_ipu_vector1d_worker_offsets(N, vector_size=2, wdtype=np.uint16)}
    temps: Dict[str, Any] = {}
    perf_estimate = 100
    return outputs, constants, temps, perf_estimate
