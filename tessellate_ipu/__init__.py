# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register basic JAX primitives for tile interpreter + all additional libraries + paths.
from . import lax
from ._version import __version__
from .core import (
    TileShardedArray,
    create_ipu_tile_primitive,
    declare_ipu_tile_primitive,
    ipu_cycle_count,
    tile_constant_replicated,
    tile_constant_sharded,
    tile_data_barrier,
    tile_gather,
    tile_map,
    tile_put_replicated,
    tile_put_sharded,
)
from .utils import IpuDevice, IpuTargetType, is_ipu_model
