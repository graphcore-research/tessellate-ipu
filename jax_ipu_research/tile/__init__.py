# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register basic JAX primitives for tile interpreter.
from . import (
    external_libs,
    tile_interpreter_lax_binary,
    tile_interpreter_lax_dot,
    tile_interpreter_lax_reduce,
    tile_interpreter_lax_unary,
    tile_interpreter_linalg,
)
from .tile_array import TileShardedArray, tile_put_replicated, tile_put_sharded
from .tile_interpreter import create_simple_tile_primitive, register_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_hw_primitives import hw_cycle_count_p
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_ipu_type_to_numpy_dtype,
    from_numpy_dtype_to_ipu_type,
    make_ipu_shaped_array,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_in_info,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_inputs,
    make_ipu_vertex_io_info,
    make_ipu_vertex_out_info,
    make_ipu_vertex_outputs,
)
from .tile_interpreter_random import (
    ipu_get_hw_seeds_tmap,
    ipu_random_normal_tmap,
    ipu_random_uniform_tmap,
    ipu_set_hw_seeds_tmap,
)
