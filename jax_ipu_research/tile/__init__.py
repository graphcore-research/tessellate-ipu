# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register basic JAX primitives for tile interpreter + all additional libraries + paths.
from . import (
    external_libs,
    tile_interpreter_lax_binary,
    tile_interpreter_lax_dot,
    tile_interpreter_lax_reduce,
    tile_interpreter_lax_unary,
    tile_interpreter_linalg_qr,
)
from .tile_array import (
    TileShardedArray,
    tile_barrier,
    tile_constant_replicated,
    tile_constant_sharded,
    tile_data_barrier,
    tile_gather,
    tile_put_replicated,
    tile_put_sharded,
)
from .tile_common_utils import make_ipu_shaped_array
from .tile_interpreter import create_ipu_tile_primitive, register_ipu_tile_primitive, tile_map_primitive
from .tile_interpreter_hw_primitives import hw_cycle_count_p, ipu_hw_cycle_count
from .tile_interpreter_lax_binary import scaled_add_p, scaled_sub_p
from .tile_interpreter_lax_dot import IpuConvVertexType
from .tile_interpreter_lax_unary import tile_copy
from .tile_interpreter_linalg_jacobi import ipu_eigh
from .tile_interpreter_linalg_qr import ipu_qr
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_ipu_type_to_numpy_dtype,
    from_numpy_dtype_to_ipu_type,
    get_ipu_type_name,
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
