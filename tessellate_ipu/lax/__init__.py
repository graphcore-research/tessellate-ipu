# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register JAX primitives for tile interpreter.
from . import tile_lax_binary, tile_lax_dot, tile_lax_reduce, tile_lax_unary, tile_random
from .tile_lax_binary import scaled_add_p, scaled_sub_p
from .tile_lax_dot import IpuConvVertexType
from .tile_lax_unary import tile_copy
from .tile_random import tile_get_hw_seeds, tile_random_normal, tile_random_uniform, tile_set_hw_seeds
