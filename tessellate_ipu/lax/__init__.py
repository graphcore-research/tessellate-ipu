# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register JAX primitives for tile interpreter.
from jax.lax import (
    abs_p,
    asin_p,
    cbrt_p,
    ceil_p,
    erf_p,
    exp_p,
    expm1_p,
    floor_p,
    is_finite_p,
    log1p_p,
    log_p,
    neg_p,
    population_count_p,
    round_p,
    rsqrt_p,
    select_n_p,
    sign_p,
    sin_p,
    sqrt_p,
    tan_p,
    tanh_p,
)

from . import tile_lax_binary, tile_lax_dot, tile_lax_reduce, tile_lax_unary, tile_random
from .tile_lax_array import bitcast_convert_type_p, reshape_p
from .tile_lax_binary import (
    add_inplace_p,
    atan2_inplace_p,
    div_inplace_p,
    max_inplace_p,
    min_inplace_p,
    mul_inplace_p,
    pow_inplace_p,
    rem_inplace_p,
    scaled_add_p,
    scaled_sub_p,
    sub_inplace_p,
)
from .tile_lax_cumulative_ops import cummax_p, cummin_p, cumprod_p, cumsum_p
from .tile_lax_dot import IpuConvVertexType
from .tile_lax_gather import gather_p
from .tile_lax_scatter import scatter_add_p, scatter_max_p, scatter_min_p, scatter_mul_p, scatter_p
from .tile_lax_unary import (  # tanh_inplace_p,
    abs_inplace_p,
    asin_inplace_p,
    cbrt_inplace_p,
    ceil_inplace_p,
    erf_inplace_p,
    exp_inplace_p,
    expm1_inplace_p,
    floor_inplace_p,
    is_finite_inplace_p,
    log1p_inplace_p,
    log_inplace_p,
    neg_inplace_p,
    population_count_inplace_p,
    round_inplace_p,
    rsqrt_inplace_p,
    sign_inplace_p,
    sin_inplace_p,
    sqrt_inplace_p,
    tan_inplace_p,
    tile_copy,
)
from .tile_random import tile_get_hw_seeds, tile_random_normal, tile_random_uniform, tile_set_hw_seeds
