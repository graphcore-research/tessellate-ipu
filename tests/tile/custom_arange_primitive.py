# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from jax import core

from jax_ipu_experimental_addons.tile import (
    IpuTileMapEquation,
    declare_ipu_tile_primitive,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_in_info,
    make_ipu_vertex_out_info,
    register_ipu_tile_primitive,
)
from jax_ipu_experimental_addons.utils import DTypeLike

custom_vertex_filename = os.path.join(os.path.dirname(__file__), "custom_arange_vertex.cpp")

custom_arange_p = core.Primitive("custom_arange")


def custom_arange(scales, size: int, dtype: DTypeLike):
    return custom_arange_p.bind(size=size, dtype=dtype)


def custom_arange_numpy_impl(scales, size: int, dtype: DTypeLike):
    # Artificial complexity to test a 2D input array!
    return np.arange(size).astype(dtype) * scales[0] * scales[1]


def custom_arange_abstract_eval(scales, size: int, dtype: DTypeLike):
    return core.ShapedArray((size,), dtype)


def custom_arange_tile_translation_ipu(
    p: core.Primitive,
    tiles: Tuple[int, ...],
    inavals: List[core.ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU tile translation for custom arange vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: Op attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    assert attributes is not None
    # Output shape.
    outshape = (attributes["size"],)
    outdtype = attributes["dtype"]
    outaval = core.ShapedArray(outshape, outdtype)
    gp_filename = custom_vertex_filename

    global_scale_data = np.array([7], dtype=outdtype)
    ipu_dtype = from_numpy_dtype_to_ipu_type(outdtype)
    vertex_name = f"CustomArangeVertex<{ipu_dtype.name.lower()}>"
    # Translation rule to IPU vertex
    ipu_prim_info = IpuTileMapEquation(
        vname=vertex_name,
        pname=p.name,
        tiles=tiles,
        # IO vertex infos.
        inputs_info=[
            make_ipu_vertex_in_info("scales", inavals[0], vertex_dim2=inavals[0].shape[1]),
            make_ipu_vertex_constant_info("global_scale", global_scale_data),
        ],
        outputs_info=[make_ipu_vertex_out_info("out", outaval)],
        # Additional attributes to pass to the vertex
        attributes_i32=[],
        attributes_f32=[],
        # Optional GP filename and perf. estimate.
        gp_filename=gp_filename,
        perf_estimate=outshape[0] + 5,
    )
    return ipu_prim_info


custom_arange_p.map_primitive = False
# Register the primal implementation with JAX
custom_arange_p.def_impl(custom_arange_numpy_impl)
# Register the abstract evaluation with JAX
custom_arange_p.def_abstract_eval(custom_arange_abstract_eval)
# Register tile IPU translation.
register_ipu_tile_primitive(custom_arange_p, custom_arange_tile_translation_ipu)


# Declaring a tile primitive in a very simple & fast way.
@declare_ipu_tile_primitive("CustomSingleOutVertex<{input}>", gp_filename=custom_vertex_filename)
def custom_single_out_p(input):
    outputs = {"output": input}
    perf_estimate = 100
    return outputs, None, None, perf_estimate


# Provide a JAX NumPy implementation for other backends (CPU/GPU/TPU)
def custom_single_out_impl(x):
    return -x


custom_single_out_p.def_impl(custom_single_out_impl)


# Declaring a tile primitive in a very simple & fast way.
@declare_ipu_tile_primitive("CustomMultiOutVertex<{input}>", gp_filename=custom_vertex_filename)
def custom_multi_out_p(input):
    outputs = {"out0": input, "out1": input}
    constants = {"constant_scale": np.array([input.size], input.dtype)}
    tmp_space = {"mytmp": input}
    perf_estimate = 100
    return outputs, constants, tmp_space, perf_estimate


# Provide a JAX NumPy implementation for other backends (CPU/GPU/TPU)
def custom_multi_out_impl(x, scale_value):
    constant_scale = np.array([x.size], x.dtype)
    mytmp = constant_scale[0] * scale_value * x
    return (mytmp, -mytmp)


custom_multi_out_p.def_impl(custom_multi_out_impl)
