# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple

import numpy as np
from jax.core import Primitive, ShapedArray
from jax.lax import GatherDimensionNumbers, GatherScatterMode, gather_p

from tessellate_ipu.core import (
    IpuTileMapEquation,
    make_ipu_vertex_attributes,
    make_ipu_vertex_in_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
    register_ipu_tile_primitive,
)
from tessellate_ipu.utils import DTypeLike


def make_gather_vertex_fullname(dtype: DTypeLike) -> str:
    """Generate popops Gather/MultiSlice vertex name."""
    basename = "popops::MultiSlice"
    return make_ipu_vertex_name_templated(basename, dtype)


def check_gather_dimension_numbers(dimension_numbers: GatherDimensionNumbers):
    """Check `gather` dimension_numbers is supported on TessellateIPU.

    At the moment: basically only supporting a single configuration!
    We need to expand on this at some point!
    """
    dim_numbers_default = GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))
    if dimension_numbers != dim_numbers_default:
        raise NotImplementedError(f"TessellateIPU `gather` only support dimension numbers: {dim_numbers_default}.")


def ipu_gather_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `gather` primitive translation rule to IPU vertex.

    See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.gather.html

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input data + start indices arrays.
        attributes: Gather operator attributes
    Returns:
        IPU tile map primitive structure.
    """
    # TODO: query for JAX device.
    num_context_workers = 6

    assert len(inavals) == 2
    assert attributes is not None
    operand, start_indices = inavals
    # Extract gather attributes
    dimension_numbers = attributes["dimension_numbers"]
    slice_sizes = attributes["slice_sizes"]
    # Default values from JAX LAX interface.
    indices_are_sorted = attributes.get("indices_are_sorted", False)
    unique_indices = attributes.get("unique_indices", False)
    mode = attributes.get("mode", GatherScatterMode.PROMISE_IN_BOUNDS)
    fill_value = attributes.get("fill_value", None)

    # Check gather attributes are supported by TessellateIPU.
    assert operand.ndim == 1
    assert start_indices.ndim == 2
    assert slice_sizes == (1,)
    assert (
        mode == GatherScatterMode.PROMISE_IN_BOUNDS
    ), "Only `PROMISE_IN_BOUNDS` gather mode supported in TessellateIPU."
    assert start_indices.dtype == np.uint32, "TessellateIPU `gather` only supports `uint32` indices."
    check_gather_dimension_numbers(dimension_numbers)
    # Gather output aval.
    outaval = p.abstract_eval(
        *inavals,
        dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        fill_value=fill_value,
    )[0]

    vname = make_gather_vertex_fullname(operand.dtype)
    # Construct poplibs MultiSlice vertex attributes.
    attrs_i32, attrs_f32 = make_ipu_vertex_attributes(
        baseOffset=0,  # unused?
        numBaseElements=operand.size,  # Number of elements in input.
        maxElementsPerWorker=int(np.ceil(start_indices.size / num_context_workers)),
        regionSize=1,  # TODO: understand?
        splitSingleRegion=False,  # Split regions between threads? TODO: understand!
    )
    # TODO: should we use `split offsets` between threads?
    # For now: need to do it manually at the Python `tile_map` level.
    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[make_ipu_vertex_in_info("baseT", operand), make_ipu_vertex_in_info("offsets", start_indices)],
        outputs_info=[make_ipu_vertex_out_info("subT", outaval)],
        attributes_i32=attrs_i32,
        attributes_f32=attrs_f32,
    )
    return ipu_prim_info


# Register JAX gather primitive.
register_ipu_tile_primitive(gather_p, ipu_gather_primitive_translation)
