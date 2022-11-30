# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from jax.core import Primitive, ShapedArray
from jax.lax import dot_general_p
from numpy.typing import DTypeLike

from .tile_interpreter import register_ipu_tile_primitive
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_in_info,
    make_ipu_vertex_out_info,
)
from .tile_interpreter_primitives_impl import (  # noqa: F401
    IpuVertexAttributeI32,
    ipuGetTransformedInRowStride,
    ipuGetTransformedInStride,
    ipuGetTransformedOutStride,
    ipuReverseTransformedInRowStride,
    ipuReverseTransformedInStride,
    ipuReverseTransformedOutStride,
)


@dataclass
class IpuConvPartial1x1StaticArgs:
    """IPU `ConvPartial1x1Out` vertex template/static arguments.

    Args:
        fp_dtype: Input floating dtype.
        accum_dtype: Accumulation dtype.
        use_limited_ver: Use assembly optimized vertex.
        use_128Bit_load: 128 weights load?
        num_conv_units: 16 or 8, depending on the dtypes?
        conv_input_load_elems: 2 (FP32) or 4 (FP16).
        disable_sr: Stochastic rounding config.
    """

    fp_dtype: DTypeLike
    accum_dtype: DTypeLike
    use_limited_ver: bool = True
    use_128bit_load: bool = False
    num_conv_units: int = 0
    conv_input_load_elems: int = 0
    disable_sr: bool = False

    def __post_init__(self):
        # Dtypes making sense!
        assert self.fp_dtype in (np.float16, np.float32)
        assert self.accum_dtype in (np.float16, np.float32)
        assert not (self.accum_dtype == np.float16 and self.fp_dtype == np.float32)

        # Deduce from input and accum dtypes the proper parameters.
        if self.conv_input_load_elems == 0:
            self.conv_input_load_elems = 2 if self.fp_dtype == np.float32 else 4
        if self.num_conv_units == 0:
            if self.accum_dtype == np.float16 and self.fp_dtype == np.float32:
                self.num_conv_units = 16
            else:
                self.num_conv_units = 8

        # TODO: check it has assembly implementation.
        if self.use_limited_ver:
            pass

    @property
    def vertex_name(self) -> str:
        """Make the full vertex name."""

        def bool_str(b: bool) -> str:
            return str(b).lower()

        fp_dtype_ipu = from_numpy_dtype_to_ipu_type(self.fp_dtype).name.lower()
        accum_dtype_ipu = from_numpy_dtype_to_ipu_type(self.accum_dtype).name.lower()
        return f"poplin::ConvPartial1x1Out<{fp_dtype_ipu},{accum_dtype_ipu},{bool_str(self.use_limited_ver)},{bool_str(self.use_128bit_load)},{self.num_conv_units},{self.conv_input_load_elems},{bool_str(self.disable_sr)}>"

    @property
    def worklist_dtype(self) -> DTypeLike:
        return np.uint16 if self.use_limited_ver else np.uint32

    @property
    def worklist_num_field_dtype(self) -> DTypeLike:
        return np.int16 if self.use_limited_ver else np.int32


@dataclass
class IpuConvPartial1x1Args:
    """IPU `ConvPartial1x1Out` arguments.

    Args:
        num_conv_groups: Number of convolution groups.
        num_out_groups: Number of out groups.
        num_in_groups: Number of in groups.
        out_stride: Output stride (NOT supported).
        in_stride: Input stride.
        out_chans_per_group: Output channels per group. Multiple of 16?
        in_chans_per_group: Input channels per group. 16?
        out_flip: Flipping output matrix rows.
    """

    num_conv_groups: int
    num_out_groups: int
    num_in_groups: int
    out_stride: int
    in_stride: int
    out_chans_per_group: int
    in_chans_per_group: int
    out_flip: bool

    def __post_init__(self):
        # No output stride supported.
        assert self.out_stride == 1
        # Multiples of 8 or 16.
        assert self.in_chans_per_group % 8 == 0
        assert self.out_chans_per_group % 8 == 0


def make_conv1x1_partial_attributes(
    static_args: IpuConvPartial1x1StaticArgs, args: IpuConvPartial1x1Args
) -> List[IpuVertexAttributeI32]:
    """Build the IPU `ConvPartial1x1Out` vertex attributes.

    NOTE: we are currently checking input and output sizes such we can always
    run the matmul on the IPU AMP unit (using the proper assembly vertex).

    Args:
        static_args: Vertex static/template arguments.
        args: Vertex arguments
    Returns:
        List properly encoded/transformed vertex attributes.
    """
    if static_args.fp_dtype == np.float32 and args.in_chans_per_group != 8:
        raise ValueError(f"`in_chans_per_group` should be equal to 8 for FP32, not '{args.in_chans_per_group}'.")
    if static_args.fp_dtype == np.float16 and args.in_chans_per_group != 16:
        raise ValueError(f"`in_chans_per_group` should be equal to 16 for FP16, not '{args.in_chans_per_group}'.")

    is_partial_float = static_args.accum_dtype == np.dtype(np.float32)
    # Vertex raw stride attributes require a bit of transformation!
    trans_out_stride = ipuGetTransformedOutStride(
        outStride=args.out_stride,
        outChansPerGroup=args.out_chans_per_group,
        numConvUnitsRequired=static_args.num_conv_units,
        isPartialsFloat=is_partial_float,
        flipOut=args.out_flip,
    )
    trans_in_stride = ipuGetTransformedInStride(
        convUnitWeightHeight=0,  # ampKernelHeight = 0 default.
        inRowStride=0,  # inRowStride = 0 default.
        inStride=args.in_stride,
        convInputLoadElems=static_args.conv_input_load_elems,
        inChansPerGroup=args.in_chans_per_group,
    )
    attrs_i32 = [
        IpuVertexAttributeI32("numConvGroupsM1", args.num_conv_groups - 1),
        IpuVertexAttributeI32("numOutGroupsM1", args.num_out_groups - 1),
        IpuVertexAttributeI32("numInGroups", args.num_in_groups),
        IpuVertexAttributeI32("transformedOutStride", trans_out_stride),
        IpuVertexAttributeI32("transformedInStride", trans_in_stride),
        IpuVertexAttributeI32("outChansPerGroup", args.out_chans_per_group),
        IpuVertexAttributeI32("inChansPerGroup", args.in_chans_per_group),
    ]
    return attrs_i32


def make_conv1x1_partial_worklist_entry(
    out_offset: int, num_field_elems: int, in_offset: int, worklist_dtype: DTypeLike
) -> np.ndarray:
    """Build an IPU `WorklistEntry` constant tensor, to feed to the `Conv1x1PartialOut` vertex."""
    # TODO: relax these conditions?
    assert out_offset >= 0
    # `num_field_elems` bias?
    assert num_field_elems >= -2
    assert in_offset >= 0
    return np.array([out_offset, num_field_elems, in_offset], dtype=np.uint32).astype(worklist_dtype)


def ipu_dot_general_primitive_translation(
    p: Primitive,
    tiles: Tuple[int, ...],
    inavals: List[ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU `dot_general` primitive translation rule to IPU vertex.

    NOTE: for now, we are restricting the implementation to only support inputs compatible
    with the IPU optimized AMP vertex (as an incentive for users to write performant tile code!)

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: (unused) attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 2
    assert attributes is not None
    num_context_workers = 6
    lhs_aval, rhs_aval = inavals
    assert lhs_aval.dtype == rhs_aval.dtype
    outaval = p.abstract_eval(*inavals, **attributes)[0]

    ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)) = attributes["dimension_numbers"]
    # Only last dimension contracting supported.
    assert lhs_contracting_dims == [lhs_aval.ndim - 1]
    assert rhs_contracting_dims == [rhs_aval.ndim - 1]
    # No batching supported.
    assert len(lhs_batch_dims) == 0
    assert len(rhs_batch_dims) == 0
    # Accumulator/output dtype.
    accum_dtype = attributes.get("preferred_element_type", None) or lhs_aval.dtype

    static_args = IpuConvPartial1x1StaticArgs(
        fp_dtype=lhs_aval.dtype,
        accum_dtype=accum_dtype,
        use_limited_ver=True,  # Use AMP unit assembly vertex.
        use_128bit_load=False,
        disable_sr=False,
    )
    # TODO: support more complex config?
    args = IpuConvPartial1x1Args(
        num_conv_groups=1,
        num_out_groups=1,
        num_in_groups=1,
        out_stride=1,
        in_stride=1,
        out_chans_per_group=outaval.shape[-1],
        in_chans_per_group=lhs_aval.shape[-1],
        out_flip=False,
    )
    vname = static_args.vertex_name
    attrs_i32 = make_conv1x1_partial_attributes(static_args, args)
    # By convention, subtract 3 to get `num_field_elems`?!
    worklist_data = make_conv1x1_partial_worklist_entry(
        out_offset=0, num_field_elems=lhs_aval.shape[0] - 3, in_offset=0, worklist_dtype=static_args.worklist_dtype
    )
    worklists_data = np.stack([worklist_data for _ in range(num_context_workers)]).ravel()

    ipu_prim_info = IpuTileMapEquation(
        vname=vname,
        pname=p.name,
        tiles=tiles,
        inputs_info=[
            make_ipu_vertex_in_info("in", lhs_aval, vertex_dim2=lhs_aval.size),
            make_ipu_vertex_in_info("weights", rhs_aval, vertex_dim2=rhs_aval.size),
            make_ipu_vertex_constant_info("worklists", worklists_data),
        ],
        outputs_info=[make_ipu_vertex_out_info("out", outaval, vertex_dim2=outaval.size)],
        attributes_i32=attrs_i32,
        attributes_f32=[],
    )
    return ipu_prim_info


# Register JAX dot_general op for `tile_map``.
register_ipu_tile_primitive(dot_general_p, ipu_dot_general_primitive_translation)
