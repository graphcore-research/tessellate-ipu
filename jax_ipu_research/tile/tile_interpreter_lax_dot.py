# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import DTypeLike

from .tile_interpreter_primitives import from_numpy_dtype_to_ipu_type
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
        use_limited_ver: ?
        use_128Bit_load: 128 weights load?
        num_conv_units: 16?
        conv_input_load_elems: 2 (FP32) or 4 (FP16).
        disable_sr: Stochastic rounding config.
    """

    fp_dtype: DTypeLike
    accum_dtype: DTypeLike
    use_limited_ver: bool
    use_128bit_load: bool
    num_conv_units: int
    conv_input_load_elems: int
    disable_sr: bool

    def __post_init__(self):
        # TODO: check it has assembly implementation.
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
        assert self.in_chans_per_group % 16 == 0
        assert self.out_chans_per_group % 16 == 0


def make_conv1x1_partial_attributes(
    static_args: IpuConvPartial1x1StaticArgs, args: IpuConvPartial1x1Args
) -> List[IpuVertexAttributeI32]:
    """Build the IPU `ConvPartial1x1Out` vertex attributes

    Args:
        static_args: Vertex static/template arguments.
        args: Vertex arguments
    Returns:
        List properly encoded/transformed vertex attributes.
    """
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
    """Build an IPU `WorklistEntry` constant tensor."""
    # TODO: relax these conditions?
    assert out_offset >= 0
    assert num_field_elems >= 0
    assert in_offset >= 0
    return np.array([out_offset, num_field_elems, in_offset], dtype=np.uint32).astype(worklist_dtype)
