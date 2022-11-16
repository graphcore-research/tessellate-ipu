# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
from absl.testing import parameterized

from jax_ipu_research.tile.tile_interpreter_lax_dot import (
    IpuConvPartial1x1Args,
    IpuConvPartial1x1StaticArgs,
    ipuGetTransformedOutStride,
    ipuReverseTransformedOutStride,
    make_conv1x1_partial_attributes,
)


class IpuConvPartial1x1Utils(chex.TestCase, parameterized.TestCase):
    def test__ConvPartial1x1__vertex_fullname(self):
        conv_static_args = IpuConvPartial1x1StaticArgs(np.float32, np.float32, True, False, 16, 2, False)
        assert conv_static_args.vertex_name == "poplin::ConvPartial1x1Out<float,float,true,false,16,2,false>"

    def test__ConvPartial1x1__worklist_dtypes(self):
        conv_static_args = IpuConvPartial1x1StaticArgs(np.float32, np.float32, True, False, 16, 2, False)
        assert conv_static_args.worklist_dtype == np.uint16
        assert conv_static_args.worklist_num_field_dtype == np.int16

    def test__ipuGetTransformedOutStride__proper_result(self):
        out_stride0 = 4
        trans_out_stride = ipuGetTransformedOutStride(
            outStride=out_stride0, outChansPerGroup=8, numConvUnitsRequired=16, isPartialsFloat=True, flipOut=True
        )
        out_stride1 = ipuReverseTransformedOutStride(trans_out_stride, True, numConvUnits=16, outChansPerGroup=8)
        assert out_stride1[0]
        assert out_stride1[1] == out_stride0

    def test__make_conv1x1_partial_attributes__proper_values(self):
        conv_static_args = IpuConvPartial1x1StaticArgs(np.float32, np.float32, True, False, 16, 2, False)
        conv_args = IpuConvPartial1x1Args(
            num_conv_groups=1,
            num_out_groups=2,
            num_in_groups=3,
            out_stride=1,
            in_stride=1,
            out_chans_per_group=32,
            in_chans_per_group=16,
            out_flip=True,
        )
        attrs = make_conv1x1_partial_attributes(conv_static_args, conv_args)
        assert len(attrs) == 7
        attrs_dict = {v.name: v.value for v in attrs}
        assert attrs_dict["numConvGroupsM1"] == 0
        assert attrs_dict["numOutGroupsM1"] == 1
        assert attrs_dict["numInGroups"] == 3
        assert attrs_dict["outChansPerGroup"] == 32
        assert attrs_dict["inChansPerGroup"] == 16
