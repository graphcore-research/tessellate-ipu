# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import chex
from absl.testing import parameterized

from tessellate_ipu.core.tile_common_utils import Base64Data, IpuType
from tessellate_ipu.lib.pytessellate_ipu_core import (
    IpuTensorSlice,
    IpuTileMapEquation,
    IpuVertexAttributeF32,
    IpuVertexAttributeI32,
    IpuVertexIOInfo,
    IpuVertexIOType,
)


class IpuTensorSliceTests(chex.TestCase, parameterized.TestCase):
    def test__tensor_slice__init__proper_data(self):
        slice = IpuTensorSlice(begin=10, end=15)
        assert slice.begin == 10
        assert slice.end == 15

    def test__tensor_slice__to_json_str(self):
        slice = IpuTensorSlice(begin=10, end=15)
        assert slice.to_json_str() == '{"begin":10,"end":15}'

    def test__tensor_slice__from_json_str(self):
        slice = IpuTensorSlice.from_json_str('{"begin":10,"end":15}')
        assert slice.begin == 10
        assert slice.end == 15


class IpuVertexIOTests(chex.TestCase, parameterized.TestCase):
    def test__ipu_vertex_io_info__init__proper_fields(self):
        ioinfo = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.In, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        assert ioinfo.name == "in0"
        assert ioinfo.iotype == IpuVertexIOType.In
        assert ioinfo.shape == [1, 2, 3]
        assert ioinfo.dtype == IpuType.FLOAT
        assert not ioinfo.is_constant_input
        assert ioinfo.constant_data.encoded_data == ""

    def test__ipu_vertex_io_info__constant__proper_fields(self):
        ioinfo = IpuVertexIOInfo(
            name="in0",
            iotype=IpuVertexIOType.In,
            shape=[1, 2, 3],
            dtype=IpuType.FLOAT,
            constant_data=Base64Data("12345"),
        )
        assert ioinfo.name == "in0"
        assert ioinfo.iotype == IpuVertexIOType.In
        assert ioinfo.shape == [1, 2, 3]
        assert ioinfo.dtype == IpuType.FLOAT
        assert ioinfo.is_constant_input
        assert ioinfo.constant_data.encoded_data == "12345"

    def test__ipu_vertex_io_info__eq__proper_results(self):
        ioinfo0 = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        ioinfo1 = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        ioinfo2 = IpuVertexIOInfo(name="in1", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        assert ioinfo0 == ioinfo1
        assert ioinfo2 != ioinfo0
        assert not ioinfo0.is_constant_input
        assert not ioinfo2.is_constant_input

    def test__ipu_vertex_io_info__to_json_str__proper_representation(self):
        ioinfo = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.InOut, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        assert (
            ioinfo.to_json_str()
            == '{"aval":{"dtype":12,"shape":[1,2,3]},"constant_data":null,"iotype":2,"is_scalar":false,"name":"in0","slices2d":[]}'
        )

    def test__ipu_vertex_io_info__from_json_str__proper_representation(self):
        ioinfo = IpuVertexIOInfo.from_json_str(
            '{"aval":{"dtype":12,"shape":[1,2,3]},"constant_data":null,"iotype":2,"name":"in0","slices2d":[{"begin":10,"end":15}],"is_scalar":false}'
        )
        assert ioinfo.name == "in0"
        assert ioinfo.iotype == IpuVertexIOType.InOut
        assert ioinfo.shape == [1, 2, 3]
        assert ioinfo.dtype == IpuType.FLOAT
        assert len(ioinfo.slices2d) == 1


class IpuTileEquationBaseTests(chex.TestCase, parameterized.TestCase):
    def test__ipu_tile_map_equation__init__proper_fields(self):
        eqn = IpuTileMapEquation(
            tiles=[10], vname="vertex", pname="prim", attributes_f32=[IpuVertexAttributeF32("test", 2.5)]
        )
        assert eqn.tiles == [10]
        assert eqn.vname == "vertex"
        assert eqn.pname == "prim"
        assert eqn.attributes_i32 == []
        assert eqn.attributes_f32 == [IpuVertexAttributeF32("test", 2.5)]
        assert not eqn.use_tmp_space
        assert eqn.tmp_space_aval.shape == [0]

    def test__ipu_tile_map_equation__num_inputs_outputs_properties(self):
        eqn = IpuTileMapEquation(
            tiles=[10],
            vname="vertex",
            pname="prim",
            inputs_info=[
                IpuVertexIOInfo(name="inout", iotype=IpuVertexIOType.InOut, shape=[5], dtype=IpuType.FLOAT),
                IpuVertexIOInfo(name="in", iotype=IpuVertexIOType.In, shape=[5], dtype=IpuType.FLOAT),
            ],
            outputs_info=[
                IpuVertexIOInfo(name="inout", iotype=IpuVertexIOType.InOut, shape=[5], dtype=IpuType.FLOAT),
                IpuVertexIOInfo(name="out1", iotype=IpuVertexIOType.Out, shape=[5], dtype=IpuType.FLOAT),
                IpuVertexIOInfo(name="out2", iotype=IpuVertexIOType.Out, shape=[5], dtype=IpuType.FLOAT),
            ],
            attributes_i32=[IpuVertexAttributeI32("test", 3)],
        )
        assert eqn.tiles == [10]
        assert eqn.vname == "vertex"
        assert eqn.pname == "prim"
        assert eqn.attributes_i32 == [IpuVertexAttributeI32("test", 3)]
        # Consistent number of inputs/outputs/inouts?
        assert len(eqn.inputs_info) == 2
        assert len(eqn.outputs_info) == 3

        assert eqn.num_inputs == 2
        assert eqn.num_outputs == 3
        assert eqn.num_inouts == 1
