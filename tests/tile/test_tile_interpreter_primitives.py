# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import base64

import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from custom_arange_primitive import custom_multi_out_p
from jax import lax
from jax.core import Primitive, ShapedArray

from jax_ipu_experimental_addons.tile.tile_common_utils import Base64Data, IpuType
from jax_ipu_experimental_addons.tile.tile_interpreter_primitives import (
    from_ipu_type_to_numpy_dtype,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_attributes,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_inputs,
    make_ipu_vertex_io_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_outputs,
    primitive_has_batching,
    primitive_has_impl,
)
from jax_ipu_experimental_addons.tile.tile_interpreter_primitives_impl import (
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


class IpuTileEquationBaseTests(chex.TestCase, parameterized.TestCase):
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
            == '{"aval":{"dtype":12,"shape":[1,2,3]},"constant_data":null,"iotype":2,"name":"in0","slices2d":[]}'
        )

    def test__ipu_vertex_io_info__from_json_str__proper_representation(self):
        ioinfo = IpuVertexIOInfo.from_json_str(
            '{"aval":{"dtype":12,"shape":[1,2,3]},"constant_data":null,"iotype":2,"name":"in0","slices2d":[{"begin":10,"end":15}]}'
        )
        assert ioinfo.name == "in0"
        assert ioinfo.iotype == IpuVertexIOType.InOut
        assert ioinfo.shape == [1, 2, 3]
        assert ioinfo.dtype == IpuType.FLOAT
        assert len(ioinfo.slices2d) == 1

    def test__ipu_tile_map_equation__init__proper_fields(self):
        eqn = IpuTileMapEquation(
            tiles=[10], vname="vertex", pname="prim", attributes_f32=[IpuVertexAttributeF32("test", 2.5)]
        )
        assert eqn.tiles == [10]
        assert eqn.vname == "vertex"
        assert eqn.pname == "prim"
        assert eqn.attributes_f32 == [IpuVertexAttributeF32("test", 2.5)]
        assert not eqn.use_tmp_space
        assert eqn.tmp_space_aval.shape == [0]

    def test__make_ipu_vertex_io_info__proper_result(self):
        aval = ShapedArray([1, 2, 3], np.float16)
        info = make_ipu_vertex_io_info("input0", IpuVertexIOType.InOut, aval)
        assert isinstance(info, IpuVertexIOInfo)
        assert info.name == "input0"
        assert info.iotype == IpuVertexIOType.InOut
        assert info.shape == [1, 2, 3]
        assert info.dtype == IpuType.HALF
        assert not info.is_constant_input

    def test__make_ipu_vertex_constant_info__proper_result(self):
        datain = np.array([1, 2, 3, 4], dtype=np.float32)
        info = make_ipu_vertex_constant_info("constant", datain, vertex_dim2=2)
        assert isinstance(info, IpuVertexIOInfo)
        assert info.name == "constant"
        assert info.iotype == IpuVertexIOType.In
        assert tuple(info.shape) == datain.shape
        assert info.dtype == IpuType.FLOAT
        assert info.is_constant_input
        assert len(info.slices2d) == 2

        dataout = np.frombuffer(base64.decodebytes(str.encode(info.constant_data.encoded_data)), dtype=datain.dtype)
        npt.assert_array_equal(dataout, datain)

    def test__make_ipu_vertex_inputs__proper_results(self):
        inavals = {"in0": ShapedArray((3, 2), np.float16), "in1": ShapedArray((6,), np.uint8)}
        infos = make_ipu_vertex_inputs(inavals, {"in0"}, {"in1": 3})
        assert len(infos) == 2
        assert [v.dtype for v in infos] == [IpuType.HALF, IpuType.UNSIGNED_CHAR]
        assert [v.iotype for v in infos] == [IpuVertexIOType.InOut, IpuVertexIOType.In]
        assert [len(v.slices2d) for v in infos] == [0, 2]

    def test__make_ipu_vertex_ouputs__proper_results(self):
        inavals = {"in0": ShapedArray((3, 2), np.float16), "in1": ShapedArray((10,), np.uint8)}
        infos = make_ipu_vertex_outputs(inavals, {"in0"}, {"in1": 5})
        assert len(infos) == 2
        assert [v.dtype for v in infos] == [IpuType.HALF, IpuType.UNSIGNED_CHAR]
        assert [v.iotype for v in infos] == [IpuVertexIOType.InOut, IpuVertexIOType.Out]
        assert [len(v.slices2d) for v in infos] == [0, 2]

    @parameterized.parameters([np.float32, np.float16, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32])
    def test__from_numpy_dtype_to_ipu_type__proper_round_trip(self, in_dtype):
        ipu_type = from_numpy_dtype_to_ipu_type(in_dtype)
        out_dtype = from_ipu_type_to_numpy_dtype(ipu_type)
        assert out_dtype == in_dtype

    def test__from_numpy_dtype_to_ipu_type__handle_ipu_type_input(self):
        assert from_numpy_dtype_to_ipu_type(IpuType.CHAR) == IpuType.CHAR

    def test__from_ipu_type_to_numpy_dtype__handle_numpy_type_input(self):
        assert from_ipu_type_to_numpy_dtype(np.dtype(np.float32)) == np.dtype(np.float32)
        assert from_ipu_type_to_numpy_dtype(np.float32) == np.dtype(np.float32)

    def test__make_ipu_vertex_name_templated__proper_fullname(self):
        vname = make_ipu_vertex_name_templated("MyVertex", "subclass", np.float32, np.uint32, 3, False)
        assert vname == "MyVertex<subclass,float,unsigned int,3,false>"

    def test__make_ipu_vertex_attributes__proper_list_attributes(self):
        attrs_i32, attrs_f32 = make_ipu_vertex_attributes(k1=2, k2=3.0)
        assert attrs_i32[0] == IpuVertexAttributeI32("k1", 2)
        assert attrs_f32[0] == IpuVertexAttributeF32("k2", 3.0)


class IpuPrimitiveUtilsTests(chex.TestCase):
    def test__primitive_has_impl(self):
        assert primitive_has_impl(lax.add_p)
        assert primitive_has_impl(custom_multi_out_p)
        assert not primitive_has_impl(Primitive("test"))

    def test__primitive_has_batching(self):
        assert primitive_has_batching(lax.add_p)
        assert not primitive_has_batching(custom_multi_out_p)
