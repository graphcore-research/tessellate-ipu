import chex
import numpy as np
from jax.core import ShapedArray

from jax_ipu_research.tile.tile_interpreter_primitives import make_ipu_vertex_io_info
from jax_ipu_research.tile.tile_interpreter_primitives_impl import (
    IpuTileMapEquation,
    IpuType,
    IpuVertexAttributeF32,
    IpuVertexIOInfo,
    IpuVertexIOType,
)


class IpuTileEquationBaseTests(chex.TestCase):
    def test__ipu_vertex_io_info__init__proper_fields(self):
        ioinfo = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        assert ioinfo.name == "in0"
        assert ioinfo.iotype == IpuVertexIOType.Out
        assert ioinfo.shape == [1, 2, 3]
        assert ioinfo.dtype == IpuType.FLOAT

    def test__ipu_vertex_io_info__eq__proper_results(self):
        ioinfo0 = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        ioinfo1 = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        ioinfo2 = IpuVertexIOInfo(name="in1", iotype=IpuVertexIOType.Out, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        assert ioinfo0 == ioinfo1
        assert ioinfo2 != ioinfo0

    def test__ipu_vertex_io_info__to_json_str__proper_representation(self):
        ioinfo = IpuVertexIOInfo(name="in0", iotype=IpuVertexIOType.InOut, shape=[1, 2, 3], dtype=IpuType.FLOAT)
        assert ioinfo.to_json_str() == '{"aval":{"dtype":12,"shape":[1,2,3]},"iotype":2,"name":"in0"}'

    def test__ipu_vertex_io_info__from_json_str__proper_representation(self):
        ioinfo = IpuVertexIOInfo.from_json_str('{"aval":{"dtype":12,"shape":[1,2,3]},"iotype":2,"name":"in0"}')
        assert ioinfo.name == "in0"
        assert ioinfo.iotype == IpuVertexIOType.InOut
        assert ioinfo.shape == [1, 2, 3]
        assert ioinfo.dtype == IpuType.FLOAT

    def test__ipu_tile_map_equation__init__proper_fields(self):
        eqn = IpuTileMapEquation(
            tiles=[10], vname="vertex", pname="prim", attributes_f32=[IpuVertexAttributeF32("test", 2.5)]
        )
        assert eqn.tiles == [10]
        assert eqn.vname == "vertex"
        assert eqn.pname == "prim"
        assert eqn.attributes_f32 == [IpuVertexAttributeF32("test", 2.5)]

    def test__make_ipu_vertex_io_info__proper_result(self):
        aval = ShapedArray([1, 2, 3], np.float16)
        info = make_ipu_vertex_io_info("input0", IpuVertexIOType.InOut, aval)
        assert isinstance(info, IpuVertexIOInfo)
        assert info.name == "input0"
        assert info.iotype == IpuVertexIOType.InOut
        assert info.shape == [1, 2, 3]
        assert info.dtype == IpuType.HALF
