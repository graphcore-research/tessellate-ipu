# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax.lax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_ipu_experimental_addons.tile import IpuConvVertexType, TileShardedArray, tile_map_primitive, tile_put_sharded
from jax_ipu_experimental_addons.tile.tile_interpreter_lax_dot import (
    IpuConvPartial1x1Args,
    IpuConvPartial1x1StaticArgs,
    ipuGetTransformedOutStride,
    ipuReverseTransformedOutStride,
    make_conv_partial1x1_attributes,
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

    def test__make_conv_partial1x1_attributes__proper_values(self):
        conv_static_args = IpuConvPartial1x1StaticArgs(np.float32, np.float32, True, False, 16, 2, False)
        conv_args = IpuConvPartial1x1Args(
            num_conv_groups=1,
            num_out_groups=2,
            num_in_groups=3,
            out_stride=1,
            in_stride=1,
            out_chans_per_group=32,
            in_chans_per_group=8,
            out_flip=True,
        )
        attrs = make_conv_partial1x1_attributes(conv_static_args, conv_args)
        assert len(attrs) == 7
        attrs_dict = {v.name: v.value for v in attrs}
        assert attrs_dict["numConvGroupsM1"] == 0
        assert attrs_dict["numOutGroupsM1"] == 1
        assert attrs_dict["numInGroups"] == 3
        assert attrs_dict["outChansPerGroup"] == 32
        assert attrs_dict["inChansPerGroup"] == 8


class IpuConvPartial1x1DotPrimitive(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Not very clean, but for better reproducibility.
        np.random.seed(42)

    @parameterized.parameters(
        # Basic AMP unit config, without any in/out "groups"
        {"lhs_size": 1, "rhs_size": 8, "contract_size": 8, "indtype": np.float32, "accdtype": np.float32},
        {"lhs_size": 7, "rhs_size": 16, "contract_size": 8, "indtype": np.float32, "accdtype": np.float32},
        {"lhs_size": 17, "rhs_size": 128, "contract_size": 8, "indtype": np.float32, "accdtype": np.float32},
        # Float16, with different accumulator types.
        {"lhs_size": 15, "rhs_size": 32, "contract_size": 16, "indtype": np.float16, "accdtype": np.float32},
        {"lhs_size": 11, "rhs_size": 64, "contract_size": 16, "indtype": np.float16, "accdtype": np.float16},
        # Large size matmul, requiring num_in_groups > 1
        # {"lhs_size": 11, "rhs_size": 64, "contract_size": 32, "indtype": np.float16, "accdtype": np.float32},
        # {"lhs_size": 256, "rhs_size": 256, "contract_size": 256, "indtype": np.float32, "accdtype": np.float32},
    )
    def test__dot_general__conv_partial1x1__ipu_jitting(self, lhs_size, rhs_size, contract_size, indtype, accdtype):
        tiles = (0, 3)
        lhs_data = np.random.randn(len(tiles), lhs_size, contract_size).astype(indtype)
        rhs_data = np.random.randn(len(tiles), rhs_size, contract_size).astype(indtype)

        def dot_general_fn(lhs, rhs):
            lhs = tile_put_sharded(lhs, tiles)
            rhs = tile_put_sharded(rhs, tiles)
            output = tile_map_primitive(
                jax.lax.dot_general_p,
                lhs,
                rhs,
                dimension_numbers=(([1], [1]), ([], [])),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=accdtype,
                ipu_vertex_type=IpuConvVertexType.ConvPartial1x1,
            )
            return output

        dot_general_fn_ipu = partial(jax.jit, backend="ipu")(dot_general_fn)
        output_ipu = dot_general_fn_ipu(lhs_data, rhs_data)
        dot_general_fn_cpu = partial(jax.jit, backend="cpu")(dot_general_fn)
        output_cpu = dot_general_fn_cpu(lhs_data, rhs_data)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == accdtype
        assert output_ipu.shape == output_cpu.shape
        npt.assert_array_almost_equal(output_ipu.array, output_cpu, decimal=2)

    @parameterized.parameters(
        # Power of two lhs and rhs size.
        {"lhs_size": 7, "rhs_size": 16, "contract_size": 7, "indtype": np.float32, "accdtype": np.float32},
        {"lhs_size": 7, "rhs_size": 15, "contract_size": 8, "indtype": np.float32, "accdtype": np.float32},
        # AMP vertex size.
        {"lhs_size": 7, "rhs_size": 16, "contract_size": 16, "indtype": np.float32, "accdtype": np.float32},
        {"lhs_size": 7, "rhs_size": 16, "contract_size": 32, "indtype": np.float16, "accdtype": np.float16},
    )
    def test__dot_general__conv_partial1x1__input_errors(self, lhs_size, rhs_size, contract_size, indtype, accdtype):
        tiles = (0, 3)
        lhs_data = np.random.randn(len(tiles), lhs_size, contract_size).astype(indtype)
        rhs_data = np.random.randn(len(tiles), rhs_size, contract_size).astype(indtype)

        @partial(jax.jit, backend="ipu")
        def dot_general_fn(lhs, rhs):
            lhs = tile_put_sharded(lhs, tiles)
            rhs = tile_put_sharded(rhs, tiles)
            output = tile_map_primitive(
                jax.lax.dot_general_p,
                lhs,
                rhs,
                dimension_numbers=(([1], [1]), ([], [])),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=accdtype,
            )
            return output

        with self.assertRaises(Exception):
            dot_general_fn(lhs_data, rhs_data)

    @parameterized.parameters(
        # Basic HMAC unit.
        # {"lhs_shape": (1,), "rhs_shape": (1,), "indtype": np.float32, "accdtype": np.float32},
        {"lhs_shape": (2,), "rhs_shape": (2,), "indtype": np.float32, "accdtype": np.float32},
        {"lhs_shape": (4,), "rhs_shape": (4,), "indtype": np.float32, "accdtype": np.float32},
        {"lhs_shape": (16,), "rhs_shape": (16,), "indtype": np.float32, "accdtype": np.float32},
        {"lhs_shape": (64,), "rhs_shape": (64,), "indtype": np.float32, "accdtype": np.float32},
        {"lhs_shape": (128,), "rhs_shape": (128,), "indtype": np.float32, "accdtype": np.float32},
    )
    def test__dot_general__conv_partial_hmac__ipu_jitting(self, lhs_shape, rhs_shape, indtype, accdtype):
        tiles = (0, 3)
        lhs_data = np.random.randn(len(tiles), *lhs_shape).astype(indtype)
        rhs_data = np.random.randn(len(tiles), *rhs_shape).astype(indtype)

        def dot_general_fn(lhs, rhs):
            lhs = tile_put_sharded(lhs, tiles)
            rhs = tile_put_sharded(rhs, tiles)
            output = tile_map_primitive(
                jax.lax.dot_general_p,
                lhs,
                rhs,
                dimension_numbers=(([len(lhs_shape) - 1], [len(rhs_shape) - 1]), ([], [])),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=accdtype,
                ipu_vertex_type=IpuConvVertexType.ConvPartialHMAC,
            )
            return output

        dot_general_fn_ipu = partial(jax.jit, backend="ipu")(dot_general_fn)
        output_ipu = dot_general_fn_ipu(lhs_data, rhs_data)
        dot_general_fn_cpu = partial(jax.jit, backend="cpu")(dot_general_fn)
        output_cpu = dot_general_fn_cpu(lhs_data, rhs_data)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == accdtype
        assert output_ipu.shape == output_cpu.shape
        npt.assert_array_almost_equal(output_ipu.array, output_cpu, decimal=2)

    @parameterized.parameters(
        # Basic AMP unit config, without any in/out "groups"
        # {"lhs_shape": (8, 8), "rhs_shape": (1, 8), "indtype": np.float32, "accdtype": np.float32},
        # {"lhs_shape": (8, 8), "rhs_shape": (8, ), "indtype": np.float32, "accdtype": np.float32},
        {"lhs_shape": (1, 8), "rhs_shape": (8, 8), "indtype": np.float32, "accdtype": np.float32},
    )
    def test__dot_matrix_vector__amp_slic_vertices__ipu_jitting(self, lhs_shape, rhs_shape, indtype, accdtype):
        tiles = (0, 3)
        lhs_data = np.random.randn(len(tiles), *lhs_shape).astype(indtype)
        rhs_data = np.random.randn(len(tiles), *rhs_shape).astype(indtype)

        def dot_general_fn(lhs, rhs):
            lhs = tile_put_sharded(lhs, tiles)
            rhs = tile_put_sharded(rhs, tiles)
            output = tile_map_primitive(
                jax.lax.dot_general_p,
                lhs,
                rhs,
                dimension_numbers=(([len(lhs_shape) - 1], [len(rhs_shape) - 1]), ([], [])),
                precision=None,
                preferred_element_type=None,
            )
            return output

        dot_general_fn_ipu = partial(jax.jit, backend="ipu")(dot_general_fn)
        output_ipu = dot_general_fn_ipu(lhs_data, rhs_data)
        dot_general_fn_cpu = partial(jax.jit, backend="cpu")(dot_general_fn)
        output_cpu = dot_general_fn_cpu(lhs_data, rhs_data)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == accdtype
        assert output_ipu.shape == output_cpu.shape
        npt.assert_array_almost_equal(output_ipu.array, output_cpu, decimal=2)
