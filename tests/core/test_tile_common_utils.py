# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import base64

import chex
from absl.testing import parameterized

from tessellate_ipu.core.tile_common_utils import Base64Data, IpuShapedArray, IpuType


class Base64DataTests(chex.TestCase, parameterized.TestCase):
    def test__base64_data__empty_constructor(self):
        data = Base64Data()
        assert data.encoded_data == ""
        assert data.is_empty

    def test__base64_data__data_constructor(self):
        data = Base64Data("123")
        assert data.encoded_data == "123"
        assert not data.is_empty

    def test__base64_data__from_decoded_data__factory(self):
        data = Base64Data.from_decoded_data("12345")
        assert data.encoded_data == "MTIzNDU="
        assert data.decoded_data == "12345"

    def test__base64_data__python_base64_compatibility(self):
        data = Base64Data(base64.b64encode(b"12345"))
        assert data.encoded_data == "MTIzNDU="
        assert data.decoded_data == "12345"

    def test__base64_data__from_json_str(self):
        data = Base64Data.from_json_str("{}")
        assert data.is_empty
        assert data.to_json_str() == "null"

        data = Base64Data.from_json_str("null")
        assert data.is_empty
        assert data.to_json_str() == "null"

        data = Base64Data.from_json_str('{"encoded_data":"12345"}')
        assert not data.is_empty
        assert data.encoded_data == "12345"
        assert data.to_json_str() == '{"encoded_data":"12345"}'


class IpuTypeTests(chex.TestCase, parameterized.TestCase):
    def test__ipu_type__proper_bytesize(self):
        assert IpuType.QUARTER.bytesize == 1
        assert IpuType.HALF.bytesize == 2
        assert IpuType.FLOAT.bytesize == 4


class IpuShapedArrayTests(chex.TestCase, parameterized.TestCase):
    def test__shaped_array__init__default_values(self):
        aval = IpuShapedArray()
        assert aval.shape == []
        assert aval.dtype == IpuType.UNSIGNED_CHAR
        assert aval.size == 1

    def test__shaped_array__init__arguments(self):
        aval = IpuShapedArray(shape=(1, 2, 3), dtype=IpuType.FLOAT)
        assert aval.shape == [1, 2, 3]
        assert aval.dtype == IpuType.FLOAT
        assert aval.size == 6
