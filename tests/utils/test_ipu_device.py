# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import jax

from tessellate_ipu.utils import IpuTargetType, is_ipu_model


def test__is_ipu_model__proper_result():
    d = jax.devices("ipu")[0]
    assert is_ipu_model(d) == (d.target_type == IpuTargetType.IPU_MODEL)
