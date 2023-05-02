# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import jax

from jax_ipu_experimental_addons.utils import IpuTargetType, is_ipu_model


def test__is_ipu_model__proper_result():
    d = jax.devices("ipu")[0]
    assert is_ipu_model(d) == (d.target_type == IpuTargetType.IPU_MODEL)
