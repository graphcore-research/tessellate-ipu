# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from jaxlib.ipu_xla_client import IpuDevice, IpuTargetType


def is_ipu_model(device: IpuDevice) -> bool:
    """Is the IPU JAX device an IPU model?"""
    return isinstance(device, IpuDevice) and device.target_type == IpuTargetType.IPU_MODEL
