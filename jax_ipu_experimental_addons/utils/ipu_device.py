# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from jaxlib.ipu_xla_client import IpuDevice, IpuPjRtDevice, IpuTargetType


def is_ipu_model(device: IpuDevice) -> bool:
    """Is the IPU JAX device an IPU model?"""
    # Support latest & legacy IPU device classes.
    return isinstance(device, (IpuDevice, IpuPjRtDevice)) and device.target_type == IpuTargetType.IPU_MODEL
