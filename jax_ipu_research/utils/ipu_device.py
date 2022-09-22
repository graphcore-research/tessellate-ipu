# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from dataclasses import dataclass

import jax


@dataclass
class IpuConfig:
    """IPU device configuration."""

    num_tiles: int = 4
    num_worker_contexts: int = 6
    version: str = "ipu2"
    is_model: bool = False


def get_ipu_config() -> IpuConfig:
    """Infer the IPU config used.

    TODO: proper support directly in Jaxlib.
    """
    # Make sure we have IPUs!
    assert len(jax.devices("ipu")) > 0
    POPLAR_FLAGS = os.getenv("TF_POPLAR_FLAGS")
    if POPLAR_FLAGS is not None and "--use_ipu_model" in POPLAR_FLAGS:
        # TODO: parse number of tiles.
        return IpuConfig(num_tiles=4, num_worker_contexts=6, is_model=True)
    # Assuming MK2 hardware, for now.
    return IpuConfig(num_tiles=1472, num_worker_contexts=6, is_model=False)
