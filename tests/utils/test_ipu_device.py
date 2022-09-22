# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from jax_ipu_research.utils import IpuConfig, get_ipu_config


def test__get_ipu_config__proper_device():
    cfg = get_ipu_config()
    assert isinstance(cfg, IpuConfig)
    assert (cfg.num_tiles == 4 and cfg.is_model) or (cfg.num_tiles == 1472 and not cfg.is_model)
    assert cfg.num_worker_contexts == 6
