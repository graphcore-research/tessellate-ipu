# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TileMapMaxInOutAliasingArgs  # noqa: E402


def test__tesselate_jax__max_inout_aliasing_args():
    assert TileMapMaxInOutAliasingArgs == 4
