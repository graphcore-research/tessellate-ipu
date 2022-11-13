from functools import partial

import chex
import jax
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_ipu_research.tile import TileShardedArray, tile_map_primitive, tile_put_sharded
from jax_ipu_research.tile.tile_interpreter_lax_reduce import make_continuous_reduce_vertex_fullname


def test__make_continuous_reduce_vertex_fullname__proper_name():
    fullname = make_continuous_reduce_vertex_fullname(jax.lax.reduce_prod_p, np.float32, np.float16, False)
    assert fullname == "popops::ContinuousReduce<popops::ReduceMul,float,half,false>"


class IpuTilePrimitivesLaxReduce(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.device = jax.devices("ipu")[0]
        self.num_tiles = self.device.num_tiles

    @parameterized.parameters([np.float32, np.float16])
    def test__tile_map_primitive__reduce__ipu_jitting__proper_result(self, dtype):
        tiles = (1, 2, self.num_tiles - 1)
        size = 100
        indata = np.random.randn(len(tiles), size).astype(dtype)
        reduce_p = jax.lax.reduce_sum_p

        def compute_fn(in0):
            input0 = tile_put_sharded(in0, tiles)
            output = tile_map_primitive(reduce_p, input0, axes=(0,))
            return output

        compute_fn_ipu = partial(jax.jit, backend="ipu")(compute_fn)
        output_ipu = compute_fn_ipu(indata)
        # compute_fn_cpu = partial(jax.jit, backend="cpu")(compute_fn)
        # output_cpu = compute_fn_cpu(indata)

        assert isinstance(output_ipu, TileShardedArray)
        assert output_ipu.tiles == tiles
        assert output_ipu.dtype == indata.dtype
        # TODO: compare to CPU backend JAX result.
        npt.assert_array_almost_equal(output_ipu.array, np.sum(indata, axis=1), decimal=2)
