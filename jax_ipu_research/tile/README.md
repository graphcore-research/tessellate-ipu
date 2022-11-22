# IPU basic tile API

The sub-module `jax_ipu_research.tile` provides basic tooling for programming the IPU at the low level directly in JAX (i.e. exposing some of the Poplar vertex API at the higher level).

## Tile mapping of an array

The first aspect is to be able to define tile mapping directly in JAX. For that purpose, two methods are provided (mimicking `jax.device_put`):
```python
arr = np.array(...)
tiles = (10, 15, 18)

# Tile shard an array over the first axis. Same shape.
arr_sharded = tile_put_sharded(arr, tiles)
# Tile replicate an array over a new first axis.
arr_replicated = tile_put_replicated(arr, tiles)
```
Both methods are returning a `TileShardedArray` object, where by convention, the tile sharding is always on the first axis (i.e. `arr_sharded[idx]` is located on a single tile, and is contiguous). (TODO: make sure these are no-op if the array is already properly sharded.)


## Tile primitives

Once a `TileShardedArray` object has been created, one can map a JAX primitive at the tile level. For instance, basic unary and binary primitives such as `lax.add_p` are supported:
```python
tiles = (3, 4, 5)

@partial(jax.jit, backend="ipu")
def compute_fn(in0, in1):
    assert in0.shape == (len(tiles), ...)
    assert in1.shape == (len(tiles), ...)

    # Sharding along the first axis.
    input0 = tile_put_sharded(in0, tiles)
    input1 = tile_put_sharded(in1, tiles)

    # Also existing: replicating along the first axis.
    # semantics: tile_put_replicated(x, tiles) = tile_put_sharded(repeat(X, len(tiles)), tiles)
    input2 = tile_put_replicated(in2, tiles)

    # TODO: combination of tile_put_replicated and tile_put_sharded
    # input2 = tile_put_partial_sharding(in2, tiles, axis=(...))

    output = tile_map_primitive(lax.add_p, [input0, input1])
    return output
```
The previous example will shard arrays over a pre-defined collection of tiles, and then execute on every tile an IPU vertex `add`.

Under the hood, `tile_map_primitive` is building a Poplar compute set with the proper vertex input/output mapping.

The purpose of this API is not only to support standard JAX primitives, but also to make it as easy as possible to call IPU custom vertices. JAX provides an API to introduce [new JAX primitives very easily](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#defining-new-jax-primitives).

Once a new custom primitive has been defined, one needs to register the translation rule between this JAX primitive and the IPU custom vertex (e.g. `register_ipu_tile_primitive(custom_arange_p, custom_arange_tile_translation_ipu)`), allowing to then call similarly
```python
@partial(jax.jit, backend="ipu")
def compute_fn():
    output = tile_map_primitive(custom_arange_p, [], attributes={"size": size, "dtype": dtype}, tiles=tiles)
    return output
```

For more information and details, please refer to the test example:
* [tests/tile/custom_arange_primitive.py](../../tests/tile/custom_arange_primitive.py)
* [tests/tile/custom_arange_vertex.cpp](../../tests/tile/custom_arange_vertex.cpp)
