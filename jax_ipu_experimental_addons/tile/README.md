# Tile Poplar programming in Python

The sub-module `jax_ipu_experimental_addons.tile` provides a low level Poplar tile programming directly in JAX, by exposing Poplar tensor tile mapping and vertex calling directly in Python.

This light API is based on three main concepts:
* `TileShardedArray` data structure, wrapping a classic JAX array with tile mapping information;
* Tile mapping functions `tile_put_replicated` and `tile_put_sharded`;
* Tile vertex mapping function `tile_map_primitive`;

Note that even though these APIs are IPU specific, they are still compatible with other backends (CPU, GPU, ...). On the latter, any of these call is either a no-op, or a redirection to JAX standard `vmap`.

## `TileShardedArray` data structure

`TileShardedArray` is a thin wrapper on top of JAX array adding IPU tile mapping information, and compatible with JAX Pytree mechanism. In short:
```python
@register_pytree_node_class
@dataclass(frozen=True)
class TileShardedArray:
    # Underlying data.
    array: jax.Array
    # Tile mapping of the first axis.
    tiles: Tuple[int, ...]
```
At the moment, `TileShardedArray` can only represent tile sharding over the first array axis (i.e. `axis=0` in Numpy world). Some extensions to represent multi-axis sharding are planned. Compatibility with JAX `vmap` and `grad` is also part of future improvements.

`TileShardedArray` implements the basic Numpy array API (i.e. `dtype`, `shape`, slicing, ...), and is fully compatible with the standard JAX Numpy API.

Considering a `TileShardedArray` array `v` of shape `(3, 4)` sharded over the first axis on tiles `(0, 2, 5)`, it means that every slice `v[0]`, `v[1]` and `v[2]` will be contiguous arrays living pn tiles `0`, `2` and `5` SRAM memory. The on-tile memory contiguity is always ensured, meaning that no additional on-tile copy is necessary when calling an IPU vertex with `tile_map_primitive` (with the exception of some memory alignment edge cases).

## IPU tile sharding using `tile_put_replicated` and `tile_put_sharded`

Two methods are provided to construct a `TileShardedArray` (mimicking `jax.device_put` API):
* `tile_put_replicated`;
* `tile_put_sharded`;

As indicated by its name, `tile_put_replicated` will replicate an array over a collection of tiles, meaning that the resulting array will have an additional first tile axis. In other words,
```python
v = ... # JAX array of shape (4, 5)
out = tile_put_replicated(v, (1, 2, 5))
```
will return a `TileShardedArray` of shape `(3, 4, 5)`, with identical data on tiles `1`, `2` and `5`.


Similarly, `tile_put_sharded` will shard an array over the first axis, splitting the data between the provided collection of tiles. In other words,
```python
v = ... # JAX array of shape (4, 5)
out = tile_put_sharded(v, (1, 2, 5, 8))
```
will return a `TileShardedArray` of shape `(4, 5)`, with data sharded on tiles `1`, `2`, `5` and `8`. The previous call will raise an exception if the number of tiles does not correspond to the first dimension.

**Note:** `tile_put_sharded` and `tile_put_replicated` can be combined with standard JAX slicing, transpose, ... operations to build complex IPU tile inter-exchange patterns.

Tile JAX addon also provides the equivalent `tile_constant_replicated` and `tile_constant_sharded` to build Poplar on tile constant arrays from Numpy tensors.

## IPU vertex call using `tile_map_primitive`

Once array(s) have been sharded over IPU tiles, one can map a function on the former using `tile_map_primitive`. Under the hood, it will add a Poplar vertex call to the graph on all tiles where data is present. All tiles workload will run independently in parallel (no sync being required).

`tile_map_primitive` first argument is a [JAX LAX](https://jax.readthedocs.io/en/latest/jax.lax.html) primitive. Tile JAX addon provides a mapping from (most) standard JAX LAX primitives to Graphcore Poplibs optimized vertices, meaning you will be able to take full advantage of the IPU hardware in just a couple of lines of Python.

For instance, here is simple example calling a JAX LAX primitive on a collection of tiles:

```python
data = np.array([1, -2, 3], np.float32)
tiles = (0, 2, 5)

@partial(jax.jit, backend="ipu")
def compute_fn(input):
    # Shard data over tiles.
    input = tile_put_sharded(input, tiles)
    # Call Popops NEG vertex on tiles (0, 2, 5).
    return tile_map_primitive(jax.lax.neg_p, input)

output = compute_fn(data) # Returns TileShardedArray
```

Popvision graph analyser will generate a profile exactly reflecting the code written in Python:
![tile_map_popvision](../../docs/images/tile_map_popvision.png)

**Note:** since built on top of standard JAX LAX primitive, the previous example is fully compatible with other badkends (i.e. `cpu`, `gpu`, ...). `tile_map_primitive` call will just be translated into a standard JAX `vmap`.

## IPU custom vertex integration

JAX can easily be extended with [custom primitives](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#defining-new-jax-primitives). Using this extension API, we provide an easy way to integrate custom IPU C++ vertices in Tile JAX addon. In short, once you have a `Vertex` C++ class, you will only need to the following lines to expose it in Python:
```python
@declare_ipu_tile_primitive("DemoVertex<{x}>", gp_filename=demo_vertex_filename)
def demo_vertex_p(x: jax.ShapedArray):
    r, c = x.shape
    # Declare Vertex outputs: name, shape and dtype.
    outputs = {
        "out0": jax.ShapedArray((r, c // 2), x.dtype),
        "out1": jax.ShapedArray((r, c // 2), x.dtype)
    }
    # Additional constant tensor to pass to the vertex
    constants = {
        "constant_scale": 0.5 * np.ones((r, c), dtype=x.dtype)
    }
    # Temporary scratch space allocated for the vertex.
    temps = {
        "tmp": jax.ShapedArray((r, c), x.dtype)
    }
    # Performance estimate of the vertex. Used by IPU model.
    perf_estimate = r * c * 12
    return outputs, constants, temps, perf_estimate
```

Once declared in this way, the custom vertex can easily be called using `tile_map_primitive`:
```python
@partial(jax.jit, backend="ipu")
def compute_fn(input):
    input = tile_put_sharded(input, tiles)
    # Poplar call to custom vertex `DemoVertex`.
    return tile_map_primitive(demo_vertex_p, input)
```

Can you maintain compatibility with other backends? Yes! For that, you will just need to provide a default JAX Numpy implementation of the custom primitive:
```python
def demo_vertex_impl(x, scale_value=None):
    r, c = x.shape
    out0 = ... # JAX Numpy code.
    out1 = -out0
    return out0, out1

# Primitive default implementation, in JAX Numpy.
demo_vertex_p.def_impl(demo_vertex_impl)
```

Please refer to the [tile demo examples](../../examples/demo/) for more details. The library unit tests also implement some custom vertex examples:
* [tests/tile/custom_arange_primitive.py](../../tests/tile/custom_arange_primitive.py)
* [tests/tile/custom_arange_vertex.cpp](../../tests/tile/custom_arange_vertex.cpp)
