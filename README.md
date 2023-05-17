<div align="center">
<img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" alt="logo"></img>
</div>


# JAX IPU **Experimental** Addons

[**Features**](#features)
| [**Installation guide**](#installation)
| [**Quickstart**](#minimal-example)

JAX IPU :red_circle: **experimental** :red_circle: Addons is a collection of tools to bring low-level Poplar IPU programming to JAX and Python.

The package is maintained by the Graphcore Research team. Expect bugs and sharp edges! Please let us know what you think!

## Features

At the moment, the package features one module [`tile`](jax_ipu_experimental_addons/tile/README.md) bringing low-level Poplar IPU programming to JAX (and is fully compatible with the standard JAX API). More specifically:

* Control tile mapping of arrays using `tile_put_replicated` or `tile_put_sharded`
* Support of standard JAX LAX operations at tile level (using `tile_map_primitive`)
* Easy integration of custom IPU C++ vertex (see [vertex example](examples/demo/demo_vertex.py))
* Access to low-level IPU hardware functionalities such as cycle count and random seed set/get
* Full compatibility with other backends

This additional API allows easy and quick implementation of algorithms on IPUs, while keeping compatibility with other backends (CPU/GPU/TPU).

## Installation guide

This package requires **[JAX IPU experimental](https://github.com/graphcore-research/jax-experimental)** (available for Python 3.8 and Poplar SDK versions 3.1 or 3.2).

For Poplar SDK 3.1:
```bash
pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk310 -f https://graphcore-research.github.io/jax-experimental/wheels.html
```

For Poplar SDK 3.2:
```bash
pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk320 -f https://graphcore-research.github.io/jax-experimental/wheels.html
```

As a pure Python repo, JAX IPU experimental addons can then be directly installed from Github using `pip`:
```bash
pip install git+https://github.com/graphcore-research/jax-ipu-experimental-addons.git@main
```
Note: `main` can be replaced with any tag (`v0.1`, ...) or commit hash in order to install a specific version.


## Minimal example

The following is a simple example showing how to set the tile mapping of JAX arrays, and run a JAX LAX operation on these tiles.

```python
import numpy as np
import jax
from jax_ipu_experimental_addons.tile import tile_put_sharded, tile_map_primitive

# Which IPU tiles do we want to use?
tiles = (0, 1, 3)

@jax.jit
def compute_fn(data0, data1):
    # Tile sharding arrays along the first axis.
    input0 = tile_put_sharded(data0, tiles)
    input1 = tile_put_sharded(data1, tiles)
    # Map a JAX LAX primitive on tiles.
    output = tile_map_primitive(jax.lax.add_p, input0, input1)
    return output

data = np.random.rand(len(tiles), 2, 3).astype(np.float32)
output = compute_fn(data, 3 * data)

print("Output:", output)
```

### Useful environment variables and flags

JAX IPU experimental addons flags, using `from jax.config import config`:


| Flag | Description |
| ---- | --- |
| `config.FLAGS.jax_platform_name ='ipu'/'cpu'` | Configure default JAX backend. Useful for CPU initialization. |
| `config.FLAGS.jax_ipu_use_model = True`       | Use IPU model emulator. |
| `config.FLAGS.jax_ipu_model_num_tiles = 8`    | Set the number of tiles in the IPU model. |
| `config.FLAGS.jax_ipu_device_count = 2`       | Set the number of IPUs visible in JAX. Can be any local IPU available. |
| `config.FLAGS.jax_ipu_visible_devices = '0,1'`  | Set the specific collection of local IPUs to be visible in JAX. |

Alternatively, like other JAX flags, these can be set using environment variables (for example `JAX_IPU_USE_MODEL` and `JAX_IPU_MODEL_NUM_TILES`).


[PopVision](https://www.graphcore.ai/developer/popvision-tools) environment variables:
* Generate a PopVision Graph analyser profile: `PVTI_OPTIONS='{"enable":"true", "directory":"./reports"}'`
* Generate a PopVision system analyser profile: `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "debug.allowOutOfMemory":"true"}'`

## License

Copyright (c) 2023 Graphcore Ltd. The project is licensed under the [**Apache License 2.0**](LICENSE).

JAX IPU tile programming is implemented using C++ custom operations. The later has the following [C++ libraries](jax_ipu_experimental_addons/external) as dependencies, statically compiled into a shared library:

| Component | Description | License |
| --- | --- | --- |
| [fastbase64](https://github.com/lemire/fastbase64) | Base64 fast decoder library | [Simplified BSD (FreeBSD) License](https://github.com/lemire/fastbase64/blob/master/LICENSE) |
| [fmt](https://github.com/fmtlib/fmt) | A modern C++ formatting library | [MIT license](https://github.com/fmtlib/fmt/blob/master/LICENSE.rst) |
| [half](https://sourceforge.net/projects/half/) | IEEE-754 conformant half-precision library | MIT license |
| [json](https://github.com/nlohmann/json) | JSON for modern C++ | [MIT license](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT) |
| [pybind11](https://github.com/pybind/pybind11) | C++11 python bindings | [BSD License 2.0](https://github.com/pybind/pybind11/blob/master/LICENSE) |
