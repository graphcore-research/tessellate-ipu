<div align="center">
  <img src="https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png" alt="logo"></img>
  <h1>TessellateIPU Library</h1>
</div>

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/tessellate-ipu?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fnotebooks%2F01-tessellate-ipu-tile-api-basics.ipynb)
[![tests](https://github.com/graphcore-research/tessellate-ipu/actions/workflows/tests-public.yaml/badge.svg)](https://github.com/graphcore-research/tessellate-ipu/actions/workflows/tests-public.yaml)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/graphcore-research/tessellate-ipu/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/graphcore-research/tessellate-ipu)](https://github.com/graphcore-research/tessellate-ipu/stargazers)

<!-- [![codecov](https://codecov.io/gh/datamol-io/graphium/branch/main/graph/badge.svg?token=bHOkKY5Fze)](https://codecov.io/gh/datamol-io/graphium) -->

[**Features**](#features)
| [**Installation guide**](#installation-guide)
| [**Quickstart**](#minimal-example)
| [**Documentation**](#documentation)

:red_circle: :warning: **Non-official Graphcore Product** :warning: :red_circle:

**TessellateIPU** is a library bringing low-level Poplar IPU programming to Python ML frameworks (JAX at the moment, and PyTorch in the near future).

The package is maintained by the Graphcore Research team. Expect bugs and sharp edges! Please let us know what you think!

## Features

TessellateIPU brings low-level Poplar IPU programming to Python, while being fully compatible with ML framework standard APIs. The main features are:

* Control tile mapping of arrays using `tile_put_replicated` or `tile_put_sharded`
* Support of standard JAX LAX operations at tile level (using `tile_map`)
* Easy integration of custom IPU C++ vertex (see [vertex example](examples/demo/demo_vertex.py))
* Access to low-level IPU hardware functionalities such as cycle count and random seed set/get
* Full compatibility with other backends

The TessellateIPU API allows easy and efficient implementation of algorithms on IPUs, while keeping compatibility with other backends (CPU, GPU, TPU). For more details on the API, please refer to the [TessellateIPU documentation](docs/basics.md).

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

As a pure Python repo, TessellateIPU can then be directly installed from GitHub using `pip`:
```bash
pip install git+https://github.com/graphcore-research/tessellate-ipu.git@main
```
Note: `main` can be replaced with any tag (`v0.1`, ...) or commit hash in order to install a specific version.


## Minimal example

The following is a simple example showing how to set the tile mapping of JAX arrays, and run a JAX LAX operation on these tiles.

```python
import numpy as np
import jax
from tessellate_ipu import tile_put_sharded, tile_map

# Which IPU tiles do we want to use?
tiles = (0, 1, 3)

@jax.jit
def compute_fn(data0, data1):
    # Tile sharding arrays along the first axis.
    input0 = tile_put_sharded(data0, tiles)
    input1 = tile_put_sharded(data1, tiles)
    # Map a JAX LAX primitive on tiles.
    output = tile_map(jax.lax.add_p, input0, input1)
    return output

data = np.random.rand(len(tiles), 2, 3).astype(np.float32)
output = compute_fn(data, 3 * data)

print("Output:", output)
```

### Useful environment variables and flags

JAX IPU experimental flags, using `from jax.config import config`:


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

## Documentation

* [Tessellate API basics](docs/basics.md)
* [Tessellate development guidelines](docs/development.md)

## License

Copyright (c) 2023 Graphcore Ltd. The project is licensed under the [**Apache License 2.0**](LICENSE).

TessellateIPU is implemented using C++ custom operations. These have the following [C++ libraries](tessellate_ipu/external) as dependencies, statically compiled into a shared library:

| Component | Description | License |
| --- | --- | --- |
| [fastbase64](https://github.com/lemire/fastbase64) | Base64 fast decoder library | [Simplified BSD (FreeBSD) License](https://github.com/lemire/fastbase64/blob/master/LICENSE) |
| [fmt](https://github.com/fmtlib/fmt) | A modern C++ formatting library | [MIT license](https://github.com/fmtlib/fmt/blob/master/LICENSE.rst) |
| [half](https://sourceforge.net/projects/half/) | IEEE-754 conformant half-precision library | MIT license |
| [json](https://github.com/nlohmann/json) | JSON for modern C++ | [MIT license](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT) |
| [pybind11](https://github.com/pybind/pybind11) | C++11 python bindings | [BSD License 2.0](https://github.com/pybind/pybind11/blob/master/LICENSE) |
