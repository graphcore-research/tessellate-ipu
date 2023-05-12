# JAX IPU **experimental** addons

JAX IPU **experimental** addons: JAX custom primitives and methods maintained by Graphcore research team.

## Installation

This package requires JAX IPU experimental to be installed:
```bash
pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk310 -f https://graphcore-research.github.io/jax-experimental/wheels.html
```

As a pure Python repo, JAX IPU experimental addons can then be directly installed using `pip`:
```bash
pip install git+ssh://git@github.com/graphcore/jax-ipu-experimental-addons.git@main
```
NOTE: `main` can be replaced by any tag (`v0.1`, ...) or commit hash in order to install a specific version.

## What's in there?

The repository contains various pieces of IPU JAX research code, some specific to projects, some more generic:

* [Tile API](jax_ipu_experimental_addons/tile/README.md): basic API on how to directly call IPU vertex in JAX;
* [Popops](...): how to build `popops` expression directly from JAX;
* [DFT](...): code specific to DFT project;

Despite being research code, we still aim at having decent unit test coverage to allow others to reuse it.

## Example

To complete!
```python
import jax_ipu_experimental_addons
```

## Development

We rely on `pre-commit` to perform basic checks on the Python code. The setup is fairly simple:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Additionally, unit tests can be run using `pytest`:
```bash
pip install -r test-requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 pytest -v --tb=short ./tests/
```

How to create the wheel package:
```bash
pip install -U wheel setuptools
python setup.py bdist_wheel --universal
```

How to run a terminal with IPU model (useful for local debugging):
```bash
JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 ipython
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE='poplar_compiler=1,poplar_executor=1' JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 ipython
```

How to benchmark a test / piece of code:
* `PVTI_OPTIONS='{"enable":"true", "directory":"./reports"}'`
* `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "debug.allowOutOfMemory":"true"}'`
