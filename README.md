# JAX IPU **research** addons

JAX IPU **research** addons: JAX custom primitives and methods maintained by Graphcore research team.

## Installation

As a pure Python repo, JAX IPU research can be directly installed using `pip`:
```bash
pip install git+ssh://git@github.com/graphcore/jax-ipu-research.git@main
```
NOTE: `main` can be replaced by any tag (`v0.1`, ...) or commit hash in order to install a specific version.

## What's in there?

The repository contains various pieces of IPU JAX research code, some specific to projects, some more generic:

* [Tile API](jax_ipu_research/tile/README.md): basic API on how to directly call IPU vertex in JAX;
* [Popops](...): how to build `popops` expression directly from JAX;
* [DFT](...): code specific to DFT project;

Despite being research code, we still aim at having decent unit test coverage to allow others to reuse it.

## Example

To complete!
```python
import jax_ipu_research
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
TF_POPLAR_FLAGS='--use_ipu_model --ipu_model_tiles=8' pytest -vv ./tests/
```

How to create the wheel package:
```bash
pip install -U wheel setuptools
python setup.py bdist_wheel --universal
```

How to run a terminal with IPU model (useful for local debugging):
```bash
TF_POPLAR_FLAGS='--use_ipu_model --ipu_model_tiles=8' ipython
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE='poplar_compiler=1,poplar_executor=1' TF_POPLAR_FLAGS='--use_ipu_model' ipython
```

How to benchmark a test / piece of code:
* `PVTI_OPTIONS='{"enable":"true", "directory":"./reports"}'`
* `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "debug.allowOutOfMemory":"true"}'`
