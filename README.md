# JAX IPU addons

JAX IPU addons is gathering code useful for running JAX on the IPU. Features developed in this repository can fall in several categories:
* Modifications of standard JAX (and FLAX, etc) examples to run fast on the IPU.
* IPU bug fixes: JAX pure Python fix, specific to the IPU, which can patched outside core JAX;
* IPU specific features, like  code outlining, pipelining, ...

## Installation

As a pure Python repo, JAX IPU addons can be directly installed using `pip`:
```bash
pip install git+ssh://git@github.com/graphcore/jax-ipu-addons.git@main
```
NOTE: `main` can be replaced by any tag (`v0.1`, ...) or commit hash in order to install a specific version.

## Example

To complete!
```python
import jax_ipu_addons
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
TF_POPLAR_FLAGS='--use_ipu_model' pytest ./tests/
```

How to create the wheel package:
```bash
pip install -U wheel setuptools
python setup.py bdist_wheel --universal
```

How to run a terminal with IPU model (useful for local debugging):
```bash
TF_POPLAR_FLAGS='--use_ipu_model' ipython
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE='poplar_compiler=1,poplar_executor=1' TF_POPLAR_FLAGS='--use_ipu_model' ipython
```
