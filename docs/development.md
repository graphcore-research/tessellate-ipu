# TessellateIPU Development Guidelines

There are some guidelines for doing development on the TessellateIPU library.

## Local install & packaging

For local development, we recommend using a pip editable install:
```bash
pip install cmake ninja nanobind scikit-build-core[pyproject]
pip install --no-build-isolation -ve .
```
The first line will install the required build dependencies, and the second will create an editable install
of TessellateIPU. Note that the argument `--no-build-isolation` is optional, but speed up the compilation by avoiding re-creating a compilation virtual environnment at every call.

Source (sdist) and build (wheels) packages can easily be generated using `build` Python package:
```bash
pip install build
python -m build --no-isolation
```
Source tarball and compiled wheels will be created in the `dist` directory.


## Development

We rely on `pre-commit` to perform basic checks on the Python code. Set up `pre-commit` with:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Run unit tests using `pytest`:
```bash
pip install --no-build-isolation -ve .
pip install -r test-requirements.txt
JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=16 pytest -v --tb=short ./tests/
```

Run a terminal with an IPU model (useful for local debugging):
```bash
JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 ipython
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE='poplar_compiler=1,poplar_executor=1' JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 ipython
```

## Useful dev. flags/env variables

* `XLA_FLAGS='--xla_dump_to=./xla_dumps --xla_dump_hlo_pass_re=.'`: XLA passes dump;
* `"debug.branchRecordFlush":"false"`: Deactivate Poplar loop profiling flush => smaller overhead;
