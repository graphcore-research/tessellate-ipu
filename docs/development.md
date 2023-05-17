# JAX IPU Experimental Addons development guidelines

There are some guidelines for doing development on JAX IPU experimental addons.

## Development

We rely on `pre-commit` to perform basic checks on the Python code. Set up `pre-commit` with:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Run unit tests using `pytest`:
```bash
pip install -r test-requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 pytest -v --tb=short ./tests/
```

Create the wheel package with:
```bash
pip install -U wheel setuptools
python setup.py bdist_wheel --universal
```

Run a terminal with an IPU model (useful for local debugging):
```bash
JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 ipython
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE='poplar_compiler=1,poplar_executor=1' JAX_IPU_USE_MODEL=true JAX_IPU_MODEL_NUM_TILES=8 ipython
```
