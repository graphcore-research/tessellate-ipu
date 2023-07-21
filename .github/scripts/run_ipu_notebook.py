# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_ipu_notebook(filename: str, timeout: int = 600):
    """Run an IPU notebook using IPU model."""
    filename = os.path.abspath(filename)
    root_dir = os.path.dirname(filename)
    print("Notebook to run:", filename)
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)
    # Use JAX IPU model
    os.environ["JAX_IPU_USE_MODEL"] = "1"
    os.environ["JAX_IPU_NUM_TILES"] = "8"
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{root_dir}"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run an IPU notebook, using IPU model.", epilog="Provide a Jupyter notebook filename."
    )
    parser.add_argument("filename")
    args = parser.parse_args()
    run_ipu_notebook(args.filename)
