# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import glob
import itertools
import os.path
import sys
from typing import List

import setuptools

PACKAGE_NAME = "tessellate_ipu"
repository_dir = os.path.dirname(__file__)

try:
    # READ README.md for long description on PyPi.
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

__version__ = None
with open(os.path.join(repository_dir, PACKAGE_NAME, "_version.py")) as fh:
    exec(fh.read())

with open(os.path.join(repository_dir, "requirements.txt")) as f:
    requirements = f.readlines()

with open(os.path.join(repository_dir, "test-requirements.txt")) as f:
    test_requirements = f.readlines()

# C++ source to include in the package.
# Enable users to build their own custom primitives just using `pip install tessellate`
cpp_extensions = [".h", ".hpp", ".c", ".cpp"]
package_data_cpp_list = [glob.glob(f"{PACKAGE_NAME}/**/*{ext}", recursive=True) for ext in cpp_extensions]
package_data_cpp: List[str] = list(itertools.chain(*package_data_cpp_list))
package_data_cpp = [f.replace(f"{PACKAGE_NAME}/", "") for f in package_data_cpp]

setuptools.setup(
    name=PACKAGE_NAME,
    author="Graphcore Research team",
    version=__version__,
    description="Tessellate IPU library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/graphcore-research/tessellate-ipu",
    packages=setuptools.find_packages(),
    keywords="ipu, graphcore, jax, poplar",
    license="Apache-2.0",
    test_suite="tests",
    tests_require=test_requirements,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={"test": test_requirements},
    package_data={PACKAGE_NAME: ["py.typed"] + package_data_cpp},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
