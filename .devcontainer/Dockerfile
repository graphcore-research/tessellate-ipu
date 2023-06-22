# JAX IPU currently built on top of SDK 3.1/3.2
# PyTorch rather than TF to avoid numpy version issues
FROM graphcore/pytorch:3.2.0-ubuntu-20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cmake \
    clang-format \
    git \
    git-lfs \
    nano \
    pkg-config \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Additional Python3 packages useful for development.
RUN pip3 install --no-cache-dir -U pip wheel pip-tools
RUN pip3 install --no-cache-dir \
    ipython \
    jupyter \
    numpy \
    scipy \
    pytest \
    flake8 \
    black \
    mypy \
    pre-commit
