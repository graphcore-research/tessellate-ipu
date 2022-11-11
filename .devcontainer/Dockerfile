# JAX IPU currently built on top SDK 3.0
FROM graphcore/tensorflow:2-amd-3.0.0-ubuntu-20.04

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
    numpy
