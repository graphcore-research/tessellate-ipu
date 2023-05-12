# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING, Any, Sequence, Union

import numpy as np

# TODO: replace with upstream JAX when available.
from numpy.typing import ArrayLike, DTypeLike, NDArray  # noqa:  F401

# Supported only on Python 3.9 for running.
if TYPE_CHECKING:
    DType = np.dtype[Any]
else:
    DType = np.dtype
DTypeLike = DTypeLike

DimSize = Union[int, Any]
Shape = Sequence[DimSize]
