# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Sequence, Union

import numpy as np

# TODO: replace with upstream JAX when available.
Array = Any
DType = np.dtype
DTypeLike = Union[Any, str, np.dtype]

DimSize = Union[int, Any]
Shape = Sequence[DimSize]

ArrayLike = Union[
    Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_,
    np.number,  # NumPy scalar types
    bool,
    int,
    float,
    complex,  # Python scalar types
]
