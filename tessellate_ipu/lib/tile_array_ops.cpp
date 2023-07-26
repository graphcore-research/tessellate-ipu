// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "tile_array_ops.hpp"

namespace ipu {

poplar::Tensor tileBarrierReinterpretTensor(const poplar::Tensor& t,
                                            bool is_half_accurate) {
  // 8 bits data types.
  if (t.elementType() == poplar::BOOL)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  else if (t.elementType() == poplar::CHAR)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  else if (t.elementType() == poplar::SIGNED_CHAR)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  else if (t.elementType() == poplar::UNSIGNED_CHAR)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  // 16 bits data types.
  else if (t.elementType() == poplar::SHORT)
    return t.reinterpret(poplar::UNSIGNED_SHORT);
  else if (t.elementType() == poplar::UNSIGNED_SHORT)
    return t.reinterpret(poplar::UNSIGNED_SHORT);
  // 32 bits data types.
  else if (t.elementType() == poplar::INT)
    return t.reinterpret(poplar::UNSIGNED_INT);
  else if (t.elementType() == poplar::UNSIGNED_INT)
    return t.reinterpret(poplar::UNSIGNED_INT);
  else if (t.elementType() == poplar::FLOAT)
    return t.reinterpret(poplar::UNSIGNED_INT);
  // Special case of FP16/Half!
  else if (t.elementType() == poplar::HALF) {
    if (is_half_accurate) {
      // 16 bits format => can reinterpret as short.
      return t.reinterpret(poplar::UNSIGNED_SHORT);
    } else {
      // IPU model: need to keep as HALF/FP16.
      return t.reinterpret(poplar::HALF);
    }
  }
  // Can handle tensor :/
  throw std::runtime_error("Unknown Poplar tensor type in tile data barrier.");
}

}  // namespace ipu
