// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "intrinsics_utils.hpp"
using namespace poplar;

namespace tl {

enum OpType : int { SUM = 0, MIN = 1, MAX = 2, PROD = 3 };

template <typename T, int OP>
T initial_accumulator_value() {
  if constexpr (OP == SUM) {
    return T(0);
  } else if constexpr (OP == MIN) {
    return std::numeric_limits<T>::max();
  } else if constexpr (OP == MAX) {
    return std::numeric_limits<T>::lowest();
  } else {
    return T(1);
  }
}
template <typename T, int OP>
T cumulative_op(T acc, T rhs) {
  if constexpr (OP == SUM) {
    return acc + rhs;
  } else if constexpr (OP == MIN) {
    return std::min(acc, rhs);
  } else if constexpr (OP == MAX) {
    return std::max(acc, rhs);
  } else {
    return acc * rhs;
  }
}

/**
 * @brief Cumulative op vertex.
 * Very simple implementation at first, no big optimization!
 */
template <typename T, int OP>
class CumulativeOp : public Vertex {
 public:
  Input<Vector<T, poplar::VectorLayout::SPAN>> in;
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out;

  bool compute() {
    T accumulator = initial_accumulator_value<T, OP>();
    const int32_t size = in.size();
    for (int32_t idx = 0; idx < size; ++idx) {
      accumulator = cumulative_op<T, OP>(accumulator, in[idx]);
      out[idx] = accumulator;
    }
    return true;
  }
};

// explicit instantiations
template class CumulativeOp<int, SUM>;
template class CumulativeOp<float, SUM>;

template class CumulativeOp<int, MIN>;
template class CumulativeOp<float, MIN>;

template class CumulativeOp<int, MAX>;
template class CumulativeOp<float, MAX>;

template class CumulativeOp<int, PROD>;
template class CumulativeOp<float, PROD>;

}  // namespace tl
