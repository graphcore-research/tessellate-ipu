// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

// clang-format off
/**
 * Compilation for all supported targets:
 *      popc -O2 tests/tile/custom_arange_vertex.cpp -o tests/tile/custom_arange_vertex.gp
 */
template<typename T>
class CustomArangeVertex : public Vertex {
 public:
  // Testing 2d tensor IO supported.
  Vector<Input<Vector<T>>, poplar::VectorLayout::ONE_PTR> scales; // (2, size)
  Output<Vector<T, poplar::VectorLayout::SPAN>> out;  // (size, )

  bool compute() {
    const auto outsize = out.size();
    for (std::size_t idx = 0; idx < outsize; ++idx) {
      out[idx] = T(idx) * scales[0][idx] * scales[1][idx];
    }
    return true;
  }
};

// explicit instantiations
template class CustomArangeVertex<int>;
template class CustomArangeVertex<half>;
template class CustomArangeVertex<float>;
