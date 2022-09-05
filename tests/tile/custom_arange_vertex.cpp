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
  // using T = float;
  Output<Vector<T, poplar::VectorLayout::SPAN>> out;

  bool compute() {
    const auto outsize = out.size();
    for (std::size_t idx = 0; idx < outsize; ++idx) {
      out[idx] = T(idx);
    }
    return true;
  }
};

// explicit instantiations
template class CustomArangeVertex<int>;
template class CustomArangeVertex<half>;
template class CustomArangeVertex<float>;
