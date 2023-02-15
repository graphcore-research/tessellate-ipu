// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

template<typename T>
class DemoVertex: public Vertex {
public:
  Input<Vector<T, poplar::VectorLayout::SPAN>> in;     // (size, )
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> constant_scale; // (1, )

  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out0;  // (size, )
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out1;  // (size, )

  // Temporary cache entry (automatically allocated by JAX-tile equation)
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> tmp;  // (size, )

  // Attribute to pass directly to the vertex.
  T scale_value;

  bool compute() {
    const auto outsize = in.size();
    for (std::size_t idx = 0; idx < in.size(); ++idx) {
      // Most basic compute!
      tmp[idx] = constant_scale[0] * scale_value * in[idx];
      out0[idx] = tmp[idx];
      out1[idx] = -tmp[idx];
    }
    return true;
  }
};

template class DemoVertex<int>;
template class DemoVertex<float>;
