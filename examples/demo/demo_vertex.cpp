// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;


template<typename T>
class DemoVertex: public Vertex {
public:
  Input<Vector<T>> x;               // (r,c)
  Input<Vector<T>> constant_scale;  // (r,c)

  Output<Vector<T>> out0;           // (r,c/2)
  Output<Vector<T>> out1;           // (r,c/2)

  // Temporary cache entry (automatically allocated by JAX-tile equation)
  Output<Vector<T>> tmp;            // (r,c)

  // Attribute to pass directly to the vertex.
  T scale_value;

  bool compute() {
    const auto outsize = x.size();
    for (std::size_t idx = 0; idx < x.size(); ++idx) {
      // Most basic compute!
      tmp[idx] = constant_scale[0] * scale_value * x[idx];
      out0[idx / 2] = tmp[idx];
      out1[idx / 2] = -tmp[idx];
    }
    return true;
  }
};

template class DemoVertex<int>;
template class DemoVertex<float>;
