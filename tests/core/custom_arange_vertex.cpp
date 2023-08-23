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
  // Testing constant vertex tensor.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> global_scale; // (1,)
  Output<Vector<T, poplar::VectorLayout::SPAN>> out;  // (size, )

  bool compute() {
    const auto outsize = out.size();
    for (std::size_t idx = 0; idx < outsize; ++idx) {
      out[idx] = T(idx) * scales[0][idx] * scales[1][idx] * global_scale[0];
    }
    return true;
  }
};

template<typename T>
class CustomSingleOutVertex : public Vertex {
 public:
  Input<Vector<T, poplar::VectorLayout::SPAN>> input;       // (size, )
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> output;  // (size, )

  bool compute() {
    const auto outsize = input.size();
    for (std::size_t idx = 0; idx < input.size(); ++idx) {
      // Most basic compute!
      output[idx] = -input[idx];
    }
    return true;
  }
};


class CustomInplaceVertex : public Vertex {
 public:
  InOut<Vector<float, poplar::VectorLayout::SPAN>> inout;   // (size, )

  bool compute() {
    for (std::size_t idx = 0; idx < inout.size(); ++idx) {
      // Most basic inplace compute!
      inout[idx] *= 2.0f;
    }
    return true;
  }
};


template<typename T>
class CustomMultiOutVertex : public Vertex {
 public:
  Input<Vector<T, poplar::VectorLayout::SPAN>> input;     // (size, )
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> constant_scale; // (1, )

  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out0;  // (size, )
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> out1;  // (size, )

  // Temporary cache entry (automatically allocated by JAX-tile equation)
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> mytmp;  // (size, )

  // Attribute to pass directly to the vertex.
  T scale_value;

  bool compute() {
    const auto outsize = input.size();
    for (std::size_t idx = 0; idx < input.size(); ++idx) {
      // Most basic compute!
      mytmp[idx] = constant_scale[0] * scale_value * input[idx];
      out0[idx] = mytmp[idx];
      out1[idx] = -mytmp[idx];
    }
    return true;
  }
};

// explicit instantiations
template class CustomArangeVertex<int>;
template class CustomArangeVertex<half>;
template class CustomArangeVertex<float>;

template class CustomSingleOutVertex<int>;
template class CustomSingleOutVertex<float>;

template class CustomMultiOutVertex<int>;
template class CustomMultiOutVertex<float>;
