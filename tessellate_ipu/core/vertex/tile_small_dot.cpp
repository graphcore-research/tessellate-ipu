// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "tile_small_dot.hpp"

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

/**
 * @brief 2d rotation vertex.
 */
class Rotation2dVertex : public MultiVertex {
 public:
  using T = float;
  using T2 = float2;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  static constexpr size_t MIN_ALIGN = 8;

  Input<Vector<T, poplar::VectorLayout::ONE_PTR, MIN_ALIGN>>
      cs;  // (2,) rotation cosinus/sinus values
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, MIN_ALIGN>>
      inrow0;  // (N,) first input row vector
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, MIN_ALIGN>>
      inrow1;  // (N,) second input row vector

  Input<Vector<IndexType, poplar::VectorLayout::ONE_PTR>>
      worker_offsets;  // (7,) number threads + 1.

  Output<Vector<T, poplar::VectorLayout::ONE_PTR>>
      outrow0;  // (N,) first input row vector
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>>
      outrow1;  // (N,) first input row vector

  bool compute(unsigned wid) {
    // vectorized offsets.
    const IndexType wstart = worker_offsets[wid];
    const IndexType wend = worker_offsets[wid + 1];
    const IndexType wsize = wend - wstart;

    // Vertex inputs/outputs assuring proper alignment.
    const T2* inrow0_ptr = reinterpret_cast<const T2*>(inrow0.data()) + wstart;
    const T2* inrow1_ptr = reinterpret_cast<const T2*>(inrow1.data()) + wstart;
    const T2* cs_ptr = reinterpret_cast<const T2*>(cs.data());
    T2* outrow0_ptr = reinterpret_cast<T2*>(outrow0.data()) + wstart;
    T2* outrow1_ptr = reinterpret_cast<T2*>(outrow1.data()) + wstart;

    rotation2d_f32(cs_ptr[0], inrow0_ptr, inrow1_ptr, outrow0_ptr, outrow1_ptr,
                   wsize, IPU_DISPATCH_TAG);
    return true;
  }
};
