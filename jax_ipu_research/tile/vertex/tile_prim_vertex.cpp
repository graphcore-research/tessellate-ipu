// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "intrinsics_utils.hpp"

using namespace poplar;

/**
 * @brief Tile barrier vertex: not doing anything, but setting a barrier
 * to put constraints on Poplar program workflow.
 *
 * TODO: support multiple tensor datatypes. Issue: Poplar general reinterpret
 * cast.
 */
template <typename T>
class TileDataBarrierVertex : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

 public:
  // data gated by the barrier.
  Vector<InOut<Vector<T, poplar::VectorLayout::ONE_PTR, 1>>,
         poplar::VectorLayout::ONE_PTR, 1>
      data;

  SUPERVISOR_TARGET bool compute() {
    // Hihihi, not doing anything!
    return true;
  }
};

// explicit instantiations
template class TileDataBarrierVertex<bool>;
template class TileDataBarrierVertex<unsigned char>;
template class TileDataBarrierVertex<signed char>;
template class TileDataBarrierVertex<unsigned short>;
template class TileDataBarrierVertex<short>;
template class TileDataBarrierVertex<unsigned>;
template class TileDataBarrierVertex<int>;
template class TileDataBarrierVertex<float>;
template class TileDataBarrierVertex<half>;
