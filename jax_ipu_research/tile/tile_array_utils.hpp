// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include "common.hpp"

namespace ipu {

/**
 * @brief Create a tensor/variable sharded over IPU tiles
 *
 * @param graph Poplar graph.
 * @param type Datatype of the tensor.
 * @param shape Tensor shape on every tile.
 * @param tiles Tiles on which to shared the tensor/variable.
 * @param debugContext Optional debug context.
 * @return Allocated variable/tensor of shape (T, *shape)
 */
inline poplar::Tensor createShardedVariable(
    poplar::Graph& graph, const poplar::Type& type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context = {}) {
  // Full shape of the sharded tensor.
  ShapeType sharded_shape;
  sharded_shape.push_back(tiles.size());
  for (const auto d : shape) {
    sharded_shape.push_back(d);
  }
  // Create Poplar variable + map on tiles.
  auto t = graph.addVariable(type, sharded_shape, debug_context);
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    graph.setTileMapping(t[idx], tiles[idx]);
  }
  return t;
}

}  // namespace ipu
