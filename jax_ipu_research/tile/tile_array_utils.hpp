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

/**
 * @brief Create a replicated constant tensor.
 * @param graph Poplar graph.
 * @param ipu_type IPU datatype of the tensor.
 * @param shape Tensor shape on every tile.
 * @param raw_values Constant data raw values.
 * @param tiles Tiles on which to shared the tensor/variable.
 * @param debugContext Optional debug context.
 * @return Allocated constant tensor of shape (T, *shape)
 */
inline poplar::Tensor createReplicatedConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context = {}) {
  // Expanded shape. TODO: re-factor outside.
  ShapeType expand_shape;
  expand_shape.push_back(1);
  for (const auto d : shape) {
    expand_shape.push_back(d);
  }
  auto poplar_type = toPoplar(ipu_type);
  // Create Poplar constant per tile. Should I create a single one?
  std::vector<poplar::Tensor> tensor_list;
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    auto t = applyFnToArray(
        [&graph, &poplar_type, &expand_shape, &debug_context](auto&& values) {
          return graph.addConstant(poplar_type, expand_shape, values,
                                   debug_context);
        },
        raw_values, ipu_type);
    graph.setTileMapping(t, tiles[idx]);
    tensor_list.push_back(t);
  }
  return poplar::concat(tensor_list, 0);
}

}  // namespace ipu
