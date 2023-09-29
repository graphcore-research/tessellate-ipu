// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "tile_array_utils.hpp"

namespace ipu {
ShapeType shapePrependAxis(std::size_t d, poplar::ArrayRef<std::size_t> shape) {
  ShapeType ext_shape;
  ext_shape.reserve(shape.size() + 1);
  ext_shape.push_back(d);
  for (const auto& d : shape) {
    ext_shape.push_back(d);
  }
  return ext_shape;
}

std::size_t sizeFromShape(poplar::ArrayRef<std::size_t> shape) noexcept {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<std::size_t>());
}

poplar::Tensor createShardedVariable(
    poplar::Graph& graph, const poplar::Type& type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context) {
  // Full shape of the sharded tensor.
  const auto sharded_shape = shapePrependAxis(tiles.size(), shape);
  // Create Poplar variable + map on tiles.
  auto t = graph.addVariable(type, sharded_shape, debug_context);
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    graph.setTileMapping(t[idx], tiles[idx]);
  }
  return t;
}

poplar::Tensor createConstantTensor(poplar::Graph& graph,
                                    const IpuType& ipu_type,
                                    poplar::ArrayRef<std::size_t> shape,
                                    poplar::ArrayRef<char> raw_values,
                                    const poplar::DebugContext& debug_context) {
  auto poplar_type = toPoplar(ipu_type);
  // Special case of FP16: need to use a different API.
  if (ipu_type == IpuType::HALF) {
    const uint16_t* val = reinterpret_cast<const uint16_t*>(raw_values.data());
    return graph.addConstantHalf(poplar_type, shape, val, debug_context);
  }
  return applyFnToArray(
      [&graph, &poplar_type, &shape, &debug_context](auto&& values) {
        return graph.addConstant(poplar_type, shape, values, debug_context);
      },
      raw_values, ipu_type);
}

poplar::Tensor createReplicatedConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context) {
  // TODO: check raw_values, dtype and shape are consistent.
  // Replicating raw values on the host. Should never be >1GB (worse case!).
  // Allows creating a single constant tensor, which is better for Popvision
  // profile.
  std::vector<char> replicated_raw_values(raw_values.size() * tiles.size());
  auto it = replicated_raw_values.begin();
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    it = std::copy(raw_values.begin(), raw_values.end(), it);
  }
  // Build the full constant tensor at once.
  // TODO: make sure it works with FP16?
  const auto replicated_shape = shapePrependAxis(tiles.size(), shape);
  auto t = createConstantTensor(graph, ipu_type, replicated_shape,
                                replicated_raw_values, debug_context);
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    graph.setTileMapping(t[idx], tiles[idx]);
  }
  return t;
}

poplar::Tensor createShardedConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context) {
  // TODO: check raw_values, dtype and shape are consistent.
  // Creating a single tensor, to avoid Popvision profile bloating.
  auto t =
      createConstantTensor(graph, ipu_type, shape, raw_values, debug_context);
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    graph.setTileMapping(t[idx], tiles[idx]);
  }
  return t;
}

}  // namespace ipu
