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
  // TODO: get it working with FP16!
  // Expanded shape (used in concat).
  const auto expand_shape = shapePrependAxis(1, shape);
  // Create Poplar constant per tile. Should I create a single one?
  std::vector<poplar::Tensor> tensor_list;
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    auto t = createConstantTensor(graph, ipu_type, expand_shape, raw_values,
                                  debug_context);
    graph.setTileMapping(t, tiles[idx]);
    tensor_list.push_back(t);
  }
  return poplar::concat(tensor_list, 0);
}

poplar::Tensor createShardedConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context) {
  // TODO: check consistent raw values size.
  // Expanded shape on every tile.
  const auto expand_shape =
      shapePrependAxis(1, arraySlice(shape, 1, shape.size()));
  const auto dtype_size = ipuTypeSize(ipu_type);
  const std::size_t bytes_size = sizeFromShape(expand_shape) * dtype_size;
  auto poplar_type = toPoplar(ipu_type);
  // Create Poplar constant per tile. Should I create a single one?
  std::vector<poplar::Tensor> tensor_list;
  for (size_t idx = 0; idx < tiles.size(); ++idx) {
    // Slicing the raw data corresponding to the tile.
    auto raw_values_tile =
        arraySlice(raw_values, idx * bytes_size, (idx + 1) * bytes_size);
    auto t = createConstantTensor(graph, ipu_type, expand_shape,
                                  raw_values_tile, debug_context);
    graph.setTileMapping(t, tiles[idx]);
    tensor_list.push_back(t);
  }
  return poplar::concat(tensor_list, 0);
}

}  // namespace ipu
