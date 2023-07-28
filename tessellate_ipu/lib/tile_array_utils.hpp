// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <algorithm>
#include <poplar/Graph.hpp>

#include "base_types.hpp"

namespace ipu {

/**
 * @brief Prepend an axis with given size to a shape object.
 * @param d Size of the new dimension to prepend.
 * @param shape Existing shape.
 * @return New shape instance (d, *shape).
 */
ShapeType shapePrependAxis(std::size_t d, poplar::ArrayRef<std::size_t> shape);

/**
 * @brief Get the size (number elements) from a shape object.
 */
std::size_t sizeFromShape(poplar::ArrayRef<std::size_t> shape) noexcept;

/**
 * @brief Slice an array, extracting a view.
 * @param arr Array to slice.
 * @param start Start index (included).
 * @param end End index (excluded).
 * @return Slice of the array.
 */
template <typename T>
poplar::ArrayRef<T> arraySlice(poplar::ArrayRef<T> arr, std::size_t start,
                               std::size_t end) noexcept {
  const std::size_t size = end - start;
  return poplar::ArrayRef<T>(arr.data() + start, size);
}

/**
 * @brief Create a tensor/variable sharded over IPU tiles
 *
 * @param graph Poplar graph.
 * @param type Datatype of the tensor.
 * @param shape Tensor shape on every tile.
 * @param tiles Tiles on which to shard the tensor/variable.
 * @param debugContext Optional debug context.
 * @return Allocated variable/tensor of shape (T, *shape)
 */
poplar::Tensor createShardedVariable(
    poplar::Graph& graph, const poplar::Type& type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context = {});

/**
 * @brief Create a constant tensor from dtype, shape and raw values.
 */
poplar::Tensor createConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    const poplar::DebugContext& debug_context = {});

/**
 * @brief Create a replicated constant tensor.
 * @param graph Poplar graph.
 * @param ipu_type IPU datatype of the tensor.
 * @param shape Tensor shape on every tile.
 * @param raw_values Constant data raw values.
 * @param tiles Tiles on which to replicate the tensor/variable.
 * @param debugContext Optional debug context.
 * @return Allocated constant tensor of shape (T, *shape)
 */
poplar::Tensor createReplicatedConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context = {});

/**
 * @brief Create a sharded constant tensor.
 * @param graph Poplar graph.
 * @param ipu_type IPU datatype of the tensor.
 * @param shape Tensor shape on every tile.
 * @param raw_values Constant data raw values (of the full tensor).
 * @param tiles Tiles on which to shard the tensor/variable.
 * @param debugContext Optional debug context.
 * @return Allocated constant tensor of shape (T, *shape)
 */
poplar::Tensor createShardedConstantTensor(
    poplar::Graph& graph, const IpuType& ipu_type,
    poplar::ArrayRef<std::size_t> shape, poplar::ArrayRef<char> raw_values,
    poplar::ArrayRef<TileIndexType> tiles,
    const poplar::DebugContext& debug_context = {});

}  // namespace ipu
