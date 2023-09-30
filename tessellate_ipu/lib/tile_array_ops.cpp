// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define FMT_HEADER_ONLY
#include "tile_array_ops.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <poplar/Program.hpp>
#include <poputil/exceptions.hpp>

#include "tile_array_utils.hpp"

namespace ipu {

std::string makeTileOpDebugPrefix(const std::string& raw_debug_prefix,
                                  const std::string& basename) {
  const auto format_debug_prefix = [&raw_debug_prefix,
                                    &basename](std::size_t idx) {
    const std::string debug_prefix =
        fmt::format("{}{}", raw_debug_prefix.substr(0, idx), basename);
    return debug_prefix;
  };
  std::string::size_type idx;
  // A bit of ugly string pattern matching to remove the metadata, but keep
  // the existing namespace.
  idx = raw_debug_prefix.rfind(basename + "[");
  if (idx != std::string::npos) {
    return format_debug_prefix(idx);
  }
  // Not found => keep the same debug prefix.
  return raw_debug_prefix;
}

poplar::Tensor tileBarrierReinterpretTensor(const poplar::Tensor& t,
                                            bool is_half_accurate) {
  // 8 bits data types.
  if (t.elementType() == poplar::BOOL)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  else if (t.elementType() == poplar::CHAR)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  else if (t.elementType() == poplar::SIGNED_CHAR)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  else if (t.elementType() == poplar::UNSIGNED_CHAR)
    return t.reinterpret(poplar::UNSIGNED_CHAR);
  // 16 bits data types.
  else if (t.elementType() == poplar::SHORT)
    return t.reinterpret(poplar::UNSIGNED_SHORT);
  else if (t.elementType() == poplar::UNSIGNED_SHORT)
    return t.reinterpret(poplar::UNSIGNED_SHORT);
  // 32 bits data types.
  else if (t.elementType() == poplar::INT)
    return t.reinterpret(poplar::UNSIGNED_INT);
  else if (t.elementType() == poplar::UNSIGNED_INT)
    return t.reinterpret(poplar::UNSIGNED_INT);
  else if (t.elementType() == poplar::FLOAT)
    return t.reinterpret(poplar::UNSIGNED_INT);
  // Special case of FP16/Half!
  else if (t.elementType() == poplar::HALF) {
    if (is_half_accurate) {
      // 16 bits format => can reinterpret as short.
      return t.reinterpret(poplar::UNSIGNED_SHORT);
    } else {
      // IPU model: need to keep as HALF/FP16.
      return t.reinterpret(poplar::HALF);
    }
  }
  // Can handle tensor :/
  throw std::runtime_error("Unknown Poplar tensor type in tile data barrier.");
}

poplar::program::Program lowerTilePutShardedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileArrayType& tile_array,
    const poplar::DebugContext& debug_context) {
  if (inputs.size() != 1) {
    throw poputil::poplibs_error(
        "IPU tile put sharded expecting a single input tensor.");
  }
  static_assert(sizeof(TileIndexType) == 4);
  auto input = inputs[0];

  if (input.shape()[0] != tile_array.size()) {
    throw poputil::poplibs_error(
        fmt::format("IPU tile put sharding: inconsistent input size {} and "
                    "tiles length {}.",
                    input.shape()[0], tile_array.size()));
  }

  // Create output tensor, with proper tile mapping.
  // TODO: link to Slack discussion on VarRegion contiguity.
  auto output = createShardedVariable(
      graph, input.elementType(), input[0].shape(), tile_array, debug_context);
  // Copy data tensor into the output.
  auto prog = poplar::program::Copy(input, output, false, debug_context);
  outputs.push_back(output);
  return prog;
}

poplar::program::Program lowerTilePutReplicatedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileArrayType& tile_array,
    const poplar::DebugContext& debug_context) {
  if (inputs.size() != 1) {
    throw poputil::poplibs_error(
        "IPU tile put replicated expecting a single input tensor.");
  }
  static_assert(sizeof(TileIndexType) == 4);
  auto input = inputs[0];

  // Create output tensor, with proper tile mapping.
  auto input_broadcasted = input.expand({0}).broadcast(tile_array.size(), 0);
  auto output = createShardedVariable(graph, input.elementType(), input.shape(),
                                      tile_array, debug_context);
  // Copy data tensor into the output.
  auto prog =
      poplar::program::Copy(input_broadcasted, output, false, debug_context);
  outputs.push_back(output);
  return prog;
}

poplar::program::Program lowerTileGatherToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileGatherParams& params,
    const poplar::DebugContext& debug_context) {
  if (inputs.size() != 1) {
    throw poputil::poplibs_error(
        "IPU tile gather expecting a single input tensor.");
  }
  const auto& input = inputs[0];
  const auto item_shape = input[0].shape();
  const auto item_type = input.elementType();
  const size_t num_tiles = params.tiles.size();

  // Create the output tensor per gather index, then concat.
  auto seq = poplar::program::Sequence();
  // All output slices
  std::vector<poplar::Tensor> output_slices;
  // Slices requiring copying.
  std::vector<poplar::Tensor> input_copy_slices;
  std::vector<poplar::Tensor> output_copy_slices;
  for (std::size_t idx = 0; idx < params.tiles.size(); ++idx) {
    const auto gather_idx = params.indices[idx];
    // Get the proper item at the gather index.
    const auto input_item = input[gather_idx];
    const auto input_tile = params.previous_tiles[gather_idx];
    const auto output_tile = params.tiles[idx];
    if (input_tile == output_tile) {
      // No copy => using directly the existing data on the tile.
      output_slices.push_back(input_item.expand({0}));
    } else {
      // New Poplar tensor + copy to the proper tile.
      auto output_item =
          graph.addVariable(item_type, item_shape, debug_context);
      graph.setTileMapping(output_item, output_tile);
      input_copy_slices.push_back(input_item.expand({0}));
      output_copy_slices.push_back(output_item.expand({0}));
      output_slices.push_back(output_item.expand({0}));
    }
  }
  // Copy input to output.
  auto input_copy = poplar::concat(input_copy_slices);
  auto output_copy = poplar::concat(output_copy_slices);
  seq.add(poplar::program::Copy(input_copy, output_copy, false, debug_context));
  // Full gather output tensor.
  auto output = poplar::concat(output_slices);
  outputs.push_back(output);
  return seq;
}

poplar::program::Program lowerTileDataBarrierToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileDataBarrierParams& params,
    const poplar::DebugContext& debug_context) {
  if (inputs.size() < 1) {
    throw poputil::poplibs_error(
        "IPU tile data barrier expecting multiple input tensors.");
  }
  // Half precision different on IPU model.
  const auto& target = graph.getTarget();
  const bool is_half_accurate =
      (target.getTargetType() == poplar::TargetType::IPU);

  // Association of barrier tensors per tile.
  std::vector<std::vector<poplar::Tensor>> tensors_per_tiles(params.max_tile +
                                                             1);
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    // Reinterpret input tensor to a reference type.
    const auto& in_reinterpret =
        tileBarrierReinterpretTensor(inputs[idx], is_half_accurate);
    const auto& tiles = params.inputs_tiles[idx];
    for (size_t k = 0; k < tiles.size(); ++k) {
      // Flatten the tensor on every tile to 1D.
      tensors_per_tiles[tiles[k]].push_back(in_reinterpret[k].flatten());
    }
  }

  auto prog = poplar::program::Sequence();
  poplar::ComputeSet cs = graph.addComputeSet(debug_context);
  for (TileIndexType tile = 0; tile < TileIndexType(tensors_per_tiles.size());
       ++tile) {
    const auto& tensors = tensors_per_tiles[tile];
    if (tensors.size() == 0) {
      continue;
    }
    // Add barrier vertex on the tile.
    auto v = graph.addVertex(cs, params.vname);
    graph.setTileMapping(v, tile);
    graph.setPerfEstimate(v, 14);
    // Map collection of tensors to vertex IO.
    graph.connect(v["data"], tensors);
  }
  prog.add(poplar::program::Execute(cs, debug_context));
  outputs = inputs;
  return prog;
}

poplar::program::Program lowerTileConstantShardedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileConstantParams& params,
    const poplar::DebugContext& debug_context) {
  const std::string raw_values = params.data.decode();
  const auto raw_values_ref =
      poplar::ArrayRef<char>(raw_values.data(), raw_values.size());
  auto t =
      createShardedConstantTensor(graph, params.aval.dtype, params.aval.shape,
                                  raw_values_ref, params.tiles, debug_context);
  outputs.push_back(t);
  return poplar::program::Sequence();
}

poplar::program::Program lowerTileConstantReplicatedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileConstantParams& params,
    const poplar::DebugContext& debug_context) {
  const std::string raw_values = params.data.decode();
  const auto raw_values_ref =
      poplar::ArrayRef<char>(raw_values.data(), raw_values.size());
  auto t = createReplicatedConstantTensor(graph, params.aval.dtype,
                                          params.aval.shape, raw_values_ref,
                                          params.tiles, debug_context);
  outputs.push_back(t);
  return poplar::program::Sequence();
}

}  // namespace ipu
