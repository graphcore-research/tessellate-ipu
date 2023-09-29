// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#include <json/json.hpp>
#include <poplar/Graph.hpp>

#include "base_types.hpp"

namespace ipu {

/**
 * @brief Make a (readable/clean) tile op debug prefix.
 * Help having a more readable naming in PopVision profile.
 */
std::string makeTileOpDebugPrefix(const std::string& raw_debug_prefix,
                                  const std::string& basename);

/**
 * @brief IPU tile gather op parameters.
 */
struct TileGatherParams {
  /** Previous input tile mapping (if existing). */
  TileArrayType previous_tiles;
  /** Gather indices. */
  TileArrayType indices;
  /** New tile mapping */
  TileArrayType tiles;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileGatherParams, previous_tiles, indices,
                                   tiles)

/**
 * @brief Tile data Poplar barrier parameters.
 */
struct TileDataBarrierParams {
  /** Vertex name to use. */
  std::string vname;
  /** Input tensors tiles. */
  std::vector<TileArrayType> inputs_tiles;
  /** Max tile index used by inputs. */
  TileIndexType max_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileDataBarrierParams, vname, inputs_tiles,
                                   max_tile)

/**
 * @brief IPU tile constant (replicated or sharded) parameters.
 */
struct TileConstantParams {
  /** Abstract shaped array (per tile). */
  ShapedArray aval;
  /** Tile mapping of the constant. */
  TileArrayType tiles;
  /** Raw data, encoded as base64. */
  Base64Data data = Base64Data();
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileConstantParams, aval, tiles, data)

/**
 * @brief Reinterpret tensor to a reference type used in the tile data barrier.
 * @param t Poplar tensor to reinterpret.
 * @param is_half_accurate Is FP16/half accurate? On IPU model, float is used
 * for simulating half.
 */
poplar::Tensor tileBarrierReinterpretTensor(const poplar::Tensor& t,
                                            bool is_half_accurate);

/**
 * @brief Lower `tile_put_sharded` to a Poplar program.
 */
poplar::program::Program lowerTilePutShardedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileArrayType& tile_array,
    const poplar::DebugContext& debug_context);
/**
 * @brief Lower `tile_put_replicated` to a Poplar program.
 */
poplar::program::Program lowerTilePutReplicatedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileArrayType& tile_array,
    const poplar::DebugContext& debug_context);
/**
 * @brief Lower `tile_gather` to a Poplar program.
 */
poplar::program::Program lowerTileGatherToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileGatherParams& params,
    const poplar::DebugContext& debug_context);
/**
 * @brief Lower `tile_data_barrier` to a Poplar program.
 */
poplar::program::Program lowerTileDataBarrierToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileDataBarrierParams& params,
    const poplar::DebugContext& debug_context);
/**
 * @brief Lower `tile_constant_sharded` to a Poplar program.
 */
poplar::program::Program lowerTileConstantShardedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileConstantParams& params,
    const poplar::DebugContext& debug_context);
/**
 * @brief Lower `tile_constant_replicated` to a Poplar program.
 */
poplar::program::Program lowerTileConstantReplicatedToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileConstantParams& params,
    const poplar::DebugContext& debug_context);

}  // namespace ipu
