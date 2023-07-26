// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#include <json/json.hpp>

#include "base_types.hpp"
#include "tile_array_utils.hpp"

namespace ipu {
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

}  // namespace ipu
