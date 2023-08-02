// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <nanobind/nanobind.h>

#include "ipu_custom_primitive.hpp"
#include "tile_array_ops.hpp"
#include "tile_map_ops.hpp"

using namespace ipu;

/**
 * @brief Base class for tile put primitives, with common features.
 */
class TilePutBase : public jax::ipu::PrimitiveInterface {
 public:
  /**
   * @brief Extract (and copy) the tile array from raw JSON attributes.
   */
  static std::vector<TileIndexType> extractTileArray(
      const std::string& attributes) {
    return ipu::from_json_str<std::vector<TileIndexType>>(attributes);
  }
};

/**
 * @brief IPU tile put sharded primitive: sharding an array over tiles on
 * the first axis.
 */
class TilePutShardedPrimitive : public TilePutBase {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{.num_inputs = num_inputs,
                                       .is_elementwise = true,
                                       .is_stateless = true,
                                       .is_hashable = true,
                                       .input_to_output_tensor_aliasing = {{}},
                                       .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    // Passing the tile array as attributes.
    const auto tile_array = extractTileArray(attributes);
    return lowerTilePutShardedToPoplar(graph, inputs, outputs, tile_array,
                                       debug_context);
  }
};

/**
 * @brief IPU tile put replicated primitive: replicating an array over tiles on
 * the first axis.
 */
class TilePutReplicatedPrimitive : public TilePutBase {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{
        .num_inputs = num_inputs,
        .is_elementwise = false,  // Broadcasting over the first axis.
        .is_stateless = true,
        .is_hashable = true,
        .input_to_output_tensor_aliasing = {{}},
        .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    const auto tile_array = extractTileArray(attributes);
    return lowerTilePutReplicatedToPoplar(graph, inputs, outputs, tile_array,
                                          debug_context);
  }
};

/**
 * @brief IPU tile array (general) gather op across tiles.
 */
class TileGatherPrimitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{
        .num_inputs = num_inputs,
        .is_elementwise = true,  // Broadcasting over the first axis.
        .is_stateless = true,
        .is_hashable = true,
        .input_to_output_tensor_aliasing = {{}},
        .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    // Tile gather parameters.
    const auto params = ipu::from_json_str<TileGatherParams>(attributes);
    return lowerTileGatherToPoplar(graph, inputs, outputs, params,
                                   debug_context);
  }
};

/**
 * @brief IPU tile array data barrier: force to introduce a barrier in Poplar
 * with a single compute set across tiles.
 */
class TileDataBarrierPrimitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    // TODO: input/output aliasing.
    return jax::ipu::PrimitiveMetadata{
        .num_inputs = num_inputs,
        .is_elementwise = false,  // Broadcasting over the first axis.
        .is_stateless = true,
        .is_hashable = true,
        .input_to_output_tensor_aliasing = {{}},
        .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    // Tile barrier parameters (with tile sharding).
    const auto params = ipu::from_json_str<TileDataBarrierParams>(attributes);
    return lowerTileDataBarrierToPoplar(graph, inputs, outputs, params,
                                        debug_context);
  }
};

/**
 * @brief IPU tile constant replicated primitive: replicating a constant array
 * over tiles (on the first axis).
 */
class TileConstantReplicatedPrimitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{.num_inputs = num_inputs,
                                       .is_elementwise = false,
                                       .is_stateless = true,
                                       .is_hashable = true,
                                       .input_to_output_tensor_aliasing = {{}},
                                       .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    const auto params = ipu::from_json_str<TileConstantParams>(attributes);
    return lowerTileConstantReplicatedToPoplar(graph, inputs, outputs, params,
                                               debug_context);
  }
};

/**
 * @brief IPU tile constant sharded primitive: sharding a constant array
 * over tiles (on the first axis).
 */
class TileConstantShardedPrimitive : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    return jax::ipu::PrimitiveMetadata{.num_inputs = num_inputs,
                                       .is_elementwise = false,
                                       .is_stateless = true,
                                       .is_hashable = true,
                                       .input_to_output_tensor_aliasing = {{}},
                                       .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    const auto params = ipu::from_json_str<TileConstantParams>(attributes);
    return lowerTileConstantShardedToPoplar(graph, inputs, outputs, params,
                                            debug_context);
  }
};

// Export the IPU JAX primitives in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(TilePutShardedPrimitive);
EXPORT_IPU_JAX_PRIMITIVE(TilePutReplicatedPrimitive);
EXPORT_IPU_JAX_PRIMITIVE(TileGatherPrimitive);
EXPORT_IPU_JAX_PRIMITIVE(TileDataBarrierPrimitive);
EXPORT_IPU_JAX_PRIMITIVE(TileConstantReplicatedPrimitive);
EXPORT_IPU_JAX_PRIMITIVE(TileConstantShardedPrimitive);

/**
 * @brief IPU tile put sharded primitive: sharding an array over tiles on
 * the first axis.
 */
class TileMapEquationCall : public jax::ipu::PrimitiveInterface {
 public:
  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    // TODO. check InOut tensors for aliasing.
    return jax::ipu::PrimitiveMetadata{.num_inputs = num_inputs,
                                       .is_elementwise = false,
                                       .is_stateless = true,
                                       .is_hashable = true,
                                       .input_to_output_tensor_aliasing = {},
                                       .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(debug_prefix);
    const auto tile_equation =
        ipu::from_json_str<ipu::TileMapEquation>(attributes);
    return lowerTileMapCallToPoplar(graph, inputs, outputs, tile_equation,
                                    debug_context);
  }
};

// Export the IPU JAX primitives in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCall);

NB_MODULE(pytessellate_ipu_ops_jax, m) {
  // Avoid leak warning from the library.
  nanobind::set_leak_warnings(false);
  // Tile array operations.
  nanobind::class_<TilePutShardedPrimitive>(m, "TilePutShardedPrimitive");
  nanobind::class_<TilePutReplicatedPrimitive>(m, "TilePutReplicatedPrimitive");
  nanobind::class_<TileGatherPrimitive>(m, "TileGatherPrimitive");
  nanobind::class_<TileDataBarrierPrimitive>(m, "TileDataBarrierPrimitive");
  nanobind::class_<TileConstantReplicatedPrimitive>(
      m, "TileConstantReplicatedPrimitive");
  nanobind::class_<TileConstantShardedPrimitive>(
      m, "TileConstantShardedPrimitive");
  // Tile map operations.
  nanobind::class_<TileMapEquationCall>(m, "TileMapEquationCall");
}
