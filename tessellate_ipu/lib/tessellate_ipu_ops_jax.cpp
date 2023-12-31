// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <nanobind/nanobind.h>

#include <iostream>

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
                                       .allocating_indices = {0}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_put_sharded"));
    // Passing the tile array as attributes.
    const auto tile_array = extractTileArray(attributes);
    return lowerTilePutShardedToPoplar(graph, inputs, outputs, tile_array,
                                       debug_context);
  }

  static poplar::Tensor allocator(poplar::Graph& graph, std::uint32_t operand,
                                  const std::vector<size_t>& shape,
                                  poplar::Type type,
                                  const std::string& attributes,
                                  const std::string& debug_prefix) {
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_put_sharded"));
    const auto tile_array = extractTileArray(attributes);
    const auto item_shape =
        poplar::ArrayRef<std::size_t>(shape.data() + 1, shape.size() - 1);
    // If not allocated => already pre-allocate input with proper tile mapping.
    // TODO: fix (unnecessary) on-tile-copy when doing that?
    return createShardedVariable(graph, type, item_shape, tile_array,
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
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_put_replicated"));
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
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_gather"));
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
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_data_barrier"));
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
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_constant_replicated"));
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
    const auto debug_context = poplar::DebugContext(
        makeTileOpDebugPrefix(debug_prefix, "tile_constant_sharded"));
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
template <int64_t TNumInOutAliasingArgs>
class TileMapEquationCall : public jax::ipu::PrimitiveInterface {
 public:
  // Static number of inout arguments.
  static constexpr int64_t NumInOutAliasingArgs = TNumInOutAliasingArgs;

  static jax::ipu::PrimitiveMetadata metadata(std::uint32_t num_inputs) {
    if (num_inputs < NumInOutAliasingArgs) {
      throw std::runtime_error(
          "Inconsistent number of inputs and in/out aliased arguments.");
    }
    // Input/output aliasing map.
    // Note: only support 1-to-1 mapping of same index, on first arguments.
    std::map<std::int64_t, std::int64_t> inout_aliasing;
    for (int64_t idx = 0; idx < NumInOutAliasingArgs; ++idx) {
      inout_aliasing[idx] = idx;
    }
    return jax::ipu::PrimitiveMetadata{
        .num_inputs = num_inputs,
        .is_elementwise = false,
        .is_stateless = true,
        .is_hashable = true,
        .input_to_output_tensor_aliasing = inout_aliasing,
        .allocating_indices = {}};
  }

  static poplar::program::Program program(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      std::vector<poplar::Tensor>& outputs, const std::string& attributes,
      const std::string& debug_prefix) {
    const auto tile_equation =
        ipu::from_json_str<ipu::TileMapEquation>(attributes);
    return lowerTileMapCallToPoplar(graph, inputs, outputs, tile_equation,
                                    debug_prefix);
  }
};

// Manually un-rolling for up to 2 InOut arguments (for now...)
// No other way due to metadata being a static class function.
using TileMapEquationCallInOut0 = TileMapEquationCall<0>;
using TileMapEquationCallInOut1 = TileMapEquationCall<1>;
using TileMapEquationCallInOut2 = TileMapEquationCall<2>;
using TileMapEquationCallInOut3 = TileMapEquationCall<3>;
using TileMapEquationCallInOut4 = TileMapEquationCall<4>;

// Export the IPU JAX primitives in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCallInOut0);
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCallInOut1);
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCallInOut2);
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCallInOut3);
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCallInOut4);

template <typename T>
decltype(auto) makeTileMapEquationCallBindings(nanobind::module_& m,
                                               const char* name) {
  nanobind::class_<T>(m, name).def_ro_static("NumInOutAliasingArgs",
                                             &T::NumInOutAliasingArgs);
}

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
  // Manually un-rolling for up to 2 InOut arguments (for now...).
  makeTileMapEquationCallBindings<TileMapEquationCallInOut0>(
      m, "TileMapEquationCallInOut0");
  makeTileMapEquationCallBindings<TileMapEquationCallInOut1>(
      m, "TileMapEquationCallInOut1");
  makeTileMapEquationCallBindings<TileMapEquationCallInOut2>(
      m, "TileMapEquationCallInOut2");
  makeTileMapEquationCallBindings<TileMapEquationCallInOut3>(
      m, "TileMapEquationCallInOut3");
  makeTileMapEquationCallBindings<TileMapEquationCallInOut4>(
      m, "TileMapEquationCallInOut4");
  // Export max in/out aliasing args constant.
  m.attr("TileMapMaxInOutAliasingArgs") = nanobind::int_(4);
}
