// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "base_types.hpp"
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
    const auto debugContext = poplar::DebugContext(debug_prefix);
    if (inputs.size() != 1) {
      throw poputil::poplibs_error(
          "IPU tile put sharded expecting a single input tensor.");
    }
    static_assert(sizeof(TileIndexType) == 4);
    auto input = inputs[0];

    // Passing the tile array as attributes.
    const auto tile_array = extractTileArray(attributes);
    if (input.shape()[0] != tile_array.size()) {
      throw poputil::poplibs_error(
          fmt::format("IPU tile put sharding: inconsistent input size {} and "
                      "tiles length {}.",
                      input.shape()[0], tile_array.size()));
    }

    // Create output tensor, with proper tile mapping.
    // TODO: link to Slack discussion on VarRegion contiguity.
    auto output = createShardedVariable(graph, input.elementType(),
                                        input[0].shape(), tile_array);
    // Copy data tensor into the output.
    auto prog = poplar::program::Copy(input, output);
    outputs.push_back(output);
    return prog;
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
    const auto debugContext = poplar::DebugContext(debug_prefix);
    if (inputs.size() != 1) {
      throw poputil::poplibs_error(
          "IPU tile put replicated expecting a single input tensor.");
    }
    static_assert(sizeof(TileIndexType) == 4);
    auto input = inputs[0];

    const auto tile_array = extractTileArray(attributes);
    // Create output tensor, with proper tile mapping.
    auto input_broadcasted = input.expand({0}).broadcast(tile_array.size(), 0);
    auto output = createShardedVariable(graph, input.elementType(),
                                        input.shape(), tile_array);
    // Copy data tensor into the output.
    auto prog = poplar::program::Copy(input_broadcasted, output, false);
    outputs.push_back(output);
    return prog;
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
    if (inputs.size() != 1) {
      throw poputil::poplibs_error(
          "IPU tile gather expecting a single input tensor.");
    }
    const auto& input = inputs[0];
    const auto item_shape = input[0].shape();
    const auto item_type = input.elementType();

    // Tile gather parameters.
    const auto params = ipu::from_json_str<TileGatherParams>(attributes);
    // Create the output tensor per gather index, then concat.
    auto seq = poplar::program::Sequence();
    std::vector<poplar::Tensor> output_slices;
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
        seq.add(poplar::program::Copy(input_item, output_item));
        output_slices.push_back(output_item.expand({0}));
      }
    }
    auto output = poplar::concat(output_slices);
    outputs.push_back(output);
    return seq;
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
    if (inputs.size() < 1) {
      throw poputil::poplibs_error(
          "IPU tile data barrier expecting multiple input tensors.");
    }
    // Half precision different on IPU model.
    const auto& target = graph.getTarget();
    const bool is_half_accurate =
        (target.getTargetType() == poplar::TargetType::IPU);
    // Tile barrier parameters (with tile sharding).
    const auto params = ipu::from_json_str<TileDataBarrierParams>(attributes);

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
    const std::string raw_values = params.data.decode();
    const auto raw_values_ref =
        poplar::ArrayRef<char>(raw_values.data(), raw_values.size());
    auto t = createReplicatedConstantTensor(graph, params.aval.dtype,
                                            params.aval.shape, raw_values_ref,
                                            params.tiles, debug_context);
    outputs.push_back(t);
    return poplar::program::Sequence();
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
    const std::string raw_values = params.data.decode();
    const auto raw_values_ref =
        poplar::ArrayRef<char>(raw_values.data(), raw_values.size());
    auto t = createShardedConstantTensor(graph, params.aval.dtype,
                                         params.aval.shape, raw_values_ref,
                                         params.tiles, debug_context);
    outputs.push_back(t);
    return poplar::program::Sequence();
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
    // Deserialize tile mapped equation, and add to the graph.
    const auto tile_equation =
        ipu::from_json_str<ipu::TileMapEquation>(attributes);
    auto prog = poplar::program::Sequence();
    // IPU tiles synchronization before compute set.
    if (tile_equation.sync) {
      const auto sync_type = poplar::SyncType::INTERNAL;
      prog.add(poplar::program::Sync(sync_type, debug_context));
    }
    outputs = tile_equation.add(graph, prog, inputs, debug_context);
    return prog;
  }
};

// Export the IPU JAX primitives in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(TileMapEquationCall);

PYBIND11_MODULE(pytessellate_ipu_ops_jax, m) {
  // Tile array operations.
  pybind11::class_<TilePutShardedPrimitive>(m, "TilePutShardedPrimitive")
      .def_static("metadata", &TilePutShardedPrimitive::metadata,
                  pybind11::arg("num_inputs"));
  pybind11::class_<TilePutReplicatedPrimitive>(m, "TilePutReplicatedPrimitive")
      .def_static("metadata", &TilePutReplicatedPrimitive::metadata,
                  pybind11::arg("num_inputs"));
  pybind11::class_<TileGatherPrimitive>(m, "TileGatherPrimitive")
      .def_static("metadata", &TileGatherPrimitive::metadata,
                  pybind11::arg("num_inputs"));
  pybind11::class_<TileDataBarrierPrimitive>(m, "TileDataBarrierPrimitive")
      .def_static("metadata", &TileDataBarrierPrimitive::metadata,
                  pybind11::arg("num_inputs"));
  pybind11::class_<TileConstantReplicatedPrimitive>(
      m, "TileConstantReplicatedPrimitive")
      .def_static("metadata", &TileConstantReplicatedPrimitive::metadata,
                  pybind11::arg("num_inputs"));
  pybind11::class_<TileConstantShardedPrimitive>(m,
                                                 "TileConstantShardedPrimitive")
      .def_static("metadata", &TileConstantShardedPrimitive::metadata,
                  pybind11::arg("num_inputs"));

  // Tile map operations.
  pybind11::class_<TileMapEquationCall>(m, "TileMapEquationCall")
      .def_static("metadata", &TileMapEquationCall::metadata,
                  pybind11::arg("num_inputs"));
}
