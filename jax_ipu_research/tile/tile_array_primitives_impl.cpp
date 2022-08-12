// cppimport
// NOTE: comment necessary for automatic JIT compilation of the module!
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <ipu_custom_primitive.hpp>

// Standard tile index used.
using TileIndexType = int32_t;

/**
 * @brief Base class for tile put primitives, with common features.
 */
class TilePutBase : public jax::ipu::PrimitiveInterface {
 public:
  /**
   * @brief Extract (and copy) the tile array from raw attributes.
   */
  static std::vector<TileIndexType> extractTileArray(
      const std::string& attributes) {
    const size_t tile_array_size = attributes.size() / sizeof(TileIndexType);
    const TileIndexType* ptr_tile_array =
        reinterpret_cast<const TileIndexType*>(attributes.data());
    std::vector<TileIndexType> tile_array;
    std::copy(ptr_tile_array, ptr_tile_array + tile_array_size,
              std::back_inserter(tile_array));
    return tile_array;
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
          "IPU tile put sharded expected a single input tensor.");
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
    auto output = graph.addVariable(input.elementType(), input.shape());
    for (size_t idx = 0; idx < tile_array.size(); idx++) {
      graph.setTileMapping(output[idx], tile_array[idx]);
    }
    // Copy data tensor into the output.
    auto prog = poplar::program::Copy(input, output);
    outputs.push_back(output);
    return prog;
  }
};

/**
 * @brief IPU tile put sharded primitive: sharding an array over tiles on
 * the first axis.
 */
class TilePutReplicatedPrimitive : public TilePutBase {
 public:
  using TileIndexType = int32_t;

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
          "IPU tile put replicated expected a single input tensor.");
    }
    static_assert(sizeof(TileIndexType) == 4);
    auto input = inputs[0];

    const auto tile_array = extractTileArray(attributes);
    // Create output tensor, with proper tile mapping.
    auto input_broadcasted = input.expand({0}).broadcast(tile_array.size(), 0);
    auto output =
        graph.addVariable(input.elementType(), input_broadcasted.shape());
    for (size_t idx = 0; idx < tile_array.size(); idx++) {
      graph.setTileMapping(output[idx], tile_array[idx]);
    }
    // Copy data tensor into the output.
    auto prog = poplar::program::Copy(input_broadcasted, output);
    outputs.push_back(output);
    return prog;
  }
};

// Export the IPU JAX primitives in the shared library.
EXPORT_IPU_JAX_PRIMITIVE(TilePutShardedPrimitive);
EXPORT_IPU_JAX_PRIMITIVE(TilePutReplicatedPrimitive);

// Declare a pybind11, to provide easy compilation & import from Python.
PYBIND11_MODULE(tile_array_primitives_impl, m) {
  pybind11::class_<TilePutShardedPrimitive>(m, "TilePutShardedPrimitive")
      .def_static("metadata", &TilePutShardedPrimitive::metadata,
                  pybind11::arg("num_inputs"));
  pybind11::class_<TilePutReplicatedPrimitive>(m, "TilePutReplicatedPrimitive")
      .def_static("metadata", &TilePutReplicatedPrimitive::metadata,
                  pybind11::arg("num_inputs"));
}

// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-fPIC', '-O2', '-Wall']
cfg['libraries'] = ['poplar', 'poputil', 'poprand', 'popops']
cfg['include_dirs'] = []
setup_pybind11(cfg)
%>
*/
