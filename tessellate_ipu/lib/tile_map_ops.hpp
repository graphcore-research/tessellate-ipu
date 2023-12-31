// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <algorithm>
#include <half/half.hpp>
#include <json/json.hpp>
#include <optional>

#include "base_types.hpp"
#include "tile_array_utils.hpp"

namespace ipu {
using json = nlohmann::json;

/**
 * @brief Vertex IO tensor type.
 */
enum class VertexIOType : int {
  In = 0,    // Input only tensor.
  Out = 1,   // Output only tensor.
  InOut = 2  // Input/output tensor.
};

/**
 * @brief 1d tensor slice.
 */
struct TensorSlice {
  /** 1d begin index. */
  std::size_t begin;
  /** 1d end index. */
  std::size_t end;

  /**
   * @brief Make a collection of slices corresponding to a 2D tensor of shape
   * (d0, d1).
   */
  static std::vector<TensorSlice> makeTensor2dSlices(size_t dim0, size_t dim1) {
    std::vector<TensorSlice> slices;
    slices.reserve(dim0);
    for (std::size_t idx = 0; idx < dim0; ++idx) {
      slices.push_back(TensorSlice{idx * dim1, (idx + 1) * dim1});
    }
    return slices;
  }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TensorSlice, begin, end)

/**
 * @brief Vertex IO tensor info.
 */
struct VertexIOInfo {
  /** Name of the vertex IO tensor. */
  std::string name;
  /** IO tensor iotype. */
  VertexIOType iotype;
  /** IO tensor aval. */
  ShapedArray aval;
  /** Optional data for constant tensors. */
  Base64Data constant_data = Base64Data();
  /** Slices, in the case of 2d tensor input. */
  std::vector<TensorSlice> slices2d;
  /** Is the vertex IO tensor just a scalar? */
  bool is_scalar = false;

  /** Default constructors/assignment. */
  VertexIOInfo() noexcept = default;
  VertexIOInfo(VertexIOInfo&&) noexcept = default;
  VertexIOInfo(const VertexIOInfo&) = default;
  VertexIOInfo& operator=(VertexIOInfo&&) noexcept = default;
  VertexIOInfo& operator=(const VertexIOInfo&) = default;

  VertexIOInfo(const std::string& _name, VertexIOType _iotype,
               const ShapeType& _shape, IpuType _dtype,
               const Base64Data& _constant_data,
               const std::vector<TensorSlice>& _slices2d)
      : name{_name},
        iotype{_iotype},
        aval{_shape, _dtype},
        constant_data{_constant_data},
        slices2d{_slices2d} {}

  VertexIOInfo(const std::string& _name, VertexIOType _iotype,
               const ShapedArray& _aval, const Base64Data& _constant_data,
               const std::vector<TensorSlice>& _slices2d)
      : name{_name},
        iotype{_iotype},
        aval{_aval},
        constant_data{_constant_data},
        slices2d{_slices2d} {}
  /**
   * @brief Build a vertex IO info (with vertex second dim info).
   */
  VertexIOInfo(const std::string& _name, VertexIOType _iotype,
               const ShapeType& _shape, IpuType _dtype, int64_t _vertex_dim2,
               const Base64Data& _constant_data)
      : name{_name},
        iotype{_iotype},
        aval{_shape, _dtype},
        constant_data{_constant_data} {
    // Generate 2d slices when required.
    if (_vertex_dim2 > 0) {
      slices2d = TensorSlice::makeTensor2dSlices(aval.size() / _vertex_dim2,
                                                 _vertex_dim2);
    }
    // Negative => code for scalar.
    if (_vertex_dim2 < 0) {
      is_scalar = true;
    }
    // Zero => normal flattened case.
  }

  /**
   * @brief Build a vertex IO info (with vertex second dim info).
   */
  static VertexIOInfo makeVertexIOInfo(const std::string& name,
                                       VertexIOType iotype,
                                       const ShapeType& shape, IpuType dtype,
                                       std::size_t vertex_dim2,
                                       const Base64Data& constant_data) {
    auto ioinfo = VertexIOInfo{
        name, iotype, ShapedArray{shape, dtype}, constant_data, {}};
    // Generate 2d slices when required.
    if (vertex_dim2 > 0) {
      ioinfo.slices2d = TensorSlice::makeTensor2dSlices(
          ioinfo.aval.size() / vertex_dim2, vertex_dim2);
    }
    return ioinfo;
  }

  /**
   * @brief Is it a constant vertex IO input?
   */
  bool isConstantInput() const noexcept {
    FMT_ASSERT(constant_data.empty() || iotype == VertexIOType::In,
               "Constant IO tensor can only be input.");
    return !constant_data.empty() && iotype == VertexIOType::In;
  }

  /**
   * @brief Is it a 2d IO tensor, i.e. vertext `vector<IO<vector...>>`.
   */
  bool isTensor2d() const noexcept { return !slices2d.empty(); }

  /**
   * @brief Reshape a tensor to the proper rank for vertex connection.
   *
   * This bit of logic is necessary as Poplar vertices only support:
   *    rank 0: i.e. scalar entry;
   *    rank 1: flattened array;
   *    rank 2: collection of tensor slices;
   */
  poplar::Tensor connectReshape(const poplar::Tensor& t) const {
    if (is_scalar) {
      if (t.numElements() != 1) {
        throw std::logic_error(
            "Expecting a single scalar element to connect to the vertex.");
      }
      return t.flatten()[0];
    }
    if (slices2d.empty()) {
      // Rank 1 (no 2d slices): flatten the IO tensor.
      return t.flatten();
    } else {
      // 2d slices: extract the sub-tensors, and re-concat.
      auto flat_tensor = t.flatten();
      std::vector<poplar::Tensor> tensors;
      for (const auto& slice : slices2d) {
        tensors.push_back(flat_tensor.slice(slice.begin, slice.end));
      }
      // Concat and reshape to 2d vertex IO tensor.
      auto t = poplar::concat(tensors);
      const std::size_t vertex_dim1 = slices2d.size();
      const std::size_t vertex_dim2 = t.numElements() / vertex_dim1;
      return t.reshape({vertex_dim1, vertex_dim2});
    }
  }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VertexIOInfo, name, iotype, aval,
                                   constant_data, slices2d, is_scalar)

inline bool operator==(const VertexIOInfo& lhs, const VertexIOInfo& rhs) {
  return lhs.name == rhs.name && lhs.iotype == rhs.iotype &&
         lhs.aval.shape == rhs.aval.shape && lhs.aval.dtype == rhs.aval.dtype;
  // TODO: compare 2d slices and is_scalar?
}

/**
 * @brief Vertex (static) attribute
 * @tparam T Attribute type.
 */
template <typename T>
struct VertexAttribute {
  /** Name of the attribute. */
  std::string name;
  /** Value of the attribute. */
  T value;
};
using VertexAttributeI32 = VertexAttribute<int32_t>;
using VertexAttributeF32 = VertexAttribute<float>;

template <typename T>
bool operator==(const VertexAttribute<T>& lhs, const VertexAttribute<T>& rhs) {
  return lhs.name == rhs.name && lhs.value == rhs.value;
}
template <typename T>
void to_json(json& j, const VertexAttribute<T>& v) {
  j = json{{"name", v.name}, {"value", v.value}};
}
template <typename T>
void from_json(const json& j, VertexAttribute<T>& v) {
  j.at("name").get_to(v.name);
  j.at("value").get_to(v.value);
}

/**
 * @brief IPU tile map(ped) equation (on the model of JAX equation `JaxprEqn`).
 *
 * This class represent a tile equation mapped on multiple tiles (which the same
 * input/output shapes, and constant attributes).
 *
 * IPU parallelization between tiles: disjoint compute sets should be executed
 * in parallel:
 *   https://graphcore.slack.com/archives/C013LPHPX61/p1661937739927649
 */
struct TileMapEquation {
  /** Primitive name. */
  std::string pname;
  /** Vertex name. */
  std::string vname;

  /** Tiles on which is the equation is mapped. */
  std::vector<TileIndexType> tiles;

  /** Input vertex tensor infos (per tile). */
  std::vector<VertexIOInfo> inputs_info;
  /** Output vertex tensor infos (per tile). */
  std::vector<VertexIOInfo> outputs_info;

  /** Attributes, with different types. */
  std::vector<VertexAttributeI32> attributes_i32;
  std::vector<VertexAttributeF32> attributes_f32;

  /** Temporary vertex scratch space name. */
  std::string tmp_space_name = "";

  /** Temporary vertex scratch space (empty by default). */
  ShapedArray tmp_space_aval =
      ipu::ShapedArray{ipu::ShapeType{0}, IpuType::UNSIGNED_CHAR};

  /** (Optional) IPU gp vertex (absolute) filename. */
  std::string gp_filename = "";
  /** Vertex performance estimate (optional). */
  uint64_t perf_estimate = 0;
  /** Synchronization of tiles before the compute set. */
  bool sync = false;

  /**
   * @brief Does it require temporary vertex space?
   */
  bool useTmpSpace() const { return tmp_space_aval.size() > 0; }

  /**
   * @brief Allocate all input tensors (including missing constant).
   * @param graph Poplar graph.
   * @param inputs Pre-existing input tensors.
   * @return Collection of input tensors.
   */
  std::vector<poplar::Tensor> allocateInputTensors(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      const poplar::DebugContext& debug_context) const;

  /**
   * @brief Allocate output (or use existing input) tensors.
   * @param graph Poplar graph.
   * @param inputs Corresponding tensor inputs.
   * @return Collection of output tensors.
   */
  std::vector<poplar::Tensor> allocateOutputTensors(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
      const poplar::DebugContext& debug_context) const;

  /**
   * @brief Allocate the temporary-scratch space tensor (if used).
   */
  std::optional<poplar::Tensor> allocateTmpSpaceTensor(
      poplar::Graph& graph, const poplar::DebugContext& debug_context) const;

  /**
   * @brief Add vertex/equation to Poplar graph & compute set.
   *
   * @param graph Poplar graph.
   * @param prog Poplar sequence program.
   * @param inputs Vector of (sharded) input tensors (including constant
   * tensors).
   * @param outputs Vector of (sharded) output tensors (already allocated).
   * @param debug_prefix Debug context prefix.
   */
  void add(poplar::Graph& graph, poplar::program::Sequence& prog,
           const std::vector<poplar::Tensor>& inputs,
           const std::vector<poplar::Tensor>& outputs,
           const poplar::DebugContext& debug_prefix) const;

  /**
   * @brief Add vertex/equation to Poplar graph & compute set (with outputs
   * allocated).
   *
   * @param graph Poplar graph.
   * @param prog Poplar sequence program.
   * @param inputs Vector of (sharded) input tensors.
   * @param debug_prefix Debug context prefix.
   * @return Vector of (tile sharded) output tensors.
   */
  std::vector<poplar::Tensor> add(
      poplar::Graph& graph, poplar::program::Sequence& prog,
      const std::vector<poplar::Tensor>& inputs,
      const poplar::DebugContext& debug_prefix) const;

  /** Number of inputs in the equation. */
  std::size_t numInputs() const noexcept { return inputs_info.size(); }
  /** Number of outputs in the equation. */
  std::size_t numOutputs() const noexcept { return outputs_info.size(); }
  /** Number of inputs/outputs (i.e. aliased) in the equation. */
  std::size_t numInOuts() const;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileMapEquation, pname, vname, tiles,
                                   inputs_info, outputs_info, attributes_i32,
                                   attributes_f32, tmp_space_name,
                                   tmp_space_aval, gp_filename, perf_estimate,
                                   sync)

/**
 * @brief Lower `tile_map` call to Poplar.
 * @param graph Poplar graph to update.
 * @param inputs List of inputs.
 * @param outputs List of outputs, to update.
 * @param tile_map_eqn TileMapEquation info.
 * @param debug_prefix Poplar (raw) debug prefix.
 * @return Poplar program.
 */
poplar::program::Program lowerTileMapCallToPoplar(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
    std::vector<poplar::Tensor>& outputs, const TileMapEquation& tile_map_eqn,
    const std::string& debug_prefix);

}  // namespace ipu
