// cppimport
// NOTE: comment necessary for automatic JIT compilation of the module!
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <algorithm>
#include <half/half.hpp>
#include <ipu_custom_primitive.hpp>
#include <json/json.hpp>
#include <optional>

#include "tile_array_utils.hpp"
#include "tile_dot_vertex_utils.hpp"

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
   */
  poplar::Tensor connectReshape(const poplar::Tensor& t) const {
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
                                   constant_data, slices2d)

bool operator==(const VertexIOInfo& lhs, const VertexIOInfo& rhs) {
  return lhs.name == rhs.name && lhs.iotype == rhs.iotype &&
         lhs.aval.shape == rhs.aval.shape && lhs.aval.dtype == rhs.aval.dtype;
  // TODO: compare 2d slices.
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
 * @brief Make Python bindings of VertexAttribute class.
 */
template <typename T>
decltype(auto) makeVertexAttributeBindings(pybind11::module& m,
                                           const char* name) {
  using VertexAttrType = VertexAttribute<T>;
  pybind11::class_<VertexAttrType>(m, name)
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string&, T>(), pybind11::arg("name"),
           pybind11::arg("value"))
      .def(pybind11::self == pybind11::self)
      .def("to_json_str",
           [](const VertexAttrType& v) { return to_json_str(v); })
      .def_static(
          "from_json_str",
          [](const std::string& j) { return from_json_str<VertexAttrType>(j); })
      .def_readwrite("name", &VertexAttrType::name)
      .def_readwrite("value", &VertexAttrType::value);
}

using VertexAttributeI32 = VertexAttribute<int32_t>;
using VertexAttributeF32 = VertexAttribute<float>;

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
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs) const {
    FMT_ASSERT(inputs.size() <= inputs_info.size(),
               "Inconsistent input vector size.");

    std::vector<poplar::Tensor> inputs_all;
    int input_idx = 0;
    for (const auto& input_info : inputs_info) {
      if (input_info.isConstantInput()) {
        // Create a replicated constant tensor.
        // TODO: support sharded constant as well.
        const std::string raw_values = input_info.constant_data.decode();
        const auto raw_values_ref =
            poplar::ArrayRef<char>(raw_values.data(), raw_values.size());
        auto t = createReplicatedConstantTensor(graph, input_info.aval.dtype,
                                                input_info.aval.shape,
                                                raw_values_ref, this->tiles);
        inputs_all.push_back(t);
      } else {
        // Keep existing input tensor.
        inputs_all.push_back(inputs[input_idx]);
        input_idx++;
      }
    }
    return inputs_all;
  }

  /**
   * @brief Allocate output (or use existing input) tensors.
   * @param graph Poplar graph.
   * @param inputs Corresponding tensor inputs.
   * @return Collection of output tensors.
   */
  std::vector<poplar::Tensor> allocateOutputTensors(
      poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs) const {
    FMT_ASSERT(inputs.size() == inputs_info.size(),
               "Inconsistent input vector size.");

    std::vector<poplar::Tensor> outputs;
    for (const auto& outinfo : outputs_info) {
      if (outinfo.iotype == VertexIOType::InOut) {
        // Find the input tensor used as output.
        const auto it = std::find_if(inputs_info.begin(), inputs_info.end(),
                                     [&outinfo](const VertexIOInfo& ininfo) {
                                       return ininfo.name == outinfo.name;
                                     });
        const auto idx = std::distance(inputs_info.begin(), it);
        outputs.push_back(inputs.at(idx));
      } else if (outinfo.iotype == VertexIOType::Out) {
        // Allocate an output tensor with proper shape.
        outputs.push_back(
            createShardedVariable(graph, toPoplar(outinfo.aval.dtype),
                                  outinfo.aval.shape, this->tiles));
      } else {
        throw std::runtime_error("Unknown IO type for vertex output tensor.");
      }
    }
    return outputs;
  }

  /**
   * @brief Allocate the temporary-scratch space tensor (if used).
   */
  std::optional<poplar::Tensor> allocateTmpSpaceTensor(
      poplar::Graph& graph) const {
    if (!useTmpSpace()) {
      return std::nullopt;
    }
    return createShardedVariable(graph, toPoplar(tmp_space_aval.dtype),
                                 {tmp_space_aval.size()}, this->tiles);
  }

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
           const poplar::DebugContext& debug_prefix) const {
    FMT_ASSERT(inputs.size() == inputs_info.size(),
               "Inconsistent inputs vector size.");
    FMT_ASSERT(outputs.size() == outputs_info.size(),
               "Inconsistent outputs vector size.");
    poplar::DebugContext debug_context(debug_prefix, this->pname);

    // Tensor used for vertex temp. scratch space.
    auto tmp_space_tensor_opt = allocateTmpSpaceTensor(graph);

    poplar::ComputeSet cs = graph.addComputeSet(debug_context);
    for (size_t tidx = 0; tidx < tiles.size(); ++tidx) {
      const auto tile = tiles[tidx];
      // Add vertex on the tile.
      auto v = graph.addVertex(cs, this->vname);
      graph.setTileMapping(v, tile);
      if (perf_estimate > 0) {
        graph.setPerfEstimate(v, perf_estimate);
      }
      // Map/connect vertex input tensors.
      for (size_t k = 0; k < inputs.size(); ++k) {
        const auto& info = inputs_info[k];
        graph.connect(v[info.name], info.connectReshape(inputs[k][tidx]));
      }
      // Map/connect vertex output tensors.
      for (size_t k = 0; k < outputs.size(); ++k) {
        // InOut tensors already mapped. Just need to connect pure output.
        if (outputs_info[k].iotype == VertexIOType::Out) {
          const auto& info = outputs_info[k];
          graph.connect(v[info.name], info.connectReshape(outputs[k][tidx]));
        }
      }
      // Connect tmp scratch space.
      if (tmp_space_tensor_opt.has_value()) {
        auto tmp_space_tensor = tmp_space_tensor_opt.value();
        graph.connect(v[tmp_space_name], tmp_space_tensor[tidx]);
      }
      // Map vertex attributes.
      for (const auto& attr : attributes_i32) {
        graph.setInitialValue(v[attr.name], attr.value);
      }
      for (const auto& attr : attributes_f32) {
        graph.setInitialValue(v[attr.name], attr.value);
      }
    }
    prog.add(poplar::program::Execute(cs, debug_context));
  }

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
      const poplar::DebugContext& debug_prefix) const {
    // All input tensors: i.e. add constant tensors.
    const auto inputs_all = this->allocateInputTensors(graph, inputs);
    // No vertex => assume identity function, i.e. forward inputs.
    if (this->vname.empty()) {
      return inputs_all;
    }
    const auto outputs = this->allocateOutputTensors(graph, inputs);
    this->add(graph, prog, inputs_all, outputs, debug_prefix);
    return outputs;
  }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TileMapEquation, pname, vname, tiles,
                                   inputs_info, outputs_info, attributes_i32,
                                   attributes_f32, tmp_space_name,
                                   tmp_space_aval, gp_filename, perf_estimate,
                                   sync)

}  // namespace ipu

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

// Declare a pybind11, to provide easy compilation & import from Python.
PYBIND11_MODULE(tile_interpreter_primitives_impl, m) {
  using namespace ipu;

  pybind11::enum_<VertexIOType>(m, "IpuVertexIOType", pybind11::arithmetic())
      .value("In", VertexIOType::In)
      .value("Out", VertexIOType::Out)
      .value("InOut", VertexIOType::InOut);
  makeVertexAttributeBindings<int32_t>(m, "IpuVertexAttributeI32");
  makeVertexAttributeBindings<float>(m, "IpuVertexAttributeF32");

  pybind11::class_<TensorSlice>(m, "IpuTensorSlice")
      .def(pybind11::init<std::size_t, std::size_t>(), pybind11::arg("begin"),
           pybind11::arg("end"))
      .def_readwrite("begin", &TensorSlice::begin)
      .def_readwrite("end", &TensorSlice::end)
      .def_static("make_tensor2d_slices", &TensorSlice::makeTensor2dSlices)
      .def("to_json_str", [](const TensorSlice& v) { return to_json_str(v); })
      .def_static("from_json_str", [](const std::string& j) {
        return from_json_str<TensorSlice>(j);
      });

  pybind11::class_<VertexIOInfo>(m, "IpuVertexIOInfo")
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string&, VertexIOType, const ShapedArray&,
                          const Base64Data&, const std::vector<TensorSlice>&>(),
           pybind11::arg("name"), pybind11::arg("iotype"),
           pybind11::arg("aval"), pybind11::arg("constant_data") = Base64Data(),
           pybind11::arg("slices2d") = std::vector<TensorSlice>())
      .def(pybind11::init<const std::string&, VertexIOType, const ShapeType&,
                          IpuType, const Base64Data&,
                          const std::vector<TensorSlice>&>(),
           pybind11::arg("name"), pybind11::arg("iotype"),
           pybind11::arg("shape"), pybind11::arg("dtype"),
           pybind11::arg("constant_data") = Base64Data(),
           pybind11::arg("slices2d") = std::vector<TensorSlice>())
      .def(pybind11::init(&VertexIOInfo::makeVertexIOInfo),
           pybind11::arg("name"), pybind11::arg("iotype"),
           pybind11::arg("shape"), pybind11::arg("dtype"),
           pybind11::arg("vertex_dim2") = 0,
           pybind11::arg("constant_data") = Base64Data())
      .def(pybind11::self == pybind11::self)
      .def("to_json_str", [](const VertexIOInfo& v) { return to_json_str(v); })
      .def_static(
          "from_json_str",
          [](const std::string& j) { return from_json_str<VertexIOInfo>(j); })
      .def_readwrite("name", &VertexIOInfo::name)
      .def_readwrite("iotype", &VertexIOInfo::iotype)
      .def_readwrite("aval", &VertexIOInfo::aval)
      .def_readwrite("constant_data", &VertexIOInfo::constant_data)
      .def_readwrite("slices2d", &VertexIOInfo::slices2d)
      .def_property_readonly("shape",
                             [](const VertexIOInfo& v) { return v.aval.shape; })
      .def_property_readonly("dtype",
                             [](const VertexIOInfo& v) { return v.aval.dtype; })
      .def_property_readonly("is_constant_input",
                             &VertexIOInfo::isConstantInput);

  pybind11::class_<TileMapEquation>(m, "IpuTileMapEquation")
      .def(pybind11::init<>())
      .def(pybind11::init<
               const std::string& /* pname */, const std::string& /* vname */,
               const std::vector<TileIndexType>& /* tiles */,
               const std::vector<VertexIOInfo>& /* inputs_info */,
               const std::vector<VertexIOInfo>& /* outputs_info */,
               const std::vector<VertexAttributeI32>& /* attributes_i32 */,
               const std::vector<VertexAttributeF32>& /* attributes_f32 */,
               const std::string& /* tmp_space_name */,
               const ShapedArray& /* tmp_space_aval */,
               const std::string& /* gp_filename */,
               uint64_t /* perf_estimate */, bool /* sync */>(),
           pybind11::arg("pname"), pybind11::arg("vname"),
           pybind11::arg("tiles"),
           pybind11::arg("inputs_info") = std::vector<VertexIOInfo>(),
           pybind11::arg("outputs_info") = std::vector<VertexIOInfo>(),
           pybind11::arg("attributes_i32") = std::vector<VertexAttributeI32>(),
           pybind11::arg("attributes_f32") = std::vector<VertexAttributeF32>(),
           pybind11::arg("tmp_space_name") = "",
           pybind11::arg("tmp_space_aval") =
               ShapedArray{{0}, IpuType::UNSIGNED_CHAR},
           pybind11::arg("gp_filename") = "",
           pybind11::arg("perf_estimate") = 0, pybind11::arg("sync") = false)
      .def("to_json_str",
           [](const TileMapEquation& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileMapEquation>(j);
                  })
      .def_readwrite("pname", &TileMapEquation::pname)
      .def_readwrite("vname", &TileMapEquation::vname)
      .def_readwrite("tiles", &TileMapEquation::tiles)
      .def_readwrite("inputs_info", &TileMapEquation::inputs_info)
      .def_readwrite("outputs_info", &TileMapEquation::outputs_info)
      .def_readwrite("attributes_i32", &TileMapEquation::attributes_i32)
      .def_readwrite("attributes_f32", &TileMapEquation::attributes_f32)
      .def_readwrite("tmp_space_name", &TileMapEquation::tmp_space_name)
      .def_readwrite("tmp_space_aval", &TileMapEquation::tmp_space_aval)
      .def_readwrite("gp_filename", &TileMapEquation::gp_filename)
      .def_readwrite("perf_estimate", &TileMapEquation::perf_estimate)
      .def_readwrite("sync", &TileMapEquation::sync)
      .def_property_readonly("use_tmp_space", &TileMapEquation::useTmpSpace);

  pybind11::class_<TileMapEquationCall>(m, "TileMapEquationCall")
      .def_static("metadata", &TileMapEquationCall::metadata,
                  pybind11::arg("num_inputs"));

  // IPU vertex utils.
  makeIpuDotVertexUtilsBindings(m);
}

// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-fPIC', '-O2', '-Wall', '-mavx2']
cfg['libraries'] = ['poplar', 'poputil', 'popops']
cfg['include_dirs'] = []
cfg['sources'] = [
  'poplin/ConvPartialsStridesPacking.cpp',
  '../external/fastbase64/chromiumbase64.cpp',
  '../external/fastbase64/fastavxbase64.cpp'
]
setup_pybind11(cfg)
%>
*/
