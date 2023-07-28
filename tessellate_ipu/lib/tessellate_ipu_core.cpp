// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "base_types_py.hpp"
#include "poplin/conv_partial_vertex_utils_py.hpp"
#include "tile_array_ops.hpp"
#include "tile_map_ops_py.hpp"

using namespace ipu;

PYBIND11_MODULE(pytessellate_ipu_core, m) {
  // Common base types bindings.
  makeIpuTypeBindings(m);
  makeShapeArrayBindings(m);
  makeBase64DataBindings(m);

  // Tile array operation parameters.
  pybind11::class_<TileGatherParams>(m, "TileGatherParams")
      .def(pybind11::init<>())
      .def(pybind11::init<const TileArrayType&, const TileArrayType&,
                          const TileArrayType&>(),
           pybind11::arg("previous_tiles"), pybind11::arg("indices"),
           pybind11::arg("tiles"))
      .def("to_json_str",
           [](const TileGatherParams& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileGatherParams>(j);
                  })
      .def_readwrite("previous_tiles", &TileGatherParams::previous_tiles)
      .def_readwrite("indices", &TileGatherParams::indices)
      .def_readwrite("tiles", &TileGatherParams::tiles);

  pybind11::class_<TileDataBarrierParams>(m, "TileDataBarrierParams")
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string&, const std::vector<TileArrayType>&,
                          TileIndexType>(),
           pybind11::arg("vname"), pybind11::arg("inputs_tiles"),
           pybind11::arg("max_tile"))
      .def("to_json_str",
           [](const TileDataBarrierParams& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileDataBarrierParams>(j);
                  })
      .def_readwrite("vname", &TileDataBarrierParams::vname)
      .def_readwrite("inputs_tiles", &TileDataBarrierParams::inputs_tiles)
      .def_readwrite("max_tile", &TileDataBarrierParams::max_tile);

  pybind11::class_<TileConstantParams>(m, "TileConstantParams")
      .def(pybind11::init<>())
      .def(pybind11::init<const ShapedArray&, const TileArrayType&,
                          const Base64Data&>(),
           pybind11::arg("aval"), pybind11::arg("tiles"), pybind11::arg("data"))
      .def("to_json_str",
           [](const TileConstantParams& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileConstantParams>(j);
                  })
      .def_readwrite("aval", &TileConstantParams::aval)
      .def_readwrite("tiles", &TileConstantParams::tiles)
      .def_readwrite("data", &TileConstantParams::data);

  // IPU vertex utils.
  makeIpuDotVertexUtilsBindings(m);

  // Tile map ops bindings
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
}
