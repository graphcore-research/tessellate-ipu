// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "base_types_py.hpp"
#include "poplin/conv_partial_vertex_utils_py.hpp"
#include "tile_array_ops.hpp"
#include "tile_map_ops_py.hpp"

namespace nb = nanobind;

using namespace ipu;
using namespace nb::literals;

NB_MODULE(pytessellate_ipu_core, m) {
  // Avoid leak warning from the library.
  nanobind::set_leak_warnings(false);
  // Common base types bindings.
  makeIpuTypeBindings(m);
  makeShapeArrayBindings(m);
  makeBase64DataBindings(m);

  // IPU conv partial vertex utils.
  makeIpuDotVertexUtilsBindings(m);

  // Tile array operation parameters.
  nanobind::class_<TileGatherParams>(m, "TileGatherParams")
      .def(nanobind::init<>())
      .def(nanobind::init<const TileArrayType&, const TileArrayType&,
                          const TileArrayType&>(),
           nanobind::arg("previous_tiles"), nanobind::arg("indices"),
           nanobind::arg("tiles"))
      .def("to_json_str",
           [](const TileGatherParams& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileGatherParams>(j);
                  })
      .def_rw("previous_tiles", &TileGatherParams::previous_tiles)
      .def_rw("indices", &TileGatherParams::indices)
      .def_rw("tiles", &TileGatherParams::tiles);

  nanobind::class_<TileDataBarrierParams>(m, "TileDataBarrierParams")
      .def(nanobind::init<>())
      .def(nanobind::init<const std::string&, const std::vector<TileArrayType>&,
                          TileIndexType>(),
           nanobind::arg("vname"), nanobind::arg("inputs_tiles"),
           nanobind::arg("max_tile"))
      .def("to_json_str",
           [](const TileDataBarrierParams& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileDataBarrierParams>(j);
                  })
      .def_rw("vname", &TileDataBarrierParams::vname)
      .def_rw("inputs_tiles", &TileDataBarrierParams::inputs_tiles)
      .def_rw("max_tile", &TileDataBarrierParams::max_tile);

  nanobind::class_<TileConstantParams>(m, "TileConstantParams")
      .def(nanobind::init<>())
      .def(nanobind::init<const ShapedArray&, const TileArrayType&,
                          const Base64Data&>(),
           nanobind::arg("aval"), nanobind::arg("tiles"), nanobind::arg("data"))
      .def("to_json_str",
           [](const TileConstantParams& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileConstantParams>(j);
                  })
      .def_rw("aval", &TileConstantParams::aval)
      .def_rw("tiles", &TileConstantParams::tiles)
      .def_rw("data", &TileConstantParams::data);

  // Tile map ops bindings
  nanobind::enum_<VertexIOType>(m, "IpuVertexIOType", nanobind::is_arithmetic())
      .value("In", VertexIOType::In)
      .value("Out", VertexIOType::Out)
      .value("InOut", VertexIOType::InOut);
  makeVertexAttributeBindings<int32_t>(m, "IpuVertexAttributeI32");
  makeVertexAttributeBindings<float>(m, "IpuVertexAttributeF32");

  nanobind::class_<TensorSlice>(m, "IpuTensorSlice")
      .def(nanobind::init<std::size_t, std::size_t>(), nanobind::arg("begin"),
           nanobind::arg("end"))
      .def_rw("begin", &TensorSlice::begin)
      .def_rw("end", &TensorSlice::end)
      .def_static("make_tensor2d_slices", &TensorSlice::makeTensor2dSlices)
      .def("to_json_str", [](const TensorSlice& v) { return to_json_str(v); })
      .def_static("from_json_str", [](const std::string& j) {
        return from_json_str<TensorSlice>(j);
      });

  nanobind::class_<VertexIOInfo>(m, "IpuVertexIOInfo")
      .def(nanobind::init<>())
      // FIXME: Base64Data(), std::vector<TensorSlice>() default args.
      .def(nanobind::init<const std::string&, VertexIOType, const ShapedArray&,
                          const Base64Data&, const std::vector<TensorSlice>&>(),
           nanobind::arg("name"), nanobind::arg("iotype"),
           nanobind::arg("aval"), nanobind::arg("constant_data"),
           nanobind::arg("slices2d"))
      .def(nanobind::init<const std::string&, VertexIOType, const ShapeType&,
                          IpuType, const Base64Data&,
                          const std::vector<TensorSlice>&>(),
           nanobind::arg("name"), nanobind::arg("iotype"),
           nanobind::arg("shape"), nanobind::arg("dtype"),
           nanobind::arg("constant_data"), nanobind::arg("slices2d"))
      .def(nanobind::init<const std::string&, VertexIOType, const ShapeType&,
                          IpuType, std::size_t, const Base64Data&>(),
           nanobind::arg("name"), nanobind::arg("iotype"),
           nanobind::arg("shape"), nanobind::arg("dtype"),
           nanobind::arg("vertex_dim2") = 0,
           nanobind::arg("constant_data") = Base64Data())
      .def("__eq__", [](const VertexIOInfo& lhs,
                        const VertexIOInfo& rhs) { return lhs == rhs; })
      .def("to_json_str", [](const VertexIOInfo& v) { return to_json_str(v); })
      .def_static(
          "from_json_str",
          [](const std::string& j) { return from_json_str<VertexIOInfo>(j); })
      .def_rw("name", &VertexIOInfo::name)
      .def_rw("iotype", &VertexIOInfo::iotype)
      .def_rw("aval", &VertexIOInfo::aval)
      .def_rw("constant_data", &VertexIOInfo::constant_data)
      .def_rw("slices2d", &VertexIOInfo::slices2d)
      .def_prop_ro("shape", [](const VertexIOInfo& v) { return v.aval.shape; })
      .def_prop_ro("dtype", [](const VertexIOInfo& v) { return v.aval.dtype; })
      .def_prop_ro("is_constant_input", &VertexIOInfo::isConstantInput);

  nanobind::class_<TileMapEquation>(m, "IpuTileMapEquation")
      .def(nanobind::init<>())
      .def(nanobind::init<
               const std::string&,                      // pname
               const std::string&,                      // vname
               const std::vector<TileIndexType>&,       // tiles
               const std::vector<VertexIOInfo>&,        // inputs_info
               const std::vector<VertexIOInfo>&,        // outputs_info
               const std::vector<VertexAttributeI32>&,  // attributes_i32
               const std::vector<VertexAttributeF32>&,  // attributes_f32
               const std::string&,                      // tmp_space_name
               const ShapedArray&,                      // tmp_space_aval
               const std::string&,                      // gp_filename
               uint64_t,                                // perf_estimate
               bool>(),                                 // sync
           nanobind::arg("pname"), nanobind::arg("vname"),
           nanobind::arg("tiles"),
           nanobind::arg("inputs_info") = std::vector<VertexIOInfo>(),
           nanobind::arg("outputs_info") = std::vector<VertexIOInfo>(),
           nanobind::arg("attributes_i32") = std::vector<VertexAttributeI32>(),
           nanobind::arg("attributes_f32") = std::vector<VertexAttributeF32>(),
           nanobind::arg("tmp_space_name") = "",
           nanobind::arg("tmp_space_aval") =
               ShapedArray{{0}, IpuType::UNSIGNED_CHAR},
           nanobind::arg("gp_filename") = "",
           nanobind::arg("perf_estimate") = 0, nanobind::arg("sync") = false)
      .def("to_json_str",
           [](const TileMapEquation& v) { return to_json_str(v); })
      .def_static("from_json_str",
                  [](const std::string& j) {
                    return from_json_str<TileMapEquation>(j);
                  })
      .def_rw("pname", &TileMapEquation::pname)
      .def_rw("vname", &TileMapEquation::vname)
      .def_rw("tiles", &TileMapEquation::tiles)
      .def_rw("inputs_info", &TileMapEquation::inputs_info)
      .def_rw("outputs_info", &TileMapEquation::outputs_info)
      .def_rw("attributes_i32", &TileMapEquation::attributes_i32)
      .def_rw("attributes_f32", &TileMapEquation::attributes_f32)
      .def_rw("tmp_space_name", &TileMapEquation::tmp_space_name)
      .def_rw("tmp_space_aval", &TileMapEquation::tmp_space_aval)
      .def_rw("gp_filename", &TileMapEquation::gp_filename)
      .def_rw("perf_estimate", &TileMapEquation::perf_estimate)
      .def_rw("sync", &TileMapEquation::sync)
      .def_prop_ro("use_tmp_space", &TileMapEquation::useTmpSpace);
}
