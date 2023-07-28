// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#include <nanobind/nanobind.h>

#include "tile_map_ops.hpp"

namespace nb = nanobind;

namespace ipu {

/**
 * @brief Make Python bindings of VertexAttribute class.
 */
template <typename T>
decltype(auto) makeVertexAttributeBindings(nanobind::module_& m,
                                           const char* name) {
  using VertexAttrType = VertexAttribute<T>;
  nanobind::class_<VertexAttrType>(m, name)
      .def(nanobind::init<>())
      .def(nanobind::init<const std::string&, T>(), nanobind::arg("name"),
           nanobind::arg("value"))
      .def("__eq__", [](const VertexAttrType& lhs,
                        const VertexAttrType& rhs) { return lhs == rhs; })
      .def("to_json_str",
           [](const VertexAttrType& v) { return to_json_str(v); })
      .def_static(
          "from_json_str",
          [](const std::string& j) { return from_json_str<VertexAttrType>(j); })
      .def_rw("name", &VertexAttrType::name)
      .def_rw("value", &VertexAttrType::value);
}
}  // namespace ipu
