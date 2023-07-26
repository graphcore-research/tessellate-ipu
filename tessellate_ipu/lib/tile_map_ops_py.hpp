// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tile_map_ops.hpp"

namespace ipu {

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
}  // namespace ipu
