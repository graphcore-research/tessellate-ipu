// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base_types.hpp"

namespace ipu {
/**
 * @brief Make pybind11 bindings of IpuType enum.
 */
inline void makeIpuTypeBindings(pybind11::module& m) {
  pybind11::enum_<IpuType>(m, "IpuType", pybind11::arithmetic())
      .value("BOOL", IpuType::BOOL)
      .value("CHAR", IpuType::CHAR)
      .value("UNSIGNED_CHAR", IpuType::UNSIGNED_CHAR)
      .value("SHORT", IpuType::SHORT)
      .value("UNSIGNED_SHORT", IpuType::UNSIGNED_SHORT)
      .value("INT", IpuType::INT)
      .value("UNSIGNED_INT", IpuType::UNSIGNED_INT)
      .value("LONG", IpuType::LONG)
      .value("UNSIGNED_LONG", IpuType::UNSIGNED_LONG)
      .value("QUARTER", IpuType::QUARTER)
      .value("HALF", IpuType::HALF)
      .value("FLOAT", IpuType::FLOAT)
      .def_property_readonly("bytesize",
                             [](IpuType t) { return ipuTypeSize(t); });
}

/**
 * @brief Make pybind11 bindings of ShapeArray class.
 */
inline void makeShapeArrayBindings(pybind11::module& m) {
  pybind11::class_<ShapedArray>(m, "IpuShapedArray")
      .def(pybind11::init<>())
      .def(pybind11::init<const ShapeType&, IpuType>(), pybind11::arg("shape"),
           pybind11::arg("dtype"))
      .def("to_json_str", [](const ShapedArray& v) { return to_json_str(v); })
      .def_static(
          "from_json_str",
          [](const std::string& j) { return from_json_str<ShapedArray>(j); })
      .def_readwrite("shape", &ShapedArray::shape)
      .def_readwrite("dtype", &ShapedArray::dtype)
      .def_property_readonly("size", &ShapedArray::size);
}

/**
 * @brief Make pybind11 bindings of ShapeArray class.
 */
inline void makeBase64DataBindings(pybind11::module& m) {
  pybind11::class_<Base64Data>(m, "Base64Data")
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string&>(), pybind11::arg("encoded_data"))
      .def_readwrite("encoded_data", &Base64Data::encoded_data)
      .def_static("from_decoded_data", &Base64Data::fromDecodedData)
      .def_property_readonly("decoded_data", &Base64Data::decode)
      .def_property_readonly("is_empty", &Base64Data::empty)
      .def("to_json_str", [](const Base64Data& v) { return to_json_str(v); })
      .def_static("from_json_str", [](const std::string& j) {
        return from_json_str<Base64Data>(j);
      });
}

}  // namespace ipu
