// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "base_types.hpp"

namespace nb = nanobind;

namespace ipu {
/**
 * @brief Make nanobind bindings of IpuType enum.
 */
inline void makeIpuTypeBindings(nanobind::module_& m) {
  nanobind::enum_<IpuType>(m, "IpuType", nanobind::is_arithmetic())
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
      .def_prop_ro("bytesize", [](IpuType t) { return ipuTypeSize(t); })
      // TODO: remove once integrated in Nanobind.
      .def_prop_ro("name",
                   [](nb::object obj) { return nb::getattr(obj, "__name__"); });
}

/**
 * @brief Make nanobind bindings of ShapeArray class.
 */
inline void makeShapeArrayBindings(nanobind::module_& m) {
  nanobind::class_<ShapedArray>(m, "IpuShapedArray")
      .def(nanobind::init<>())
      .def(nanobind::init<const ShapeType&, IpuType>(), nanobind::arg("shape"),
           nanobind::arg("dtype"))
      .def("to_json_str", [](const ShapedArray& v) { return to_json_str(v); })
      .def_static(
          "from_json_str",
          [](const std::string& j) { return from_json_str<ShapedArray>(j); })
      .def_rw("shape", &ShapedArray::shape)
      .def_rw("dtype", &ShapedArray::dtype)
      .def_prop_ro("size", &ShapedArray::size);
}

/**
 * @brief Make nanobind bindings of Base64Data class.
 */
inline void makeBase64DataBindings(nanobind::module_& m) {
  nanobind::class_<Base64Data>(m, "Base64Data")
      .def(nanobind::init<>())
      .def(nanobind::init<const std::string&>(), nanobind::arg("encoded_data"))
      .def("__init__",
           [](Base64Data* t, nb::bytes data) {
             new (t) Base64Data(data.c_str(), data.size());
           })
      .def_rw("encoded_data", &Base64Data::encoded_data)
      .def_static(
          "from_encoded_data",
          [](nb::bytes data) { return Base64Data(data.c_str(), data.size()); })
      .def_static("from_decoded_data", &Base64Data::fromDecodedData)
      .def_prop_ro("decoded_data", &Base64Data::decode)
      .def_prop_ro("is_empty", &Base64Data::empty)
      .def("to_json_str", [](const Base64Data& v) { return to_json_str(v); })
      .def_static("from_json_str", [](const std::string& j) {
        return from_json_str<Base64Data>(j);
      });
}

}  // namespace ipu
