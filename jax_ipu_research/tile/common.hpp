// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <cstdint>
#include <json/json.hpp>

#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ipu {
using json = nlohmann::json;

// Standard tile index used.
using TileIndexType = int32_t;
// Shape type. TODO: use compact one from poplar?
using ShapeType = std::vector<std::size_t>;

/**
 * @brief IPU (Poplar) types supported by hardware.
 */
enum class IpuType : int8_t {
  BOOL = 1,
  CHAR,
  UNSIGNED_CHAR,
  SHORT,
  UNSIGNED_SHORT,
  INT,
  UNSIGNED_INT,
  LONG,
  UNSIGNED_LONG,
  QUARTER,
  HALF,
  FLOAT
};

/**
 * @brief Convert IPU type enum to Poplar type.
 */
poplar::Type toPoplar(IpuType type) {
  switch (type) {
    case IpuType::BOOL:
      return poplar::BOOL;
    case IpuType::CHAR:
      return poplar::CHAR;
    case IpuType::UNSIGNED_CHAR:
      return poplar::UNSIGNED_CHAR;
    case IpuType::SHORT:
      return poplar::SHORT;
    case IpuType::UNSIGNED_SHORT:
      return poplar::UNSIGNED_SHORT;
    case IpuType::INT:
      return poplar::INT;
    case IpuType::UNSIGNED_INT:
      return poplar::UNSIGNED_INT;
    case IpuType::LONG:
      return poplar::LONG;
    case IpuType::UNSIGNED_LONG:
      return poplar::UNSIGNED_LONG;
    case IpuType::QUARTER:
      return poplar::QUARTER;
    case IpuType::HALF:
      return poplar::HALF;
    case IpuType::FLOAT:
      return poplar::FLOAT;
  }
  throw std::runtime_error("Unknown IPU datatype.");
}

/**
 * @brief Make pybind11 bindings of IpuType enum.
 */
inline decltype(auto) makeIpuTypeBindings(pybind11::module& m) {
  return pybind11::enum_<IpuType>(m, "IpuType", pybind11::arithmetic())
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
      .value("FLOAT", IpuType::FLOAT);
}

/**
 * @brief Convert an object to JSON string.
 */
template <typename T>
std::string to_json_str(const T& v) {
  json j = v;
  return j.dump();
}
/**
 * @brief Build an object from a JSON string.
 */
template <typename T>
T from_json_str(const std::string& json_str) {
  json j = json::parse(json_str);
  return j.get<T>();
}

}  // namespace ipu
