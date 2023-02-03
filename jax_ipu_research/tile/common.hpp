// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <half/half.hpp>
#include <json/json.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

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
 * @brief IPU type traits.
 */
template <IpuType T>
struct IpuTypeTraits {};

#define IPU_TYPE_DECLARE_TRAITS(T1, T2) \
  template <>                           \
  struct IpuTypeTraits<T1> {            \
    using Type = T2;                    \
  };

IPU_TYPE_DECLARE_TRAITS(IpuType::BOOL, bool)
IPU_TYPE_DECLARE_TRAITS(IpuType::UNSIGNED_CHAR, unsigned char)
IPU_TYPE_DECLARE_TRAITS(IpuType::CHAR, char) // char != signed char for Poplar
IPU_TYPE_DECLARE_TRAITS(IpuType::UNSIGNED_SHORT, unsigned short)
IPU_TYPE_DECLARE_TRAITS(IpuType::SHORT, short)
IPU_TYPE_DECLARE_TRAITS(IpuType::UNSIGNED_INT, unsigned int)
IPU_TYPE_DECLARE_TRAITS(IpuType::INT, int)
IPU_TYPE_DECLARE_TRAITS(IpuType::UNSIGNED_LONG, unsigned long)
IPU_TYPE_DECLARE_TRAITS(IpuType::LONG, long)
IPU_TYPE_DECLARE_TRAITS(IpuType::HALF, half_float::half)
IPU_TYPE_DECLARE_TRAITS(IpuType::FLOAT, float)

/**
 * @brief Make an array reference, with a given IPU type.
 * @param raw_array Raw byte array (std::uint8_t).
 * @tparam T IPU dtype.
 * @return Array ref with proper C++ type.
 */
template <IpuType T>
decltype(auto) makeArrayRef(poplar::ArrayRef<char> raw_array) {
  using Type = typename IpuTypeTraits<T>::Type;
  return poplar::ArrayRef(reinterpret_cast<const Type*>(raw_array.data()),
                          std::size_t(raw_array.size() / sizeof(Type)));
}

/**
 * @brief Apply/map a generic function on an array, casting to a proper dtype
 * first.
 *
 * @param fn Generic functor to apply.
 * @param raw_array Raw byte data array.
 * @param type IPU dtype.
 * @return Returned value of the function.
 */
template <typename Fn>
decltype(auto) applyFnToArray(Fn&& fn, poplar::ArrayRef<char> raw_array,
                              IpuType type) {
  switch (type) {
    case IpuType::BOOL:
      return fn(makeArrayRef<IpuType::BOOL>(raw_array));
    case IpuType::CHAR:
      return fn(makeArrayRef<IpuType::CHAR>(raw_array));
    case IpuType::UNSIGNED_CHAR:
      return fn(makeArrayRef<IpuType::UNSIGNED_CHAR>(raw_array));
    case IpuType::SHORT:
      return fn(makeArrayRef<IpuType::SHORT>(raw_array));
    case IpuType::UNSIGNED_SHORT:
      return fn(makeArrayRef<IpuType::UNSIGNED_SHORT>(raw_array));
    case IpuType::INT:
      return fn(makeArrayRef<IpuType::INT>(raw_array));
    case IpuType::UNSIGNED_INT:
      return fn(makeArrayRef<IpuType::UNSIGNED_INT>(raw_array));
    case IpuType::LONG:
      return fn(makeArrayRef<IpuType::LONG>(raw_array));
    case IpuType::UNSIGNED_LONG:
      return fn(makeArrayRef<IpuType::UNSIGNED_LONG>(raw_array));
    case IpuType::QUARTER:
      // TODO?
      throw std::runtime_error("Unsupported Quarter IPU datatype.");
    case IpuType::HALF:
      return fn(makeArrayRef<IpuType::HALF>(raw_array));
    case IpuType::FLOAT:
      return fn(makeArrayRef<IpuType::FLOAT>(raw_array));
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
