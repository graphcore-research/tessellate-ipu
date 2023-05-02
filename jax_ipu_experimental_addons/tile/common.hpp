// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <fastbase64/fastavxbase64.h>
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
 * @brief Get the size (in bytes) of IPU datatype.
 */
std::size_t ipuTypeSize(IpuType type) {
  switch (type) {
    case IpuType::BOOL:
      return 1;
    case IpuType::CHAR:
      return 1;
    case IpuType::UNSIGNED_CHAR:
      return 1;
    case IpuType::SHORT:
      return 2;
    case IpuType::UNSIGNED_SHORT:
      return 2;
    case IpuType::INT:
      return 4;
    case IpuType::UNSIGNED_INT:
      return 4;
    case IpuType::LONG:
      return 4;
    case IpuType::UNSIGNED_LONG:
      return 4;
    case IpuType::QUARTER:
      return 1;
    case IpuType::HALF:
      return 2;
    case IpuType::FLOAT:
      return 4;
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
IPU_TYPE_DECLARE_TRAITS(IpuType::CHAR, char)  // char != signed char for Poplar
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

/**
 * @brief JAX-like shaped array data structure.
 */
struct ShapedArray {
  /** Shape of the array. */
  ShapeType shape;
  /** Dtype of the array. */
  IpuType dtype = IpuType::UNSIGNED_CHAR;

  /** @brief Size of the array (i.e. num elements). */
  std::size_t size() const noexcept {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<std::size_t>());
  }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ShapedArray, shape, dtype)

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
 * @brief Data encoded in base64.
 */
struct Base64Data {
  using byte = std::uint8_t;
  /** Raw data as base64 encoded. */
  std::string encoded_data;

  /** Is the data empty? */
  bool empty() const noexcept { return encoded_data.empty(); }
  /**
   * @brief Create base64 encoded data from raw data.
   */
  static Base64Data fromDecodedData(const std::string& data) {
    return Base64Data{chromium_base64_encode(data)};
  }
  /**
   * @brief Decode the data.
   */
  std::string decode() const { return chromium_base64_decode(encoded_data); }
};
// JSON encoding/decoding, supporting empty fields.
void to_json(json& j, const Base64Data& v) {
  if (!v.empty()) {
    j = json{{"encoded_data", v.encoded_data}};
  }
}
void from_json(const json& j, Base64Data& v) {
  const auto it = j.find("encoded_data");
  if (it != j.end()) {
    it->get_to(v.encoded_data);
  }
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
