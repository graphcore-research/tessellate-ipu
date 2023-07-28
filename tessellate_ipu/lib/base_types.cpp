// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "base_types.hpp"

namespace ipu {

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

}  // namespace ipu
