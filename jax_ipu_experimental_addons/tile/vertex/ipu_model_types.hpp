// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#pragma once

#ifndef __IPU__
#include <array>
#include <cstddef>

// IPU vector typedefs.
using float2 = std::array<float, 2>;
using float4 = std::array<float, 4>;

using char2 = std::array<char, 2>;
using uchar2 = std::array<unsigned char, 2>;
using char4 = std::array<char, 4>;
using uchar4 = std::array<unsigned char, 4>;

using short2 = std::array<short, 2>;
using ushort2 = std::array<unsigned short, 2>;
using short4 = std::array<short, 4>;
using ushort4 = std::array<unsigned short, 4>;

using int2 = std::array<int, 2>;
using uint2 = std::array<unsigned int, 2>;
using int4 = std::array<int, 4>;
using uint4 = std::array<unsigned int, 4>;

using long2 = std::array<long, 2>;
using long4 = std::array<long, 4>;

#endif
