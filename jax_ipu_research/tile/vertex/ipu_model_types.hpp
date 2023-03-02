
// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#pragma once

#ifndef __IPU__
#include <cstddef>

// IPU vector typedefs.
using float2 = float[2];
using float4 = float[4];

using char2 = char[2];
using uchar2 = unsigned char[2];
using char4 = char[4];
using uchar4 = unsigned char[4];

using short2 = short[2];
using ushort2 = unsigned short[2];
using short4 = short[4];
using ushort4 = unsigned short[4];

using int2 = int[2];
using uint2 = unsigned int[2];
using int4 = int[4];
using uint4 = unsigned int[4];

using long2 = long[2];
using long4 = long[4];

#endif
