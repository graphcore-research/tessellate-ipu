// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#ifdef __IPU__
// Use the IPU intrinsics
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#define NAMESPACE ipu
#else
// Use the std functions
#include <cmath>
#define NAMESPACE std
#endif

#ifdef __IPU__
#define SUPERVISOR_TARGET __attribute__((target("supervisor")))
#else
#define SUPERVISOR_TARGET
#endif

/**
 * @brief Efficient division by 6, on IPU hardware. Up to 98,304.
 */
template <typename T>
inline T ipu_div_by_6(T n) noexcept {
  return (n * 0xaaab) >> 18;
}

/**
 * @brief IPU intrinsics, for setting up the $TAS register.
 */
inline void __builtin_ipu_put_tas(float v) noexcept {
  // TAS register, used for __builtin_ipu_f32v2axpy.
  asm volatile(
      R"l( uput $TAS, %[sv]
        )l"
      :
      : [sv] "r"(v)
      :);
}
