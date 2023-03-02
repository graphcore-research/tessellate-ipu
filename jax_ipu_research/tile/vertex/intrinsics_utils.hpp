// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#ifdef __IPU__
// Use the IPU intrinsics
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#define NAMESPACE ipu
#else
#include "ipu_model_types.hpp"
// Use the std functions
#include <cmath>
#define NAMESPACE std
#endif

#ifdef __IPU__
#define SUPERVISOR_TARGET __attribute__((target("supervisor")))
#else
#define SUPERVISOR_TARGET
#endif

// #define ALWAYS_INLINE __attribute__((always_inline))
#define ALWAYS_INLINE inline

/**
 * @brief Efficient division by 6, on IPU hardware. Up to 98,304.
 */
template <typename T>
ALWAYS_INLINE T ipu_div_by_6(T n) noexcept {
  return (n * 0xaaab) >> 18;
}

/**
 * @brief IPU intrinsics, for setting up the $TAS register.
 */
ALWAYS_INLINE void __builtin_ipu_put_tas(float v) noexcept {
  // TAS register, used for __builtin_ipu_f32v2axpy.
  asm volatile(
      R"l( uput $TAS, %[sv]
        )l"
      :
      : [sv] "r"(v)
      :);
}

/**
 * @brief IPU cmac f32 instruction.
 */
ALWAYS_INLINE void __builtin_ipu_f32v2cmac(float2 x, float2 y) noexcept {
  asm volatile(
      R"l( f32v2mac %[x], %[y]
        )l"
      :
      : [x] "r"(x), [y] "r"(y)
      :);
}

template <typename T>
ALWAYS_INLINE float ld32(const T* address, unsigned offset) {
  float result;
  // TODO - Use intrinsic/builtin for this when one becomes available
  asm volatile(
      R"l(  ld32 %[result], %[address], %[offset]
      )l"
      : [result] "=r"(result)
      : [address] "r"(address), [offset] "r"(offset)
      :);
  return result;
}
