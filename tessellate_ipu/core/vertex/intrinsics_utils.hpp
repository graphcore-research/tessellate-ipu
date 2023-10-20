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

#include "ipu_model_types.hpp"
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
 * Tag dispatching, between IPU model and IPU hardware implementations.
 *
 * Making it hopefully easier to maintain IPU hardware and model
 * implementations, without #ifdef/#endif preprocessor spaghetti code.
 */
namespace ipu {
/** IPU hardware tag. */
struct HardwareTag {
  static constexpr bool hardware = true;
};
/** IPU model tag. */
struct ModelTag {
    static constexpr bool model = true;
};
}  // namespace ipu

// IPU dispatch tag preprocessor.
#ifdef __IPU__
#define IPU_TAG_TYPE ipu::HardwareTag
#define IPU_DISPATCH_TAG (ipu::HardwareTag{})
#else
#define IPU_TAG_TYPE ipu::ModelTag
#define IPU_DISPATCH_TAG (ipu::ModelTag{})
#endif

#ifdef __IPU__
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
  // TODO: use `__builtin_ipu_uput`?
  asm volatile(
      R"l( uput $TAS, %[sv]
        )l"
      :
      : [ sv ] "r"(v)
      :);
}

/**
 * @brief Zero AACC registers.
 */
ALWAYS_INLINE void __builtin_ipu_aacc_zero() {
  asm (R"(
    setzi $a0, 0x8
    uput $FP_CLR, $a0
  )"
      :
      :
      : "$a0");
}


/**
 * @brief IPU cmac f32 instruction.
 */
ALWAYS_INLINE void __builtin_ipu_f32v2cmac(float2 x, float2 y) noexcept {
  asm volatile(
      R"l( f32v2mac %[x], %[y]
        )l"
      :
      : [ x ] "r"(x), [ y ] "r"(y)
      :);
}

template <typename T>
ALWAYS_INLINE float ld32(const T* address, unsigned offset) {
  float result;
  // TODO - Use intrinsic/builtin for this when one becomes available
  asm volatile(
      R"l(  ld32 %[result], %[address], %[offset]
      )l"
      : [ result ] "=r"(result)
      : [ address ] "r"(address), [ offset ] "r"(offset)
      :);
  return result;
}

#else

#include <limits>

namespace ipu {
// Implementations of IPU intrinsics for IPUModel

// https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/namespaceipu.html#aa1a33d2be82a6b73549badf896cfd88e
template <class T>
void store_postinc(T** a, T const& v, int i) {
  **a = v;
  (*a) += i;
}

// https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/namespaceipu.html#acb144a365e4027998954ee1e9d98e0d3
template <class T>
T load_postinc(T const** a, int i) {
  T const* p = *a;
  (*a) += i;
  return *p;
}

// https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/namespaceipu.html#a2a81ec4b6956ea14fe230a137178ff48
template <class T, size_t N>
IpuVector<T, N> fma(IpuVector<T, N> const& x, IpuVector<T, N> const& y,
                    IpuVector<T, N> const& z) {
  IpuVector<T, N> ret = z;
  for (size_t i = 0; i < N; ++i) ret[i] += x[i] * y[i];
  return ret;
}

}  // namespace ipu

// And give useful error messages when people port from IPU to IPUModel.
template <typename T>
constexpr bool __ipu_false() {
  return !std::is_same<T, T>::value;
}

template <typename T>
void __builtin_ipu_put_tas(T v) {
  static_assert(__ipu_false<T>(), "*** Please use `ipu::AMP` class for TAS handling on IPUModel.");
}

template <typename T>
T __builtin_ipu_f32v2axpy(T const& x, T const& y) {
  static_assert(__ipu_false<T>(), "*** Please use `ipu::AMP::axpy` for `f32v2axpy` intrinsic on IPUModel.");
  return T{};
}
// clang-format on

#endif

/**
 * @brief Bitwise cast to a different type.
 */
template <class R, class T>
R as(T x) {
  return *reinterpret_cast<R*>(&x);
}
