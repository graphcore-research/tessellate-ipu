// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#pragma once
#include <type_traits>

#include "intrinsics_utils.hpp"
#include "ipu_model_types.hpp"

namespace ipu {

/**
 * @brief Thin abstraction of the IPU AMP unit(s) and registers, allowing
 * to write generic code compiling on IPU model and IPU hardware.
 *
 * NOTE: zero-cost abstraction on IPU hardware.
 *
 * The AMP class is modelling AACC registers as well as AMP unit instructions
 * on the IPU model, reproducing the expected behaviour of the hardware.
 */
template <typename T>
class AMP {
 public:
  // TODO: support half as well.
  static_assert(std::is_same_v<T, float>);
  using FPType = T;
  /** Number of AACC register available in hw. */
  // TODO: use TFPU_AMP_UNITS_PER_SET and TFPU_AACC_PER_AMP_UNIT;
  static constexpr unsigned NumAACC = 16;

  // TODO: random initialization on IPU model of registers.
  AMP() noexcept = default;
  // No copy + no move allowed!
  AMP(const AMP&) = delete;
  AMP(AMP&&) = delete;

  /**
   * @brief Set the value of the TAS register, used in
   * `axpy` operation.
   */
  ALWAYS_INLINE void tas(FPType val) noexcept {
#ifdef __IPU__
    __builtin_ipu_put_tas(val);
#else
    m_tas = val;
#endif
  }
  /**
   * @brief Zero AACC registers.
   */
  ALWAYS_INLINE void aaccZero() noexcept {
#ifdef __IPU__
    __builtin_ipu_aacc_zero();
#else
    for (unsigned idx = 0; idx < NumAACC; ++idx) {
      m_aacc[idx] = 0;
    }
#endif
  }

  /**
   * @brief Scaled-add `axpy` intrinsic. Only supported on FP32.
   * NOTE: act as 1 stage pipeline, storing result in AACC[0...2]
   */
  ALWAYS_INLINE float2 axpy(float2 x, float2 y) noexcept {
    using T2 = float2;
#ifdef __IPU__
    // Weird ordering here? Bug in the intrinsic definition?
    return __builtin_ipu_f32v2axpy(y, x);
#else
    // Simulating pipeline with storing in AACC[0] and AACC[2].
    const auto res = T2{m_aacc[0], m_aacc[2]};
    // FIXME/TODO: understand ordering!?
    m_aacc[0] = m_tas * x[0] + y[0];
    m_aacc[2] = m_tas * x[1] + y[1];
    return res;
#endif
  }

  /**
   * @brief Outer-product `aop` intrinsic. Only supported on FP32.
   * Storing results in AACC[0...6]
   */
  void aop(float2 x, float2 y) noexcept {
#ifdef __IPU__
    // Note: third argument not used by hw.
    __builtin_ipu_f32v2aop(x, y, 0);
#else
    // Multiply + accumulate.
    m_aacc[0] += x[0] * y[0];
    m_aacc[2] += x[1] * y[0];
    m_aacc[4] += x[0] * y[1];
    m_aacc[6] += x[1] * y[1];
#endif
  }

  /**
   * @brief `gina` instruction: get AACC register + propagate.
   * FIXME: support non-zero flag/index.
   */
  template <unsigned int FLAG>
  float2 gina(float2 val) noexcept {
    using T2 = float2;
#ifdef __IPU__
    return __builtin_ipu_f32v2gina(val, 0);
#else
    // TODO: implement GINA_IMMFLAGS__SET__GET
    const auto res = T2{m_aacc[0], m_aacc[2]};
    // Propagate accumulator states.
    for (unsigned idx = 4; idx < NumAACC; idx += 4) {
      m_aacc[idx - 4] = m_aacc[idx];
      m_aacc[idx - 2] = m_aacc[idx + 2];
    }
    m_aacc[NumAACC - 4] = val[0];
    m_aacc[NumAACC - 2] = val[1];
    return res;
#endif
  }

 private:
#ifndef __IPU__
  // Simulating AACC registers on IPU model.
  FPType m_aacc[NumAACC];
  // Simulating TAS register on IPU model.
  FPType m_tas;
#endif
};

}  // namespace ipu
