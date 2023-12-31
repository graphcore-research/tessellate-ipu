// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "intrinsics_utils.hpp"
#include "ipu_amp.hpp"

/**
 * @brief z = a*x + b*y float32 implementation.
 *
 * where x, y, z are 1D arrays and a, b are scalars.
 * Implementation compatible with IPU model and hardware.
 *
 * Requires input arrays with size % 2 == 0
 */
inline void axplusby_f32_v0(float a, float b, const float2 *x, const float2 *y,
                            float2 *z, rptsize_t nblocks) {
  using T2 = float2;
  const T2 av = {a, a};
  const T2 bv = {b, b};
  // Sub-optimal vectorized implementation.
  for (unsigned idx = 0; idx < nblocks; ++idx) {
    const T2 xv = ipu::load_postinc(&x, 1);
    const T2 yv = ipu::load_postinc(&y, 1);
    const T2 zv = av * xv + bv * yv;
    ipu::store_postinc(&z, zv, 1);
  }
}
/**
 * @brief z = a*x + b*y float32 implementation using `rpt` loop and `f32v2axpy`
 *
 * Compatible with IPU hardware and IPU model.
 * 30% slower than inline assembly implementation.
 */
template <class IpuTag, std::enable_if_t<IpuTag::model, bool> = true>
inline void axplusby_f32(float a, float b, const float2 *x, const float2 *y,
                         float2 *z, rptsize_t nblocks) {
  // Necessary if using unsigned `nblocks`.
  // __builtin_assume(nblocks < 4096);
  using T2 = float2;
  const T2 av = {a, a};
  // Basic AMP usage with TAS + axpy instruction.
  ipu::AMP<float> amp;
  amp.tas(b);

  T2 res, xv, yv, zv, tmp;

  xv = ipu::load_postinc(&x, 1);
  yv = ipu::load_postinc(&y, 1);
  res = xv * av;
  for (unsigned idx = 0; idx != nblocks; ++idx) {
    // Pseudo dual-issuing of instructions.
    // popc should be able to generate an optimal rpt loop.
    {
      xv = ipu::load_postinc(&x, 1);
      tmp = amp.axpy(yv, res);
    }
    {
      yv = ipu::load_postinc(&y, 1);
      zv = amp.axpy(tmp, tmp);
    }
    {
      ipu::store_postinc(&z, zv, 1);
      res = xv * av;
    }
  }
}
/**
 * @brief z = a*x + b*y float32 implementation fully optimized in inline
 * assembly.
 */
template <class IpuTag, std::enable_if_t<IpuTag::hardware, bool> = true>
inline void axplusby_f32(float a, float b, const float2 *x, const float2 *y,
                         float2 *z, rptsize_t nblocks) {
  // Necessary if using unsigned `nblocks`.
  // __builtin_assume(nblocks < 4096);
  using T2 = float2;
  // Basic AMP usage with TAS + axpy instruction.
  ipu::AMP<float> amp;
  amp.tas(b);


  T2 av = {a, a};
  // Explicit variables passed to inline assembly.
  // Easier to read + compiling on IPU model.
  T2 xv, yv, zv;
  uint2 tapaddr;
  // Inline assembly loop in order to use `ldst64pace` instruction.
  // Note: requires "unrolling" the beginning of the `f32v2axpy` pipeline.
  // TODO: investigate issue with inputs register re-use.
  asm volatile(
      R"(
        ld64step %[xv], $m15, %[xptr]+=, 1
        ld64step %[yv], $m15, %[yptr]+=, 1
        {
            ld64step %[xv], $m15, %[xptr]+=, 1
            f32v2mul %[zv], %[xv], %[av]
        }
        {
            ld64step %[yv], $m15, %[yptr]+=, 1
            f32v2axpy %[zv], %[yv], %[zv]
        }
        {
            ld64step %[xv], $m15, %[xptr]+=, 1
            f32v2mul %[zv], %[xv], %[av]
        }
        {
            ld64step %[yv], $m15, %[yptr]+=, 1
            f32v2axpy %[zv], %[yv], %[zv]
        }
        tapack %[tapaddr], %[xptr], $mzero, %[zptr]
        .align 8
        {
            rpt %[nb], 1
            fnop
        }
        {
            ldst64pace %[xv], %[zv], %[tapaddr]+=, $mzero, 0
            f32v2mul %[zv], %[xv], %[av]
        }
        {
            ld64step %[yv], $m15, %[yptr]+=, 1
            f32v2axpy %[zv], %[yv], %[zv]
        }
      )"
      : [ xptr ] "+r"(x), [ yptr ] "+r"(y), [ av ] "+r"(av), [ xv ] "=r"(xv),
        [ yv ] "=r"(yv), [ zv ] "=r"(zv), [ tapaddr ] "+r"(tapaddr),
        [ nb ] "+r"(nblocks)
      : [ zptr ] "r"(z)
      :);
  // Note: explicit list of used registers not compiling on IPU model.
  // : "$a0:1", "$a2:3", "$a4:5", "$m4", "$m5"
}

/**
 * @brief Apply 2d rotation transform (float).
 *
 * Note: input rows are separated, allowing more flexibility
 * for functions/vertices using this base compute method.
 */
template <class IpuTag>
inline void rotation2d_f32(float2 cs, const float2 *inrow0,
                           const float2 *inrow1, float2 *outrow0,
                           float2 *outrow1, rptsize_t nblocks) {
  // axplusby is using one AMP unit. TODO: investigate using more!
  axplusby_f32<IpuTag>(cs[0], -cs[1], inrow0, inrow1, outrow0, nblocks);
  // NOTE: inrow1+0, outrow1 arguments order necessary due to bank constraints!
  axplusby_f32<IpuTag>(cs[0], cs[1], inrow1, inrow0, outrow1, nblocks);
}
