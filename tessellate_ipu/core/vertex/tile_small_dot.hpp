// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "intrinsics_utils.hpp"

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

inline void axplusby_f32_v1(float a, float b, const float2 *x, const float2 *y,
                            float2 *z, rptsize_t nblocks) {
  // Necessary if using unsigned `nblocks`.
  // __builtin_assume(nblocks < 4096);
  using T2 = float2;
  const T2 av = {a, a};
  // Using TAS register for one of the scalar.
  __ipu_and_ipumodel_tas tas;
  tas.put(b);

  T2 res, xv, yv, zv, tmp;

  xv = ipu::load_postinc(&x, 1);
  yv = ipu::load_postinc(&y, 1);
  res = xv * av;
  for (unsigned idx = 0; idx != nblocks; ++idx) {
    // Pseudo dual-issuing of instructions.
    // popc should be able to generate an optimal rpt loop.
    {
      xv = ipu::load_postinc(&x, 1);
      // TODO: fix ordering of arguments in `f32v2axpy`.
      tmp = tas.f32v2axpy(res, yv);
    }
    {
      yv = ipu::load_postinc(&y, 1);
      // TODO: fix ordering of arguments in `f32v2axpy`.
      zv = tas.f32v2axpy(tmp, tmp);
    }
    {
      ipu::store_postinc(&z, zv, 1);
      res = xv * av;
    }
  }
}

/**
 * @brief Apply 2d rotation transform (float).
 *
 * Note: input rows are separated, allowing more flexibility
 * for functions/vertices using this base compute method.
 */
inline void rotation2d_f32(float2 cs, const float2 *inrow0,
                           const float2 *inrow1, float2 *outrow0,
                           float2 *outrow1, rptsize_t nblocks, ipu::ModelTag) {
  axplusby_f32_v1(cs[0], -cs[1], inrow0, inrow1, outrow0, nblocks);
  axplusby_f32_v1(cs[1], cs[0], inrow0, inrow1, outrow1, nblocks);
}

inline void rotation2d_f32(float2 cs, const float2 *inrow0,
                           const float2 *inrow1, float2 *outrow0,
                           float2 *outrow1, rptsize_t nblocks,
                           ipu::HardwareTag) {
  // Using same implementation as IPU model for now.
  rotation2d_f32(cs, inrow0, inrow1, outrow0, outrow1, nblocks,
                 ipu::ModelTag{});
}
