// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "intrinsics_utils.hpp"

using namespace poplar;

/**
 * @brief Compute the Schur decomposition of a symmetric 2x2 matrix.
 */
__attribute__((always_inline)) float2 sym_schur2(const float* pcol, const float* qcol,
                                const unsigned p, const unsigned q) noexcept {
  using T = float;
  using T2 = float2;

  const T Apq = pcol[q];
  const T App = pcol[p];
  const T Aqq = qcol[q];

  // See Algorithm 8.4.1, MATRIX computations.
  // Avoid the division `tau` by keeping the two independent factors.
  const T Cpq = 2 * Apq;
  const T Dpq = Aqq - App;
  const T sq_Cpq = Cpq * Cpq;
  const T sq_Dpq = Dpq * Dpq;

  // Avoids dividing by zero/eps!
  constexpr T eps = 1e-10f;
  T2 cs_vec{1, 0};
  if (sq_Cpq > sq_Dpq * eps) {
    const T norm_pq = sqrt(sq_Cpq + sq_Dpq);
    T t;
    if (Dpq >= 0) {
      t = Cpq / (Dpq + norm_pq);

    } else {
      t = Cpq / (Dpq - norm_pq);
    }
    const T sq_t_p1 = 1 + t * t;
    const T c = 1 / sqrt(sq_t_p1);
    return T2{c, t * c};
  }
  return cs_vec;
}

/**
 * @brief Jacobi algorithm, schur decomposition on 2x2 symmetric function.
 *
 * See:  Gene H. Golub, Charles F. Van Loan, MATRIX COMPUTATIONS, 3rd edition,
 * Johns Hopkins Chapter 8.
 *
 * This vertex should take ~250 cycles of pure compute + memory.
 */
class JacobiSymSchur2 : public Vertex {
 public:
  using T = float;
  using T2 = float2;

  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> pq;  // (2,) p and q indexes
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> pcol;  // (N,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> qcol;  // (N,) q column

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> cs;  // (2,) (c, s) values

  JacobiSymSchur2();

  bool compute() {
    // TODO: fix sub-optimal loading?
    const unsigned p = pq[0];
    const unsigned q = pq[1];
    const T2 cs_vec = sym_schur2(pcol.data(), qcol.data(), p, q);
    cs[0] = cs_vec[0];
    cs[1] = cs_vec[1];
    return true;
  }
};

/**
 * @brief Jacobi algorithm, update first step: schur + column update.
 *
 * See:  Gene H. Golub, Charles F. Van Loan, MATRIX COMPUTATIONS, 3rd edition,
 * Johns Hopkins Chapter 8.
 */
class JacobiUpdateFirstStep : public Vertex {
 public:
  using T = float;
  using T2 = float2;
  using IndexType = unsigned short;

  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>> rotset;  // (2,) rotation index p and q. p < q
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> pcol;  // (N,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> qcol;  // (N,) q column

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> cs;  // (2,) (c, s) Schur decomposition values

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      pcol_updated;  // (N,) p column updated
  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      qcol_updated;  // (N,) q column updated

  const IndexType N;  // size

  JacobiUpdateFirstStep();

  bool compute() {
    const unsigned p = rotset[0];
    const unsigned q = rotset[1];
    const T Apq = pcol[q];
    const T App = pcol[p];
    const T Aqq = qcol[q];

    // Schur2 decomposition.
    const T2 cs_vec = sym_schur2(pcol.data(), qcol.data(), p, q);
    const T& c = cs_vec[0];
    const T& s = cs_vec[1];
    cs[0] = c;
    cs[1] = s;

    // Easier to vectorized + parallelize if start with full update.
    for (IndexType idx = 0; idx != N; ++idx) {
      const T pvalue = pcol[idx];
      const T qvalue = qcol[idx];

      pcol_updated[idx] = c * pvalue - s * qvalue;
      qcol_updated[idx] = s * pvalue + c * qvalue;
    }
    // Update main values App, Apq, Aqq
    pcol_updated[p] = c * c * App - 2 * s * c * Apq + s * s * Aqq;
    qcol_updated[q] = s * s * App + 2 * s * c * Apq + c * c * Aqq;

    // pcol_updated[q] = (c * c - s * s) * Apq + s * c * (App - Aqq);
    // qcol_updated[p] = pcol_updated[q];
    // Zero on purpose!
    pcol_updated[q] = 0;
    qcol_updated[p] = 0;

    return true;
  }
};

class JacobiUpdateSecondStep : public Vertex {
 public:
  using T = float;
  using T2 = float2;
  using IndexType = unsigned short;

  InOut<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      cs_arr;  // (N/2, 2) (c, s) values
  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>>
      rotset_arr;  // (N/2, 2) (p, q) array values. p < q
  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>>
      rotset_idx_ignored;  // (1,) index in rotset to ignore.

  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> pcol;  // (N,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> qcol;  // (N,) q column

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      pcol_updated;  // (N,) p column updated
  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      qcol_updated;  // (N,) q column updated

  // const unsigned ignore_idx;  // cs/pq index to ignore.
  const IndexType halfN;      // N / 2

  JacobiUpdateSecondStep();

  bool compute() {
    // Use (p, q) = (1, 0) for ignore idx.
    const unsigned ignore_idx = rotset_idx_ignored[0];
    cs_arr[2 * ignore_idx] = 1;
    cs_arr[2 * ignore_idx + 1] = 0;

    // Parallized loop on update using other columns coefficients
    for (IndexType half_idx = 0; half_idx != halfN; ++half_idx) {
      const unsigned k = rotset_arr[2 * half_idx];
      const unsigned l = rotset_arr[2 * half_idx + 1];

      const T c = cs_arr[2 * half_idx];
      const T s = cs_arr[2 * half_idx + 1];

      // 4 coefficients updates!
      // TODO: vectorization?!
      const T Spk = pcol[k];
      const T Spl = pcol[l];

      const T Sqk = qcol[k];
      const T Sql = qcol[l];

      pcol_updated[k] = c * Spk - s * Spl;
      pcol_updated[l] = s * Spk + c * Spl;

      qcol_updated[k] = c * Sqk - s * Sql;
      qcol_updated[l] = s * Sqk + c * Sql;
    }
    return true;
  }
};
