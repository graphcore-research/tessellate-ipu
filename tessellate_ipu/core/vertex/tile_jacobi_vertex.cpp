// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "intrinsics_utils.hpp"
#include "tile_small_dot.hpp"

using namespace poplar;

/**
 * @brief Compute the Schur decomposition of a symmetric 2x2 matrix.
 */
__attribute__((always_inline)) float2 sym_schur2(const float App,
                                                 const float Aqq,
                                                 const float Apq) noexcept {
  using T = float;
  using T2 = float2;

  // See Algorithm 8.4.1, MATRIX computations.
  // Avoid the division `tau` by keeping the two independent factors.
  const T Cpq = 2 * Apq;
  const T Dpq = Aqq - App;
  const T sq_Cpq = Cpq * Cpq;
  const T sq_Dpq = Dpq * Dpq;

  // Avoids dividing by zero/eps!
  // eps value fine-tuned on some DFT and random examples.
  constexpr T eps = 1e-12f;
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

  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>>
      pq;  // (2,) p and q indexes
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> pcol;  // (N,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> qcol;  // (N,) q column

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> cs;  // (2,) (c, s) values

  JacobiSymSchur2();

  bool compute() {
    // TODO: fix sub-optimal loading?
    const unsigned p = pq[0];
    const unsigned q = pq[1];
    const T Apq = pcol[q];
    const T App = pcol[p];
    const T Aqq = qcol[q];
    const T2 cs_vec = sym_schur2(App, Aqq, Apq);
    cs[0] = cs_vec[0];
    cs[1] = cs_vec[1];
    return true;
  }
};

template <class IpuTag, typename T>
void jacob_update_first_step(const T* pcol, const T* qcol, T* pcol_updated,
                             T* qcol_updated, T* cs, unsigned p, unsigned q,
                             unsigned short wstart,
                             unsigned short wend) noexcept {
  using T2 = float2;
  using IndexType = unsigned short;

  const T Apq = pcol[q];
  const T App = pcol[p];
  const T Aqq = qcol[q];
  // Schur2 decomposition.
  const T2 cs_vec = sym_schur2(App, Aqq, Apq);
  const T& c = cs_vec[0];
  const T& s = cs_vec[1];
  cs[0] = c;
  cs[1] = s;
  // Worker load: start + end vectorized indexes.
  const IndexType wsize = wend - wstart;

  // pcol, qcol and results pointers.
  const float2* ptr_pcol = reinterpret_cast<const float2*>(pcol) + wstart;
  const float2* ptr_qcol = reinterpret_cast<const float2*>(qcol) + wstart;
  float2* ptr_pcol_updated = reinterpret_cast<float2*>(pcol_updated) + wstart;
  float2* ptr_qcol_updated = reinterpret_cast<float2*>(qcol_updated) + wstart;
  // Apply Schur2 cs rotation to p/q columns (optimized kernel).
  rotation2d_f32<IpuTag>(cs_vec, ptr_pcol, ptr_qcol, ptr_pcol_updated,
                                ptr_qcol_updated, wsize);
  // Update main values App, Apq, Aqq
  pcol_updated[p] = c * c * App - 2 * s * c * Apq + s * s * Aqq;
  qcol_updated[q] = s * s * App + 2 * s * c * Apq + c * c * Aqq;
  // Zero on purpose with Schur decomposition!
  pcol_updated[q] = 0;
  qcol_updated[p] = 0;
}

/**
 * @brief Jacobi algorithm, update first step: schur + column update.
 *
 * See:  Gene H. Golub, Charles F. Van Loan, MATRIX COMPUTATIONS, 3rd edition,
 * Johns Hopkins Chapter 8.
 */
class [[poplar::constraint(
    "elem(*pcol) != elem(*pcol_updated)",
    "elem(*qcol) != elem(*qcol_updated)")]] JacobiUpdateFirstStep
    : public MultiVertex {
 public:
  using T = float;
  using T2 = float2;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  // p/q cols + index prefix (2 x uint32).
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> pcol;  // (N + 2,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> qcol;  // (N + 2,) q column

  Input<Vector<IndexType, poplar::VectorLayout::ONE_PTR>>
      worker_offsets;  // (7,) threads work size + 1.

  Output<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>>
      rotset_sorted;  // (3,) rotset index sorted + was sorted?
  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      cs;  // (2,) (c, s) Schur decomposition values

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      pcol_updated;  // (N + 2,) p column updated
  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      qcol_updated;  // (N + 2,) q column updated

  JacobiUpdateFirstStep();

  bool compute(unsigned wid) {
    // Size of the index prefix in pcol and qcol.
    constexpr int INDEX_PREFIX = 2;
    const unsigned p = *((unsigned*)pcol.data());
    const unsigned q = *((unsigned*)qcol.data());

    const IndexType wstart = worker_offsets[wid];
    const IndexType wend = worker_offsets[wid + 1];

    // Forward p/q indices.
    pcol_updated[0] = pcol[0];
    qcol_updated[0] = qcol[0];

    if (p <= q) {
      // Proper ordering of p and q already.
      jacob_update_first_step<IPU_TAG_TYPE>(
          pcol.data() + INDEX_PREFIX, qcol.data() + INDEX_PREFIX,
          pcol_updated.data() + INDEX_PREFIX,
          qcol_updated.data() + INDEX_PREFIX, cs.data(), p, q, wstart, wend);
      rotset_sorted[0] = p;
      rotset_sorted[1] = q;
    } else {
      // Swap p and q columns as q < p
      jacob_update_first_step<IPU_TAG_TYPE>(
          qcol.data() + INDEX_PREFIX, pcol.data() + INDEX_PREFIX,
          qcol_updated.data() + INDEX_PREFIX,
          pcol_updated.data() + INDEX_PREFIX, cs.data(), q, p, wstart, wend);
      rotset_sorted[0] = q;
      rotset_sorted[1] = p;
    }
    return true;
  }
};

class JacobiUpdateSecondStep : public MultiVertex {
 public:
  using T = float;
  using T2 = float2;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  InOut<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      cs_arr;  // (N/2, 2) (c, s) values
  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>>
      rotset_sorted_arr;  // (N/2, 2) (p, q) array values. p < q
  Input<Vector<unsigned, poplar::VectorLayout::ONE_PTR, 8>>
      rotset_idx_ignored;  // (1,) index in rotset to ignore.

  Input<Vector<IndexType, poplar::VectorLayout::ONE_PTR>>
      worker_offsets;  // (7,) threads work size + 1.

  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> pcol;  // (N+2,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> qcol;  // (N+2,) q column

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      pcol_updated;  // (N+2,) p column updated
  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      qcol_updated;  // (N+2,) q column updated

  JacobiUpdateSecondStep();

  bool compute(unsigned wid) {
    // Size of the index prefix in pcol and qcol.
    constexpr int INDEX_PREFIX = 2;
    // Worker load: start + end vectorized indexes.
    const IndexType wstart = worker_offsets[wid];
    const IndexType wend = worker_offsets[wid + 1];
    const IndexType wsize = wend - wstart;

    // Use (p, q) = (1, 0) for ignore idx.
    const unsigned ignore_idx = 2 * rotset_idx_ignored[0];
    cs_arr[ignore_idx] = 1;
    cs_arr[ignore_idx + 1] = 0;

    auto pcol_ptr = pcol.data() + INDEX_PREFIX;
    auto qcol_ptr = qcol.data() + INDEX_PREFIX;
    auto pcol_updated_ptr = pcol_updated.data() + INDEX_PREFIX;
    auto qcol_updated_ptr = qcol_updated.data() + INDEX_PREFIX;

    // Forward pq indices.
    pcol_updated[0] = pcol[0];
    qcol_updated[0] = qcol[0];

    // Parallized loop on update using other columns coefficients
    for (IndexType half_idx = 0; half_idx != wsize; ++half_idx) {
      // TODO: cleaning pq indices offset.
      const unsigned k = rotset_sorted_arr[2 * half_idx + 2 * wstart];
      const unsigned l = rotset_sorted_arr[2 * half_idx + 1 + 2 * wstart];

      const T c = cs_arr[2 * half_idx + 2 * wstart];
      const T s = cs_arr[2 * half_idx + 1 + 2 * wstart];

      // 4 coefficients updates!
      // TODO: vectorization?!
      const T Spk = pcol_ptr[k];
      const T Spl = pcol_ptr[l];

      const T Sqk = qcol_ptr[k];
      const T Sql = qcol_ptr[l];

      pcol_updated_ptr[k] = c * Spk - s * Spl;
      pcol_updated_ptr[l] = s * Spk + c * Spl;

      qcol_updated_ptr[k] = c * Sqk - s * Sql;
      qcol_updated_ptr[l] = s * Sqk + c * Sql;
    }
    return true;
  }
};

template <typename T>
void jacob_update_eigenvectors(const T* vpcol, const T* vqcol, T* vpcol_updated,
                               T* vqcol_updated, T c, T s,
                               unsigned short wstart,
                               unsigned short wend) noexcept {
  using T2 = float2;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  const T2 cvec = T2{c, c};
  const T2 svec = T2{s, s};
  const IndexType wsize = wend - wstart;

  // pcol, qcol and results pointers.
  const T2* ptr_pcol = reinterpret_cast<const T2*>(vpcol) + wstart;
  const T2* ptr_qcol = reinterpret_cast<const T2*>(vqcol) + wstart;
  T2* ptr_pcol_updated = reinterpret_cast<T2*>(vpcol_updated) + wstart;
  T2* ptr_qcol_updated = reinterpret_cast<T2*>(vqcol_updated) + wstart;

  for (IndexType idx = 0; idx != wsize; ++idx) {
    const T2 vpvec = ipu::load_postinc(&ptr_pcol, 1);
    const T2 vqvec = ipu::load_postinc(&ptr_qcol, 1);

    const T2 vpvec_updated = cvec * vpvec - svec * vqvec;
    const T2 vqvec_updated = svec * vpvec + cvec * vqvec;

    ipu::store_postinc(&ptr_qcol_updated, vqvec_updated, 1);
    ipu::store_postinc(&ptr_pcol_updated, vpvec_updated, 1);
  }
}

/**
 * @brief Jacobi algorithm, update of eigen vectors matrix.
 *
 * See:  Gene H. Golub, Charles F. Van Loan, MATRIX COMPUTATIONS, 3rd edition,
 * Johns Hopkins Chapter 8.
 */
class [[poplar::constraint(
    "elem(*vpcol) != elem(*vqcol)")]] JacobiUpdateEigenvectors
    : public MultiVertex {
 public:
  using T = float;
  using T2 = float2;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      cs;  // (2,) (c, s) Schur decomposition values
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> vpcol;  // (N,) p column
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> vqcol;  // (N,) q column

  Input<Vector<IndexType, poplar::VectorLayout::ONE_PTR>>
      worker_offsets;  // (7,) threads work size + 1.

  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      vpcol_out;  // (N,) p column
  Output<Vector<T, poplar::VectorLayout::ONE_PTR, 8>>
      vqcol_out;  // (N,) q column

  JacobiUpdateEigenvectors();

  bool compute(unsigned wid) {
    constexpr int INDEX_PREFIX = 2;
    const unsigned p = *((unsigned*)vpcol.data());
    const unsigned q = *((unsigned*)vqcol.data());

    const T c = cs[0];
    const T s = cs[1];
    const IndexType wstart = worker_offsets[wid];
    const IndexType wend = worker_offsets[wid + 1];

    // Forwarding p/q (prefix) indices.
    vpcol_out[0] = vpcol[0];
    vqcol_out[0] = vqcol[0];
    // Swapping pointers if necessary.
    if (p <= q) {
      jacob_update_eigenvectors(
          vpcol.data() + INDEX_PREFIX, vqcol.data() + INDEX_PREFIX,
          vpcol_out.data() + INDEX_PREFIX, vqcol_out.data() + INDEX_PREFIX, c,
          s, wstart, wend);
    } else {
      jacob_update_eigenvectors(
          vqcol.data() + INDEX_PREFIX, vpcol.data() + INDEX_PREFIX,
          vqcol_out.data() + INDEX_PREFIX, vpcol_out.data() + INDEX_PREFIX, c,
          s, wstart, wend);
    }
    return true;
  }
};
