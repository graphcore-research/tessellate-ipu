// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

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

using namespace poplar;

// popc -O2 jax_ipu_research/tile/vertex/tile_qr_vertex.cpp -o
// jax_ipu_research/tile/vertex/tile_qr_vertex.gp
/**
 * Compilation for all supported targets:
 *      popc -O2 jax_ipu_research/tile/vertex/tile_qr_vertex.cpp -o
 * jax_ipu_research/tile/vertex/tile_qr_vertex.gp
 */
template <typename T>
class LinalgQRVertex : public Vertex {
 public:
  Input<Vector<T, poplar::VectorLayout::SPAN>> x;      // flatten (N,     N)
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> Q;  // flatten (N, N) Q
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> R;  // flatten (N, N) R

  Output<Vector<T, poplar::VectorLayout::ONE_PTR>>
      tmp;  // (N, 3) tmp scratch space.

  bool compute() {
    // Assumes square input matrix.
    std::size_t R_shape = std::sqrt(x.size());
    std::size_t num_cols = std::sqrt(x.size());

    T* diag = &tmp[0];
    T* v = &tmp[num_cols];
    T* intermediary = &tmp[num_cols * 2];

    // Copy input x into matrix R
    // Zero out matrix Q (may have -infs at init).
    for (std::size_t i = 0; i < x.size(); i++) {
      R[i] = x[i];
      Q[i] = 0;
    }

    // Initialize Q as the identity.
    for (std::size_t i = 0; i < R_shape; ++i) Q[i * num_cols + i] = 1.;

    // Save diagonal entries of R:
    //    diag[i] = R[i,i]
    for (std::size_t i = 0; i < R_shape; ++i) diag[i] = R[i * num_cols + i];

    // Main for loop of QR deceomposition.
    // Each iteration zero's out a column of R.
    for (std::size_t i = 0; i < R_shape - 1; ++i) {
      // Copy the elements of R into x, but zero out elements above diagonal.
      for (std::size_t k = 0; k < i; ++k) v[k] = 0;
      for (std::size_t k = i; k < R_shape; ++k) v[k] = R[k * num_cols + i];

      // Compute the l2 norm ||v||=sqrt(sum_i v_i**2).
      float norm = 0.;
      for (std::size_t k = 0; k < R_shape; ++k) {
        norm += v[k] * v[k];
      }
      norm = std::sqrt(norm);

      // Change the entry of v that corresponds to the diagonal element of R.
      v[i] -= norm * diag[i] / std::abs(diag[i]);  // TODO: using sign(x)=x/|x|

      // Compute the l2 norm ||v||=sqrt(sum_i v_i**2).
      norm = 0.;
      for (std::size_t k = 0; k < R_shape; ++k) norm += v[k] * v[k];
      norm = std::sqrt(norm);

      // Normalize v by the new norm.
      for (std::size_t k = 0; k < R_shape; ++k) v[k] = v[k] / norm;

      // use row 10 for intermediary
      // intermediary *= 0
      // for (std::size_t k = 0; k < R_shape; ++k) { out[4*10+k] = 0.; }
      // for i in range(R.shape[0]):
      //        for j in range(R.shape[1]):
      //                intermediary[i] += R[j,i] * v[j]
      for (std::size_t l = 0; l < R_shape; l++) {
        intermediary[l] = 0.;
        for (std::size_t k = 0; k < R_shape; k++) {
          intermediary[l] += R[k * num_cols + l] * v[k];
        }
      }

      // R[,] is stored in out[16:15+16]
      // for i in range(R.shape[0]):
      //        for j in range(R.shape[1]):
      //                R[i,j] = R[i,j] - 2 * v[i] * intermediary[j]
      // zero out intermediary vector, the 10th row
      for (std::size_t l = 0; l < R_shape; l++) {
        for (std::size_t k = 0; k < R_shape; k++) {
          R[l * num_cols + k] -= 2 * v[l] * intermediary[k];
        }
      }

      // Q = Q - 2 * (Q @ v.reshape(-1, 1)) @ v.reshape(1, -1)
      // use row 10 for intermediary
      // intermediary *= 0
      // for i in range(Q.shape[0]):
      //        for j in range(Q.shape[1]):
      //                intermediary[i] += Q[i,j] * v[j]
      for (std::size_t l = 0; l < R_shape; l++) {  // Q is out[:16]
        intermediary[l] = 0.;
        for (std::size_t k = 0; k < R_shape; k++) {
          intermediary[l] += Q[l * num_cols + k] * v[k];
        }
      }
      // for i in range(Q.shape[0]):
      //        for j in range(Q.shape[1]):
      //                Q[i,j] -= 2  *intermediary[i] * v[j]
      for (std::size_t l = 0; l < R_shape; l++) {
        for (std::size_t k = 0; k < R_shape; k++) {
          Q[l * num_cols + k] -= 2 * intermediary[l] * v[k];
        }
      }
    }

    return true;
  }
};

// explicit instantiations
template class LinalgQRVertex<float>;

/**
 * @brief Vertex implementing the inplace householder update in the QR
 * algorithm.
 *
 * More specifically:
 *  x -= 2 * w @ v.T
 */
template <typename T>
class QRHouseholderUpdateVertex : public Vertex {
 public:
  InOut<Vector<T, poplar::VectorLayout::ONE_PTR>> x;  // flatten (N, N)

  Input<Vector<T, poplar::VectorLayout::SPAN>> v;     // (N,) v
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> w;  // (N,) w

  bool compute() {
    const std::size_t size = v.size();

    for (std::size_t r = 0; r < size; r++) {
      T* data = &x[r * size];
      // Pre-compute the full scaling factor for every line.
      // const T scale = T(-2) * w[r];
      const T scale = -w[r];
      for (std::size_t c = 0; c < size; c++) {
        data[c] += scale * v[c];
      }
    }
    return true;
  }
};

// explicit instantiations
template class QRHouseholderUpdateVertex<float>;
template class QRHouseholderUpdateVertex<half>;

/**
 * @brief Vertex computing the correction vector in the QR algorithm.
 */
class QRCorrectionVectorVertex : public MultiVertex {
 public:
  using T = float;
  Input<Vector<T, poplar::VectorLayout::SPAN>> Rcol;      // (N,) R column.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> sdiag;  // (N,) R diag. sign.
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> v;  // (N,) correction vec.

  const unsigned col_idx;  // R column index.

  // Use static variables as easy sync. mechanisms between worker threads.
  // see: https://graphcore.slack.com/archives/C013LPHPX61/p1647443852989259
  static T shared_partial_sqnorms[6];
  static T shared_updated_scale;

  QRCorrectionVectorVertex();

  bool compute(unsigned wid) {
    const unsigned num_workers = 6;
    const unsigned step = num_workers * 2;
    const unsigned size = Rcol.size();

    const unsigned col_idx_rem = col_idx % 2;
    const unsigned col_idx_mrem = col_idx - col_idx_rem;
    const unsigned col_idx_prem = col_idx + col_idx_rem;

    const T initial_rcol_val = Rcol[col_idx];
    const T initial_rcol_val_sq = initial_rcol_val * initial_rcol_val;

    // FIRST: SET SHARED STATE between workers.
    shared_partial_sqnorms[wid] = -1;
    shared_updated_scale = 0;

    const float2 zeros_f2{0, 0};
    float2* ptr_outdata_f2 = reinterpret_cast<float2*>(v.data()) + wid;
    // Push to col_idx_prem, may write one zero too much, but does not matter!
    float2* ptr_outdata_end_f2 = reinterpret_cast<float2*>(&v[col_idx_prem]);
    // First chunk of v initialized with zeros.
    while (ptr_outdata_f2 < ptr_outdata_end_f2) {
      ipu::store_postinc(&ptr_outdata_f2, zeros_f2, num_workers);
    }

    float2 partials_f2{0, 0};
    const float2* ptr_indata_f2 =
        reinterpret_cast<const float2*>(&Rcol[col_idx_prem]) + wid;
    ptr_outdata_f2 = reinterpret_cast<float2*>(&v[col_idx_prem]) + wid;
    ptr_outdata_end_f2 = reinterpret_cast<float2*>(&v[size]);
    // Copy Rcol data and accumulate squared norm.
    while (ptr_outdata_f2 < ptr_outdata_end_f2) {
      const float2 v = ipu::load_postinc(&ptr_indata_f2, num_workers);
      partials_f2 += v * v;
      ipu::store_postinc(&ptr_outdata_f2, v, num_workers);
    }
    T partial = partials_f2[0] + partials_f2[1];

    // GLOBAL STATE shared by all workers.
    shared_partial_sqnorms[wid] = partial;

    if (wid == 0) {
      // Special case of odd R column index.
      // On thread 0: correction to squared normed depending on `col_idx_rem`
      T norm_squared = partial + col_idx_rem * initial_rcol_val_sq;
      // Accumulate & wait.
      for (unsigned w = 1; w < num_workers; ++w) {
        // Avoid compiler optimizer with volatile pointer.
        volatile T* ptr_partial = &shared_partial_sqnorms[w];
        while (*ptr_partial < 0) {
        }
        norm_squared += shared_partial_sqnorms[w];
      }

      // Compute the norm.
      const T norm = std::sqrt(norm_squared);
      // Change the entry of v that corresponds to the diagonal element of R.
      const auto update_vidx_val = initial_rcol_val - norm * sdiag[col_idx];
      // Re-writing the full new value is faster than updating.
      v[col_idx] = update_vidx_val;

      // Update the squared norm of v.
      norm_squared -= initial_rcol_val_sq;
      norm_squared += update_vidx_val * update_vidx_val;

      // Updating scaling factor, including the Householder constant value.
      const T inv_norm = T(1) / std::sqrt(norm_squared);
      shared_updated_scale = std::sqrt(2.0f) * inv_norm;
    }

    // Wait until updated scale available to all threads.
    volatile T* ptr_shared_updated_scale = &shared_updated_scale;
    if (wid > 0) {
      while (*ptr_shared_updated_scale == 0) {
      }
    }
    // Rescaling float2 vector
    const float2 rescale_vec{*ptr_shared_updated_scale,
                             *ptr_shared_updated_scale};

    // 64 bits aligned start idx/ptr (removing unnecessary zeros!)
    ptr_outdata_f2 = reinterpret_cast<float2*>(&v[col_idx_mrem]) + wid;
    ptr_outdata_end_f2 = reinterpret_cast<float2*>(&v[size]);
    while (ptr_outdata_f2 < ptr_outdata_end_f2) {
      ptr_indata_f2 = reinterpret_cast<const float2*>(ptr_outdata_f2);
      // Load, rescale and store.
      float2 v = ipu::load_postinc(&ptr_indata_f2, 0);
      ipu::store_postinc(&ptr_outdata_f2, v * rescale_vec, num_workers);
    }
    return true;
  }
};

float QRCorrectionVectorVertex::shared_partial_sqnorms[6] = {-1};
float QRCorrectionVectorVertex::shared_updated_scale = 0;
