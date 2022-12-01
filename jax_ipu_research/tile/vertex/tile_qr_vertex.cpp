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

/**
 * Compilation for all supported targets:
 *      popc -O2 jax_ipu_research/tile/vertex/tile_eigh_vertex.cpp -o
 * jax_ipu_research/tile/vertex/tile_eigh_vertex.gp
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
      const T scale = T(-2) * w[r];
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
 * @brief Vertex implementing the inplace householder update in the QR
 * algorithm.
 *
 * More specifically:
 *  x -= 2 * w @ v.T
 */
template <typename T>
class QRCorrectionVectorVertex : public Vertex {
 public:
  Input<Vector<T, poplar::VectorLayout::SPAN>> Rcol;      // (N,) R column.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> sdiag;  // (N,) R diag. sign.
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> v;  // (N,) correction vec.

  const unsigned col_idx;  // R column index.

  QRCorrectionVectorVertex();

  bool compute() {
    const std::size_t size = Rcol.size();
    // Initialization of v
    for (std::size_t k = 0; k < col_idx; ++k) {
      v[k] = 0;
    }
    for (std::size_t k = col_idx; k < size; ++k) {
      v[k] = Rcol[k];
    }

    // Compute the l2 norm ||v||=sqrt(sum_i v_i**2).
    float norm = 0.;
    for (std::size_t k = 0; k < size; ++k) {
      norm += v[k] * v[k];
    }
    norm = std::sqrt(norm);

    // Change the entry of v that corresponds to the diagonal element of R.
    v[col_idx] -= norm * sdiag[col_idx];  // TODO: using sign(x)=x/|x|

    // Compute the l2 norm ||v||=sqrt(sum_i v_i**2).
    norm = 0.;
    for (std::size_t k = 0; k < size; ++k) {
      norm += v[k] * v[k];
    }
    norm = std::sqrt(norm);

    // Normalize v by the new norm.
    for (std::size_t k = 0; k < size; ++k) {
      v[k] = v[k] / norm;
    }
    return true;
  }
};

// explicit instantiations
template class QRCorrectionVectorVertex<float>;
template class QRCorrectionVectorVertex<half>;
