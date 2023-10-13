// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <poplar/Vertex.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include "poplar/TileConstants.hpp"
#include <print.h>

#include "intrinsics_utils.hpp"

using namespace poplar;

class TridiagonalSolverVertex : public Vertex {
public:
  InOut<Vector<float, poplar::VectorLayout::SPAN, 8>> d;    // b  contains x
  InOut<Vector<float, poplar::VectorLayout::ONE_PTR, 8>> u;   // c
  InOut<Vector<float, poplar::VectorLayout::ONE_PTR, 8>> l;   // a
  InOut<Vector<float, poplar::VectorLayout::ONE_PTR, 8>> b;     // d

  TridiagonalSolverVertex();

//#define PIVOTING

#ifndef PIVOTING
  bool compute() {

    int n = d.size();

    for (int i=1; i<n; i++){
        float w;
        w = l[i] / d[i-1];    // CHECK div-by-0 or OVFL
        d[i] -= w * u[i-1];
        b[i] = b[i] - w * b[i-1];
    }

    d[n-1] = b[n-1] / d[n-1];
    for (int i=n-2; i>=0; i--) {
        d[i] = (b[i] - u[i] * d[i+1]) / d[i];    // We put x into d?
    }

    // Maybe we should compute the norm of the delta between x and d?
    return true;
  }

#else

/*
for i in range(1,n):
    if np.abs(d[i-1]) < np.abs(l[i]):
        w = d[i-1] / l[i]
        u_im1 = u[i-1]
        d[i-1] = l[i]
        u[i-1] = d[i]
        l[i-1] = u[i]

        d[i] = u_im1 - w * d[i]
        u[i] = -w * u[i]
        b_i = b[i]
        b[i] = b[i-1] - b[i] * w
        b[i-1] = b_i
        # l[i] = 0
    else:
        w = l[i] / d[i-1]
        d[i] -= w * u[i-1]
        b[i] = b[i] - w * b[i-1]
        l[i-1] = 0  # or l[i]


d[n-1] = b[n-1] / d[n-1]
d[n-2] = (b[n-2] - u[n-2] * d[n-1]) / d[n-2]
for i in range(n-3,-1,-1):
    d[i] = (b[i] - u[i] * d[i+1] - l[i] * d[i+2]) / d[i]
*/




  bool compute() {
    float u_im1;    // temporary for u[i-1]
    float b_i;      // temporary for b[i]  (could reuse u_im1?)
    float w;
    int n = d.size();

    for (int i=1; i<n; i++){
      if (fabs(d[i-1]) < fabs(l[i])) {
          w = d[i-1] / l[i];
          u_im1 = u[i-1];
          d[i-1] = l[i];
          u[i-1] = d[i];
          l[i-1] = u[i];

          d[i] = u_im1 - w * d[i];
          u[i] = -w * u[i];
          b_i = b[i];
          b[i] = b[i-1] - b[i] * w;
          b[i-1] = b_i;
      }
    else {
        w = l[i] / d[i-1];
        d[i] -= w * u[i-1];
        b[i] = b[i] - w * b[i-1];
        l[i-1] = 0;  // or l[i]
      }
    }

    d[n-1] = b[n-1] / d[n-1];
    d[n-2] = (b[n-2] - u[n-2] * d[n-1]) / d[n-2];

    for (int i=n-3; i>=0; i--) {
      d[i] = (b[i] - u[i] * d[i+1] - l[i] * d[i+2]) / d[i];
    }
    return true;
  }
#endif

};
