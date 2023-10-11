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
  InOut<Vector<float, poplar::VectorLayout::SPAN, 8>> ts;    // b  contains x
  Input<Vector<float, poplar::VectorLayout::ONE_PTR, 8>> tus;   // c
  Input<Vector<float, poplar::VectorLayout::ONE_PTR, 8>> tls;   // a
  Input<Vector<float, poplar::VectorLayout::ONE_PTR, 8>> b;     // d

  TridiagonalSolverVertex();

  bool compute() {
    Vector<float, poplar::VectorLayout::ONE_PTR, 8> b_;
    for (int i=0; i<ts.size(); i++)
        b_[i] = b[i];

    int n = ts.size();

    for (int i=1; i<n; i++){
        float w;
        w = tls[i] / ts[i-1];    // CHECK div-by-0 or OVFL
        ts[i] -= w * tus[i-1];
        b_[i] -= w * b_[i-1];
    }

    ts[n-1] = b_[n-1] / ts[n-1];
    for (int i=n-2; i>0; i--) {
        ts[i] = (b_[i] - tus[i] * ts[i+1]) / ts[i];    // We put x into ts?
    }

    // Maybe we should compute the norm of the delta between x and ts?
    return true;
  }
};
