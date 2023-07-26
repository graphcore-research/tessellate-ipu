// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <pybind11/pybind11.h>

#include "ConvPartialsStridesPacking.hpp"

namespace ipu {

/**
 * @brief Make pybind11 bindings for IPU `dot` vertices utils.
 */
inline decltype(auto) makeIpuDotVertexUtilsBindings(pybind11::module& m) {
  m.def("ipuGetTransformedInStride", &poplin::getTransformedInStride,
        pybind11::arg("convUnitWeightHeight"), pybind11::arg("inStride"),
        pybind11::arg("inRowStride"), pybind11::arg("convInputLoadElems"),
        pybind11::arg("inChansPerGroup"));

  m.def("ipuGetTransformedInRowStride", &poplin::getTransformedInRowStride,
        pybind11::arg("inRowStride"), pybind11::arg("convInputLoadElems"),
        pybind11::arg("inChansPerGroup"));

  m.def("ipuGetTransformedOutStride", &poplin::getTransformedOutStride,
        pybind11::arg("outStride"), pybind11::arg("outChansPerGroup"),
        pybind11::arg("numConvUnitsRequired"), pybind11::arg("isPartialsFloat"),
        pybind11::arg("flipOut"));

  m.def("ipuReverseTransformedInStride", &poplin::reverseTransformedInStride,
        pybind11::arg("transformedInStride"),
        pybind11::arg("convInputLoadElems"), pybind11::arg("inChansPerGroup"),
        pybind11::arg("ampKernelHeight") = 0, pybind11::arg("inRowStride") = 0);

  m.def("ipuReverseTransformedInRowStride",
        &poplin::reverseTransformedInRowStride,
        pybind11::arg("transformedInStride"),
        pybind11::arg("convInputLoadElems"), pybind11::arg("inChansPerGroup"));

  m.def("ipuReverseTransformedOutStride", &poplin::reverseTransformedOutStride,
        pybind11::arg("transformedOutStride"),
        pybind11::arg("accumTypeIsFloat"), pybind11::arg("numConvUnits"),
        pybind11::arg("outChansPerGroup"));
}

}  // namespace ipu
