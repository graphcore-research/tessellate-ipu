// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include "ConvPartialsStridesPacking.hpp"

namespace ipu {

/**
 * @brief Make nanobind bindings for IPU `dot` vertices utils.
 */
inline decltype(auto) makeIpuDotVertexUtilsBindings(nanobind::module_& m) {
  m.def("ipuGetTransformedInStride", &poplin::getTransformedInStride,
        nanobind::arg("convUnitWeightHeight"), nanobind::arg("inStride"),
        nanobind::arg("inRowStride"), nanobind::arg("convInputLoadElems"),
        nanobind::arg("inChansPerGroup"));

  m.def("ipuGetTransformedInRowStride", &poplin::getTransformedInRowStride,
        nanobind::arg("inRowStride"), nanobind::arg("convInputLoadElems"),
        nanobind::arg("inChansPerGroup"));

  m.def("ipuGetTransformedOutStride", &poplin::getTransformedOutStride,
        nanobind::arg("outStride"), nanobind::arg("outChansPerGroup"),
        nanobind::arg("numConvUnitsRequired"), nanobind::arg("isPartialsFloat"),
        nanobind::arg("flipOut"));

  m.def("ipuReverseTransformedInStride", &poplin::reverseTransformedInStride,
        nanobind::arg("transformedInStride"),
        nanobind::arg("convInputLoadElems"), nanobind::arg("inChansPerGroup"),
        nanobind::arg("ampKernelHeight") = 0, nanobind::arg("inRowStride") = 0);

  m.def("ipuReverseTransformedInRowStride",
        &poplin::reverseTransformedInRowStride,
        nanobind::arg("transformedInStride"),
        nanobind::arg("convInputLoadElems"), nanobind::arg("inChansPerGroup"));

  m.def("ipuReverseTransformedOutStride", &poplin::reverseTransformedOutStride,
        nanobind::arg("transformedOutStride"),
        nanobind::arg("accumTypeIsFloat"), nanobind::arg("numConvUnits"),
        nanobind::arg("outChansPerGroup"));
}

}  // namespace ipu
