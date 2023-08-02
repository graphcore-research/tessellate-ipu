cmake_minimum_required(VERSION 3.15...3.26)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

if(NOT SKBUILD)
  message(
    WARNING
      "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject] ninja
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

message(==>PROJECT_NAME="${SKBUILD_PROJECT_NAME}")

# Python components needed by nanobind
find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Common TessellateIPU static library.
add_library(
  ${SKBUILD_PROJECT_NAME}_core STATIC
  # External dependencies (and poplibs utils).
  tessellate_ipu/external/fastbase64/chromiumbase64.cpp
  tessellate_ipu/external/fastbase64/fastavxbase64.cpp
  tessellate_ipu/lib/poplin/ConvPartialsStridesPacking.cpp
  # Common types & ops.
  tessellate_ipu/lib/base_types.cpp
  tessellate_ipu/lib/tile_array_utils.cpp
  tessellate_ipu/lib/tile_array_ops.cpp
  tessellate_ipu/lib/tile_map_ops.cpp)
target_include_directories(${SKBUILD_PROJECT_NAME}_core
                           PUBLIC tessellate_ipu/external/)
target_link_libraries(${SKBUILD_PROJECT_NAME}_core PUBLIC poplar poputil)
target_compile_options(${SKBUILD_PROJECT_NAME}_core PUBLIC -mavx2 -fPIC)
target_compile_definitions(${SKBUILD_PROJECT_NAME}_core
                           PUBLIC VERSION_INFO=${PROJECT_VERSION})
target_compile_features(${SKBUILD_PROJECT_NAME}_core PUBLIC cxx_std_17)

# Common TessellateIPU library Python bindings.
nanobind_add_module(
  py${SKBUILD_PROJECT_NAME}_core
  # Target the stable ABI for Python 3.12+, which reduces the number of binary
  # wheels that must be built. This does nothing on older Python versions
  STABLE_ABI
  # Build libnanobind statically and merge it into the extension (which itself
  # remains a shared library)
  NB_STATIC
  # Common TessellateIPU library Python bindings.
  tessellate_ipu/lib/tessellate_ipu_core.cpp)
target_link_libraries(py${SKBUILD_PROJECT_NAME}_core
                      PRIVATE ${SKBUILD_PROJECT_NAME}_core)

# TessellateIPU library JAX ops bindings.
nanobind_add_module(
  py${SKBUILD_PROJECT_NAME}_ops_jax
  # Target the stable ABI for Python 3.12+, which reduces the number of binary
  # wheels that must be built. This does nothing on older Python versions
  STABLE_ABI
  # Build libnanobind statically and merge it into the extension (which itself
  # remains a shared library)
  NB_STATIC
  # TessellateIPU library JAX ops bindings.
  tessellate_ipu/lib/tessellate_ipu_ops_jax.cpp)
target_link_libraries(py${SKBUILD_PROJECT_NAME}_ops_jax
                      PRIVATE ${SKBUILD_PROJECT_NAME}_core)

# TODO: TessellateIPU library Poptorch ops bindings.

install(TARGETS py${SKBUILD_PROJECT_NAME}_core
        DESTINATION ${SKBUILD_PROJECT_NAME}/lib)
install(TARGETS py${SKBUILD_PROJECT_NAME}_ops_jax
        DESTINATION ${SKBUILD_PROJECT_NAME}/lib)
