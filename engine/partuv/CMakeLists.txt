# 3.18 needed for file(ARCHIVE_EXTRACT) in fetch_extern
cmake_minimum_required(VERSION 3.18)


# set(ENV{NVCC_PREPEND_FLAGS} "--allow-unsupported-compiler")
# # (Optional) also initialize CMake's CUDA flags with it
# set(CMAKE_CUDA_FLAGS_INIT "--allow-unsupported-compiler")
# default targets the RTX 3060 (sm_86), override with -DCMAKE_CUDA_ARCHITECTURES
# or the CUDAARCHS env var
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
# windows deps come from vcpkg; VCPKG_ROOT is set in a vs dev shell
if(WIN32 AND DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(CMAKE_CUDA_COMPILER NAMES nvcc)
  if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDA_PATH})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe")
  elseif(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
    message(WARNING "Could not find 'nvcc' in PATH. Defaulting to /usr/local/cuda-12.1/bin/nvcc.")
  endif()
endif()

project(partuv LANGUAGES CXX CUDA)

# Set up shared dependencies at the root level
set(CMAKE_CXX_STANDARD 17)


# cuda 13 (windows build) requires c++17 device code
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")
# set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type" FORCE)

# message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

# x86-64-v3 (avx2) instead of upstream's -march=native so the wheel runs on
# other machines; PARTUV_NATIVE=ON restores the machine-tuned build
option(PARTUV_NATIVE "Tune for the build machine cpu" OFF)
if(MSVC)
  # conformance mode: sources use gcc-default alternative tokens (not/and/or),
  # and openabf gates code on a truthful __cplusplus
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive-> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
  # msvc's default openmp runtime is 2.0; pipeline.cpp needs 3.x api
  # (omp_get_level etc.), only in the llvm runtime. needs cmake 3.30+
  set(OpenMP_RUNTIME_MSVC "llvm")
  # msvc hides M_PI (used by openabf) behind this
  add_compile_definitions(_USE_MATH_DEFINES)
  # avx2 matches x86-64-v3; cuda 13 headers reject the traditional preprocessor
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " /fp:fast /arch:AVX2 /Zc:preprocessor")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -mfpmath=sse -Ofast -fno-omit-frame-pointer")
  if(PARTUV_NATIVE)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
  else()
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=x86-64-v3")
  endif()
endif()
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
# elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
#     add_compile_definitions(DEBUG_MODE)
# endif()

option(ENABLE_PROFILING "Enable or disable profiling timers" ON)

# Define paths to shared dependencies
set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
# set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(LIBIGL_INCLUDE_DIR ${EXTERNAL_DIR}/libigl/include)
set(EIGEN_INCLUDE_DIR ${EXTERNAL_DIR}/eigen-3.4.0)
set(OPENABF_INCLUDE_DIR ${EXTERNAL_DIR}/OpenABF/include)
set(STB_INCLUDE_DIR ${EXTERNAL_DIR}/stb)
set(NLOHMANN_JSON_INCLUDE_DIR ${EXTERNAL_DIR}/json/include)

# eigen, libigl and json are gitignored (large), download them when missing
function(fetch_extern out_dir url dest extracted)
  if(EXISTS "${out_dir}")
    return()
  endif()
  set(archive "${CMAKE_CURRENT_BINARY_DIR}/${extracted}.archive")
  message(STATUS "Downloading ${url}")
  file(DOWNLOAD "${url}" "${archive}" STATUS dl_status)
  list(GET dl_status 0 dl_code)
  if(NOT dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download ${url}")
  endif()
  file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${dest}")
  if(NOT "${dest}/${extracted}" STREQUAL "${out_dir}")
    file(RENAME "${dest}/${extracted}" "${out_dir}")
  endif()
  file(REMOVE "${archive}")
endfunction()

fetch_extern("${EIGEN_INCLUDE_DIR}"
  "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
  "${EXTERNAL_DIR}" "eigen-3.4.0")
# libigl 2.5 removed igl/slice_mask.h, so stay on 2.4
fetch_extern("${EXTERNAL_DIR}/libigl"
  "https://github.com/libigl/libigl/archive/refs/tags/v2.4.0.tar.gz"
  "${EXTERNAL_DIR}" "libigl-2.4.0")
fetch_extern("${NLOHMANN_JSON_INCLUDE_DIR}"
  "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"
  "${EXTERNAL_DIR}/json" "include")

# set(UVPSDK_ROOT "${EXTERNAL_DIR}/uvpackmaster"          # extern/uvpackmaster
#     CACHE PATH "Path to UVPackmaster SDK (include/ + lib/)")
# find_path(UVPCORE_INCLUDE_DIR
# NAMES uvpCore.hpp
# PATHS "${UVPSDK_ROOT}/include"
# NO_DEFAULT_PATH REQUIRED)

# find_library(UVPCORE_LIBRARY
#        NAMES uvpcore libuvpcore
#        PATHS "${UVPSDK_ROOT}/lib"
#        NO_DEFAULT_PATH REQUIRED)

# add_library(uvpcore_sdk SHARED IMPORTED GLOBAL)
# set_target_properties(uvpcore_sdk PROPERTIES
#         IMPORTED_LOCATION             "${UVPCORE_LIBRARY}"
#         INTERFACE_INCLUDE_DIRECTORIES "${UVPCORE_INCLUDE_DIR}")

#   file(GLOB UVP_LIBS "${UVPSDK_ROOT}/lib/libuvpcore.so")


#   if(UVP_LIBS)
#     install(FILES ${UVP_LIBS}  DESTINATION partuv/.libs)
#   endif()
# Make these paths available to all subprojects
include_directories(
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
    ${OPENABF_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# include_directories(${EXTERNAL_DIR}/pybind11/include)
# include_directories("/usr/include/suitesparse")
# Find common dependencies

find_package(CGAL REQUIRED)
# the built library is optional: without it the EASY_ macros in the vendored
# easy_profiler headers compile to no-ops. the cmake config errors on the
# missing .so, so check for the library itself before find_package
if(EXISTS "${EXTERNAL_DIR}/easy_profiler/lib/libeasy_profiler.so")
  find_package(easy_profiler REQUIRED
      HINTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/easy_profiler/lib/cmake/easy_profiler"
  )
endif()
if(TARGET easy_profiler)
  get_target_property(_EP_SO easy_profiler IMPORTED_LOCATION_RELEASE)
  if(NOT _EP_SO)
    get_target_property(_EP_SO easy_profiler IMPORTED_LOCATION)  # fallback if only generic set
  endif()
  install(FILES "${_EP_SO}" DESTINATION partuv/.libs)
else()
  message(STATUS "easy_profiler library not found, profiling disabled")
  add_library(easy_profiler INTERFACE)
  target_include_directories(easy_profiler
    INTERFACE "${EXTERNAL_DIR}/easy_profiler/include")
endif()

find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED) 

# Find pybind11 for Python bindings
find_package(pybind11 CONFIG REQUIRED)



# Add the LSCM library subdirectory
add_subdirectory(${SRC_DIR}/lscm)
set_target_properties(lscm_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${SRC_DIR}/triangle_intersection)
set_target_properties(triangle_intersection PROPERTIES POSITION_INDEPENDENT_CODE ON)



# add_subdirectory(${EXTERNAL_DIR}/pybind11)

# Collect source files:
# 1. Add main.cpp from the project root
option(USE_ALL_SRC_FILES "Use all .cpp files in src directory" ON)

if(USE_ALL_SRC_FILES)
    file(GLOB SOURCES "${SRC_DIR}/*.cpp")
    file(GLOB CUDA_SOURCES "${SRC_DIR}/*.cu")
    set(BINDINGS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp")
    list(APPEND SOURCES ${CUDA_SOURCES})
    list(APPEND SOURCES ${BINDINGS_SOURCES})
    message(STATUS "Using all source files in ${SRC_DIR}")
else()
    set(SOURCES
        "${SRC_DIR}/Distortion.cpp"
        "${SRC_DIR}/IO.cpp"
    )
    message(STATUS "Using selected source files only")
endif()



set(ENV{OMP_NUM_THREADS} "8")

# Create Python module
# pybind11_add_module(partuv src/bindings.cpp)
# NO_EXTRAS: pybind11's lto drops the cuda fatbin from the device-link object
# (undefined symbol fatbinData at import)
pybind11_add_module(_core NO_EXTRAS bindings.cpp ${SOURCES})

# --- allow device‐side lambdas & cross‐TU __device__ calls ---
set_target_properties(_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions(_core PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)



# --- pass NVCC flags only when compiling CUDA code ---
target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-extended-lambda      # enable __device__ lambdas
    --expt-relaxed-constexpr    # (optional) allows constexpr on device
    --default-stream=per-thread
  >
)
if(MSVC)
  # cuda 13 headers reject msvc's traditional preprocessor in host passes too
  target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
  )
endif()
# Link the Python module with the same dependencies as the main executable
target_link_libraries(_core
    PUBLIC
        easy_profiler
        # uvpcore_sdk          
    PRIVATE
        lscm_lib
        triangle_intersection
        OpenMP::OpenMP_CXX
        yaml-cpp
        TBB::tbb
        CUDA::cudart
)

# Set include directories for the Python module
target_include_directories(_core PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBIGL_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
    ${OPENABF_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# --- Add at the end of CMakeLists.txt ---
# install(TARGETS _core LIBRARY DESTINATION partuv)
install(TARGETS _core DESTINATION partuv)
# install(FILES DESTINATION partuv/.libs)

if(MSVC)
  # bundle the non-system runtime dlls so the wheel imports anywhere;
  # msvcp140/libomp140 come from the vc++ redistributable instead (libomp
  # is nonredist in the vs tree), cudart is linked statically
  set(_vcpkg_bin "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
  foreach(_dll gmp-10.dll mpfr-6.dll yaml-cpp.dll)
    if(NOT EXISTS "${_vcpkg_bin}/${_dll}")
      message(FATAL_ERROR "runtime dll not found: ${_vcpkg_bin}/${_dll}")
    endif()
    install(FILES "${_vcpkg_bin}/${_dll}" DESTINATION partuv/.libs)
  endforeach()
endif()

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/partuv/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/partuv/preprocess.py
  DESTINATION partuv)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/preprocess_utils/
  DESTINATION partuv/preprocess_utils
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.yaml"
    PATTERN "*.yml"
    PATTERN "*.json"
    PATTERN "*.txt"
    PATTERN "*.pt"
    PATTERN "*.bin"
)
set_target_properties(_core PROPERTIES
  # so the built artifact also works from the build tree
  BUILD_RPATH  "\$ORIGIN;\$ORIGIN/.libs"
  # so the installed wheel works after pip install
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/.libs"
)