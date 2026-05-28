cmake_minimum_required(VERSION 3.10)


# set(ENV{NVCC_PREPEND_FLAGS} "--allow-unsupported-compiler")
# # (Optional) also initialize CMake's CUDA flags with it
# set(CMAKE_CUDA_FLAGS_INIT "--allow-unsupported-compiler")
set(CMAKE_CUDA_ARCHITECTURES 80 89 90 120)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(CMAKE_CUDA_COMPILER NAMES nvcc)
  if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
    message(WARNING "Could not find 'nvcc' in PATH. Defaulting to /usr/local/cuda-12.1/bin/nvcc.")
  endif()
endif()

project(partuv LANGUAGES CXX CUDA)

# Set up shared dependencies at the root level
set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")
# set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type" FORCE)

# message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -mfpmath=sse -Ofast -fno-omit-frame-pointer -march=native")
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
find_package(easy_profiler REQUIRED
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/easy_profiler/lib/cmake/easy_profiler"
) 
get_target_property(_EP_SO easy_profiler IMPORTED_LOCATION_RELEASE)
if(NOT _EP_SO)
  get_target_property(_EP_SO easy_profiler IMPORTED_LOCATION)  # fallback if only generic set
endif()
install(FILES "${_EP_SO}" DESTINATION partuv/.libs)

find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED) 

# Find pybind11 for Python bindings
find_package(pybind11 CONFIG REQUIRED)



# Add the LSCM library subdirectory
add_subdirectory(${SRC_DIR}/lscm)
target_compile_options(lscm_lib PUBLIC -fPIC)
add_subdirectory(${SRC_DIR}/triangle_intersection)
target_compile_options(triangle_intersection PUBLIC -fPIC)



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
pybind11_add_module(_core bindings.cpp ${SOURCES})

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