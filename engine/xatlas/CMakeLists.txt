cmake_minimum_required(VERSION 3.15)
project(xatlas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# static msvc runtime so the exe ships without vc redist, matching optcuts.
if(MSVC)
  set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(xatlas main.cpp xatlas.cpp)

if(MSVC)
  target_compile_options(xatlas PRIVATE $<$<CONFIG:Release>:/O2>)
else()
  target_compile_options(xatlas PRIVATE $<$<CONFIG:Release>:-O3>)
  # xatlas uses std::thread
  find_package(Threads REQUIRED)
  target_link_libraries(xatlas PRIVATE Threads::Threads)
endif()

# package the exe into a zip. the ci passes -DCPACK_GENERATOR=ZIP and
# -DCPACK_PACKAGE_FILE_NAME; bundle-engines.sh finds the binary by name.
install(TARGETS xatlas RUNTIME DESTINATION .)
set(CPACK_PACKAGE_NAME xatlas)
include(CPack)
