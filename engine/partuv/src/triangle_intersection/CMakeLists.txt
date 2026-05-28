# lscm-src/CMakeLists.txt

# Note: We don't need to redefine the basic project settings since they're inherited
# from the parent CMakeLists.txt

# Define LSCM source files

# We don't need cmake_minimum_required or project() calls here since they're handled by the root CMakeLists.txt

# Define the source files for the triangle intersection library
find_package(easy_profiler REQUIRED
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/easy_profiler/lib/cmake/easy_profiler"
)  # STEP 1 #########################


set(TRIANGLE_SOURCES
    triangleHelper.cpp
)

# Create a static library instead of a Python module
add_library(triangle_intersection STATIC ${TRIANGLE_SOURCES})


# Add include directories
target_include_directories(triangle_intersection
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # For header files in this directory
    PRIVATE
        ${CGAL_INCLUDE_DIRS}        # CGAL headers are only needed internally
)

# Link against required libraries
target_link_libraries(triangle_intersection
    PUBLIC
        CGAL                        # CGAL is needed by users of this library
    PRIVATE
        OpenMP::OpenMP_CXX         # OpenMP is an implementation detail
        easy_profiler
)