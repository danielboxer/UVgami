# lscm-src/CMakeLists.txt

# Note: We don't need to redefine the basic project settings since they're inherited
# from the parent CMakeLists.txt

# Define LSCM source files
set(LSCM_SOURCES
    src/Boundary.cpp
    src/Edge.cpp
    src/Face.cpp
    src/FormTrait.cpp
    src/LSCM.cpp
    src/HalfEdge.cpp
    src/Homology.cpp
    src/Mesh.cpp
    src/Point.cpp
    src/Vertex.cpp
    src/Graph.cpp
)

# Create the LSCM library
add_library(lscm_lib STATIC ${LSCM_SOURCES})

# Add LSCM-specific include directories
target_include_directories(lscm_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src  # LSCM's own headers
)

# Link necessary dependencies
target_link_libraries(lscm_lib
    PUBLIC
        # $<LINK_ONLY:MKL::MKL>
    PRIVATE
        OpenMP::OpenMP_CXX
)