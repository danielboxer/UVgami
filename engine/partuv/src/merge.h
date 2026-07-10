#ifndef MESH_MERGE_H
#define MESH_MERGE_H

#include "UnwrapBB.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <stdexcept>

/**
 * @brief Custom hash function for Eigen::VectorXd to use in unordered_map.
 *        This implementation uses quantization to handle floating-point precision.
 */
struct VectorHash
{
    std::size_t operator()(const Eigen::VectorXd& vec) const
    {
        std::size_t seed = vec.size();
        for (int i = 0; i < vec.size(); ++i)
        {
            // Quantize the vector components to handle floating-point precision
            // Adjust the factor (e.g., 1e6) based on the desired precision
            long long quantized = static_cast<long long>(std::round(vec[i] * 1e6));
            seed ^= std::hash<long long>()(quantized) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/**
 * @brief Equality comparison for Eigen::VectorXd with epsilon tolerance.
 */
struct VectorEqual
{
    bool operator()(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const
    {
        return a.isApprox(b, 1e-6); // Adjust epsilon as needed
    }
};

/**
 * @brief Merges mesh B into mesh A, producing a new merged mesh.
 *
 * This function performs the following steps:
 * 1. Initializes the merged mesh with all vertices and faces from mesh A.
 * 2. Iterates through each vertex in mesh B:
 *    - If the vertex exists in mesh A (within a specified epsilon), it reuses the existing index.
 *    - Otherwise, it appends the new vertex to the merged mesh.
 * 3. Updates the face indices from mesh B to correspond to the merged vertex indices.
 * 4. Appends the updated faces from mesh B to the merged mesh.
 *
 * @param A            The first mesh component (const reference).
 * @param B            The second mesh component to merge into A (const reference).
 * @param merge_result The resulting merged mesh component (output parameter).
 * @return int         The number of newly added vertices from mesh B.
 *
 * @throws std::invalid_argument if the dimensions of A and B do not match.
 * @throws std::out_of_range     if any face index in mesh B is invalid.
 */
int merge_mesh_B_to_A(const Component &A, const Component &B, Component &merge_result, bool use_all_faces_A = false);

#endif // MESH_MERGE_H
