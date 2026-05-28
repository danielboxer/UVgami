#include "merge.h"

/**
 * @brief Merges mesh B into mesh A, producing a new merged mesh.
 *
 * Implementation of the merge_mesh_B_to_A function declared in MeshMerge.h.
 */
int merge_mesh_B_to_A(const Component &A, const Component &B, Component &merge_result, bool use_all_faces_A )
{
    using namespace Eigen;


    // Check if A and B have the same dimensionality
    if (A.V.cols() != B.V.cols())
    {
        throw std::invalid_argument("Mesh dimensions do not match.");
    }

    // Initialize merge_result with A's vertices and faces
    merge_result.V = A.V;
    merge_result.F = A.F;

    // Number of dimensions (e.g., 3 for 3D meshes)
    const int dim = static_cast<int>(A.V.cols());

    // Create a hash map for A's vertices to speed up duplicate search
    std::unordered_map<Eigen::VectorXd, int, VectorHash, VectorEqual> vertex_map;

    int num_A_faces = use_all_faces_A ? A.V.rows() : A.original_vertex_count;
    for (int i = 0; i < num_A_faces; ++i)
    {
        Eigen::VectorXd v = A.V.row(i);
        vertex_map.emplace(v, i);
    }

    // Mapping from B's vertex index to merge_result's vertex index
    std::vector<int> b2merge_index_map(B.V.rows(), -1);

    int numNewVertices = 0;
    int num_B_faces =  B.original_vertex_count;



    

    // Iterate over B's vertices
    for (int i = 0; i < B.V.rows(); ++i)
    {

        Eigen::VectorXd vB = B.V.row(i);

        auto it = vertex_map.find(vB);
        if (it != vertex_map.end() && i < num_B_faces)
        {
            // Vertex already exists in A
            b2merge_index_map[i] = it->second;
        }
        else
        {
            // New vertex, add to merge_result
            merge_result.V.conservativeResize(merge_result.V.rows() + 1, dim);
            merge_result.V.row(merge_result.V.rows() - 1) = vB;
            b2merge_index_map[i] = merge_result.V.rows() - 1;

            // Add to the hash map
            vertex_map.emplace(vB, b2merge_index_map[i]);

            numNewVertices++;
        }

    }

    // Now, map B's faces to merge_result's vertex indices
    // Assuming faces are consistent (e.g., all triangles)
    MatrixXi mapped_FB = B.F;

    for (int f = 0; f < B.F.rows(); ++f)
    {
        for (int c = 0; c < B.F.cols(); ++c)
        {
            int oldIndex = B.F(f, c);
            if (oldIndex < 0 || oldIndex >= B.V.rows())
            {
                throw std::out_of_range("Face index out of bounds in mesh B.");
            }
            mapped_FB(f, c) = b2merge_index_map[oldIndex];
        }
    }

    // Append mapped_FB to merge_result.F
    // Resize merge_result.F to accommodate new faces
    if (B.F.rows() > 0)
    {
        merge_result.F.conservativeResize(merge_result.F.rows() + mapped_FB.rows(), merge_result.F.cols());
        merge_result.F.bottomRows(mapped_FB.rows()) = mapped_FB;
    }

    merge_result.faces.insert(merge_result.faces.end(), B.faces.begin(), B.faces.end());
    merge_result.F_original = merge_result.F;
    merge_result.V_original = merge_result.V;
    return numNewVertices;
}
