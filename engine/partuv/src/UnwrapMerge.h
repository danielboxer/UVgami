#ifndef UNWRAP_MERGE_H
#define UNWRAP_MERGE_H

#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Core>

// Include project headers for types and constants
#include "Component.h"    // Declaration of Component class.
#include "Mesh.h"         // Declaration of mesh types (e.g., MatrixX3R).
#include "UnwrapBB.h"     // For functions like prepareOBBData, findConnectedComponent, etc.
#include "Distortion.h"   // For DISTORTION_THRESHOLD and distortion functions.


#include "IO.h"

/**
 * @brief Merges Component B into Component A with distortion and overlap checks.
 *
 * This function attempts to merge two components (charts) by first merging the meshes,
 * then reparameterizing via LSCM, and finally checking for overlaps and excessive distortion.
 * If the merge fails (due to overlap, distortion, or insufficient common vertices), it returns -1.
 *
 * @param A [in,out] Reference to Component A that will receive Component B.
 * @param B [in]     Component B to be merged into Component A.
 * @param distortion_threshold Threshold for acceptable distortion (defaults to DISTORTION_THRESHOLD).
 * @return int The index of Component A on success, or -1 if the merge fails.
 */
int merge_B_to_A(Component &A, const Component &B, double distortion_threshold = DISTORTION_THRESHOLD);

/**
 * @brief Cleans up neighbor references in the component adjacency list after a merge.
 *
 * After merging a standby component into another, this function replaces all references
 * to the merged (standby) component's index with the successful component index.
 *
 * @param compAdjList [in,out] The adjacency list for the components (each row is a list of (neighbor, weight) pairs).
 * @param standbyComp [in] The component that has been merged.
 * @param success_to_index [in] The index of the component into which standbyComp was merged.
 */
void clean_up_neighbors(std::vector<std::vector<std::pair<int,int>>> &compAdjList,
                        Component &standbyComp,
                        int success_to_index);

/**
 * @brief Reassigns a face to a different component based on its adjacent faces.
 *
 * For a given face, this function examines the neighboring faces (via faceAdj) to determine if
 * the face should be reassigned from its current component to a neighboring one. The decision is
 * based on the count of adjacent faces belonging to a single alternative component.
 *
 * @param f [in] The face index.
 * @param faceAdj [in] A vector (per face) of adjacent face indices.
 * @param faceToComp [in,out] A mapping from face indices to their current component indices.
 * @param components [in,out] A vector of components, where each component is represented as a vector of face indices.
 * @return int Returns the face index of a neighbor (if reassignment occurs) or -1 otherwise.
 */
int reassignFace(int f,
                 const std::vector<std::vector<int>> &faceAdj,
                 std::vector<int> &faceToComp,
                 std::vector<std::vector<int>> &components,
                 Tree* tree=nullptr);

/**
 * @brief Reassigns faces that have at least two common neighboring faces in exactly one other component.
 *
 * This function processes all faces (grouped by their components) and reassigns them when at least two
 * neighbors belong to a single alternative component.
 *
 * @param components [in,out] A vector of components, where each component is a vector of face indices.
 * @param faceAdj [in] A vector (per face) of adjacent face indices.
 */
void smoothComponentEdge(std::vector<std::vector<int>>& components,
                                     const std::vector<std::vector<int>>& faceAdj,
                                     Tree* tree=nullptr);

/**
 * @brief Unwraps and merges mesh components by aligning and merging charts.
 *
 * This function segments the input mesh into connected components based on their oriented bounding
 * box (OBB) alignment, reassigns faces among adjacent components (if necessary), builds an adjacency
 * graph of components, and then attempts to merge smaller components into larger ones based on
 * distortion and overlap checks.
 *
 * @param V [in,out] Mesh vertices (on input, the original positions; on output, may be reoriented).
 * @param F [in] Mesh faces (indices into V).
 * @return std::vector<Component> A vector of final (merged) components/charts.
 */
std::vector<Component> unwrap_aligning_merge(const Eigen::MatrixXd &V,const Eigen::MatrixXi &F, double threshold, bool check_overlap = false, int chart_limit= (double) std::numeric_limits<int>::min());


std::vector<Component> merge_components(
    const std::vector<std::vector<int>> &components,
    const std::vector<std::vector<int>> faceAdj,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const MatrixX3R &FN,
    const int chart_limit,
    const bool check_overlap,
    const double threshold,
    const std::vector<std::vector<double>> &edge_lengths = {}
    );
 // namespace MeshLib

std::vector<Component> merge_components_parallel(
    const std::vector<std::vector<int>> &components,
    const std::vector<std::vector<int>> faceAdj,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const MatrixX3R &FN,
    const int chart_limit,
    const bool check_overlap,
    const double threshold,
    const std::vector<std::vector<double>> &edge_lengths = {}
);

#endif // UNWRAP_MERGE_H
