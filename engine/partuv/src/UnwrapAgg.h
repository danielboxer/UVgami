#ifndef UNWRAP_AGG_H
#define UNWRAP_AGG_H

// Standard and third-party includes
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <map>
#include <utility>

// Include headers for types used in the declarations.
// (Make sure that these headers properly declare the types such as Component and MatrixX3R.)
#include "Mesh.h"       // For MatrixX3R and any mesh-related types.
#include "Component.h"  // For the Component class.
#include "pipeline.h"



/**
 * @brief Unwraps the mesh by aligning it to its oriented bounding box (OBB).
 *
 * This function segments the mesh faces into connected components based on their
 * dominant OBB direction, merges leftover components into the best-matching charts,
 * and then parameterizes each chart via LSCM. The distortion for each chart is computed.
 *
 * @param V Input/Output #Eigen::MatrixXd of vertex positions.
 * @param F Input #Eigen::MatrixXi of face indices.
 * @return std::vector<Component> A vector of components (charts) with computed UV parameterizations.
 */

std::vector<Component> unwrap_aligning_Agglomerative(const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    double threshold,
    bool check_overlap = false,
    int chart_limit = std::numeric_limits<int>::min());

std::vector<Component> unwrap_aligning_Agg_helper(const Eigen::MatrixXd &V,
                                           const Eigen::MatrixXi &F,
                                           double threshold,
                                           bool check_overlap = false,
                                           int chart_limit = std::numeric_limits<int>::min(),
                                           int num_cluster=6);

std::vector<UVParts> unwrap_aligning_Agglomerative_all(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit, bool check_break=false);

std::vector<Component> unwrap_aligning_Agglomerative_merge(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit);



#endif // UNWRAPBB_H
