
#ifndef UNWRAPONE_H
#define UNWRAPONE_H

// Standard and third-party includes
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <map>
#include <utility>

#include "Mesh.h"       // For MatrixX3R and any mesh-related types.
#include "Component.h"  // For the Component class.


/**
 * @brief Unwraps a single-piece mesh directly.
 *
 *
 * @param V Input/Output Eigen::MatrixXd of vertex positions. The function may adjust these positions
 *          during the unwrapping process.
 * @param F Input Eigen::MatrixXi of face indices describing the mesh topology.
 * @return std::vector<Component> A vector of mesh components (charts) with computed UV parameterizations.
 */
std::vector<Component> unwrap_aligning_one(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold = 1,  bool check_overlap = false, int chart_limit= (double) std::numeric_limits<int>::min());


#endif // UNWRAPBB_H
