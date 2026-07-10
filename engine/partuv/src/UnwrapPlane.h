#ifndef UNWRAP_PLANE_H
#define UNWRAP_PLANE_H

#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Core>

// Include project headers for types and constants
#include "Component.h"    // Declaration of Component class.
#include "Mesh.h"         // Declaration of mesh types (e.g., MatrixX3R).
#include "UnwrapBB.h"     // For functions like prepareOBBData, findConnectedComponent, etc.
#include "Distortion.h"   // For DISTORTION_THRESHOLD and distortion functions.


int process_submesh(std::vector<int> faces, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, int id, Component &comp, bool check_overlap, cudaStream_t stream = nullptr);
 
std::vector<Component> unwrap_aligning_plane(const Eigen::MatrixXd &V,   const  Eigen::MatrixXi &F, double threshold, bool check_overlap = false, int chart_limit= (double) std::numeric_limits<int>::min());


#endif // UNWRAP_MERGE_H
