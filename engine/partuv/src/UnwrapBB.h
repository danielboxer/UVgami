#ifndef UNWRAPBB_H
#define UNWRAPBB_H

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



/** 
 * @brief Computes the principal component analysis (PCA) eigenvectors from a set of face normals.
 * 
 * @param data An F x 3 matrix where each row represents a face normal.
 * @return Eigen::Matrix3d The 3x3 matrix whose columns are the sorted eigenvectors.
 */
Eigen::Matrix3d computePCAEigenvectors(const Eigen::MatrixXd &data);

/**
 * @brief Computes the face adjacency for a triangular mesh.
 *
 * Given the faces F and vertices V of a mesh, this function returns a list where the i-th
 * element is a vector of indices of faces adjacent to face i.
 *
 * @param F An #Eigen::MatrixXi of size (num_faces x 3) representing face indices.
 * @param V An #Eigen::MatrixXd of size (num_vertices x 3) representing vertex coordinates.
 * @return std::vector<std::vector<int>> A vector of face adjacency lists.
 */
std::vector<std::vector<int>> computeFaceAdjacency(const Eigen::MatrixXi &F,
                                                    const Eigen::MatrixXd &V,
                                                    std::vector<std::vector<double>> &edge_lengths);

/**
 * @brief Finds the most matched neighbor component based on matching weights.
 *
 * Given a component index and its list of adjacent components (with associated weights),
 * this function returns the index of the neighbor with the highest weight.
 *
 * @param compIndex The index of the component to check.
 * @param compAdjList The adjacency list where each entry is a list of (neighbor index, weight) pairs.
 * @return int The index of the most matched neighbor component.
 */
int findMostMatchedNeighbor(int compIndex,
                            const std::vector<std::vector<std::pair<int, int>>> &compAdjList);

/**
 * @brief Computes dot products between face normals and 6 oriented bounding box (OBB) directions.
 *
 * @param faceNormals A matrix (F x 3) containing the face normals.
 * @param obbDirections A 6x3 matrix (row-major) representing the OBB directions.
 * @return Eigen::MatrixXd A matrix of size (F x 6) containing the dot products.
 */
Eigen::MatrixXd computeDotProducts(const MatrixX3R &faceNormals,
                                   const Eigen::Matrix<double, 6, 3, Eigen::RowMajor> &obbDirections);

/**
 * @brief Prepares mesh data for oriented bounding box (OBB) alignment.
 *
 * This function computes face normals, reorients the vertex positions using PCA,
 * assigns each face to one of 6 OBB directions, and computes face adjacency.
 *
 * @param V Input/Output #Eigen::MatrixXd of vertex positions. On output, V is reoriented.
 * @param F Input #Eigen::MatrixXi of face indices.
 * @param FN Output face normals (each row corresponds to a face normal).
 * @param faceAssignment Output vector assigning each face to an OBB direction (0–5).
 * @param faceAdj Output face adjacency list.
 */
void prepareOBBData(Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    MatrixX3R &FN,
                    std::vector<int> &faceAssignment,
                    std::vector<std::vector<int>> &faceAdj,
                    std::vector<std::vector<double>> &edge_lengths);

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
std::vector<Component> unwrap_aligning_BB(const Eigen::MatrixXd &V,
                                           const Eigen::MatrixXi &F,
                                           double threshold,
                                           bool check_overlap = false,
                                           int chart_limit = std::numeric_limits<int>::min());


#endif // UNWRAPBB_H
