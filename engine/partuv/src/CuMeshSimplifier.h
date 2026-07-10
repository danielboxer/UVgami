#pragma once
#include <Eigen/Core>
#include <utility>      // std::pair
#include <cuda_runtime.h>

/// Lightweight wrapper around the CUDA PaSP simplifier in cusimp.h.
/// All heavy work happens on the GPU; this function just prepares data,
/// runs the iterative loop and returns a simplified copy.
///
/// @param V         (#V × 3) float vertices
/// @param F         (#F × 3) int   triangle indices
/// @param threshold edge‑collapse threshold (same meaning as Python version)
/// @param max_iter  maximum outer iterations
///
/// @return {V_out, F_out}  simplified mesh (not in‑place)
///
namespace CuMeshSimplifier
{
std::pair<Eigen::MatrixXf, Eigen::MatrixXi>
simplify(const Eigen::MatrixXf& V,
         const Eigen::MatrixXi& F,
         float  threshold = 1e-2f,
         int    max_iter  = 1000,
         cudaStream_t stream = nullptr);


         
} // namespace CuMeshSimplifier
