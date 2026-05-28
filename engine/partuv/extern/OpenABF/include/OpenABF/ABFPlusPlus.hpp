#pragma once

#include <cmath>

#include <Eigen/SparseLU>
// #include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>

#include "OpenABF/ABF.hpp"
#include "OpenABF/Exceptions.hpp"
#include "OpenABF/HalfEdgeMesh.hpp"
#include "OpenABF/Math.hpp"

#include "OpenABF/Overlap.hpp"

#include <easy/profiler.h>

namespace OpenABF
{


typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrixRM;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrixCM;

/**
 * @brief Compute parameterized interior angles using ABF++
 *
 * Iteratively computes a new set of interior angles which minimize the total
 * angular error of the parameterized mesh. This follows the ABF++ formulation,
 * which solves a 5x smaller system of equations than standard ABF at the
 * expense of more iterations.
 *
 * This class **does not** compute a parameterized mesh. Rather, it calculates
 * the optimal interior angles for such a mesh. To convert this information
 * into a full parameterization, pass the processed HalfEdgeMesh to
 * AngleBasedLSCM.
 *
 * Implements "ABF++: Fast and Robust Angle Based Flattening" by Sheffer
 * _et al._ (2005) \cite sheffer2005abf++.
 *
 * @tparam T Floating-point type
 * @tparam MeshType HalfEdgeMesh type which implements the ABF traits
 * @tparam Solver A solver implementing the
 * [Eigen Sparse solver
 * concept](https://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html)
 * and templated on Eigen::SparseMatrix<T>
 */
template <
    typename T,
    class MeshType = detail::ABF::Mesh<T>,
    class Solver =
        // Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>>,
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Lower, Eigen::AMDOrdering<int>>,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>


class ABFPlusPlus
{
public:
    /** @brief Mesh type alias */
    using Mesh = MeshType;

    /** @brief Set the maximum number of iterations */
    void setMaxIterations(std::size_t it) { maxIters_ = it; }

    /**
     * @brief Get the mesh gradient
     *
     * **Note:** Result is only valid after running compute().
     */
    auto gradient() const -> T { return grad_; }

    /**
     * @brief Get the number of iterations of the last computation
     *
     * **Note:** Result is only valid after running compute().
     */
    auto iterations() const -> std::size_t { return iters_; }

    /** @copydoc ABFPlusPlus::Compute */
    void compute(typename Mesh::Pointer& mesh)
    {
        Compute(mesh, iters_, grad_, maxIters_);
    }

    /**
     * @brief Compute parameterized interior angles
     *
     * @throws SolverException If matrix cannot be decomposed or if solver fails
     * to find a solution.
     * @throws MeshException If mesh gradient cannot be calculated.
     */
    static void Compute(
        typename Mesh::Pointer& mesh,
        std::size_t& iters,
        T& gradient,
        std::size_t maxIters = 10)
    {
        using namespace detail::ABF;
        Solver solver;
        auto f_start = std::chrono::high_resolution_clock::now();

        // Initialize angles and weights
        InitializeAnglesAndWeights<T>(mesh);

        // while ||∇F(x)|| > ε
        gradient = Gradient<T>(mesh);
        if (std::isnan(gradient) or std::isinf(gradient)) {
            throw MeshException("Mesh gradient cannot be computed");
        }
        auto gradDelta = INF<T>;
        iters = 0;

        // Helpful parameters
        auto vIntCnt = mesh->num_vertices_interior();
        auto edgeCnt = mesh->num_edges();
        auto faceCnt = mesh->num_faces();
        // std::cout << "face count: " << faceCnt << std::endl;
        double limit = faceCnt > 100? 1: 0.001;
        using Triplet = Eigen::Triplet<T>;
        using SparseMatrix = Eigen::SparseMatrix<T>;
        using DenseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        


        std::vector<Triplet> J_triplets;
        std::size_t idx{0};

        // Jacobian of the CTri constraints
        for (; idx < faceCnt; idx++) {
            J_triplets.emplace_back(idx, 3 * idx, 1);
            J_triplets.emplace_back(idx, 3 * idx + 1, 1);
            J_triplets.emplace_back(idx, 3 * idx + 2, 1);
        }

        for (const auto& v : mesh->vertices_interior()) {
            for (const auto& e0 : v->wheel()) {
                J_triplets.emplace_back(idx, e0->idx, 1);
            }
            ++idx;
        }

        while (gradient > limit and gradDelta > limit and iters < maxIters) {
            // std::cout << "Iteration: " << iters << " Gradient: " << gradient << std::endl;
            
            EASY_BLOCK("ABF_COMPUTE_ITERATION");
            EASY_BLOCK("init and Construct b1");

            if (std::isnan(gradient) or std::isinf(gradient)) {
                throw MeshException("Mesh gradient cannot be computed");
            }
            // Typedefs


            std::vector<Triplet> triplets;
            // auto numEdges = mesh->edges().size();
            // triplets.reserve( numEdges);

            idx = 0;
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, 0, -AlphaGrad<T>(e));
                ++idx;
            }
            // std::cout << "number of edges: " << idx << std::endl;
            Eigen::SparseMatrix<T, Eigen::RowMajor> b1(edgeCnt, 1);
            b1.reserve(triplets.size());
            b1.setFromTriplets(triplets.begin(), triplets.end());
            EASY_END_BLOCK;
        
            EASY_BLOCK("Construct b2");
            triplets.clear();
            idx = 0;
            // lambda tri
            for (const auto& f : mesh->faces()) {
                triplets.emplace_back(idx, 0, -TriGrad<T>(f));
                idx++;
            }
            // lambda plan and lambda len
            for (const auto& v : mesh->vertices_interior()) {
                triplets.emplace_back(idx, 0, -PlanGrad<T>(v));
                triplets.emplace_back(vIntCnt + idx, 0, -LenGrad<T>(v));
                idx++;
            }
            Eigen::SparseMatrix<T, Eigen::ColMajor> b2(faceCnt + 2 * vIntCnt, 1);
            b2.reserve(triplets.size());
            b2.setFromTriplets(triplets.begin(), triplets.end());
            
            // vertex idx -> interior vertex idx permutation
            std::map<std::size_t, std::size_t> vIdx2vIntIdx;
            std::size_t newIdx{0};
            for (const auto& v : mesh->vertices_interior()) {
                vIdx2vIntIdx[v->idx] = newIdx++;
            }
            
            EASY_END_BLOCK;
            EASY_BLOCK("Construct J");
            triplets.clear();
            
            // idx = 0;
            // // Jacobian of the CTri constraints
            // for (; idx < faceCnt; idx++) {
            //     triplets.emplace_back(idx, 3 * idx, 1);
            //     triplets.emplace_back(idx, 3 * idx + 1, 1);
            //     triplets.emplace_back(idx, 3 * idx + 2, 1);
            // }
            
            // std::cout << "J_triplets size: " << J_triplets.size() << std::endl;
            // for (const auto& triplet : J_triplets) {
            //     std::cout << "(" << triplet.row() << "," << triplet.col() << ") = " << triplet.value() << std::endl;
            // }

            triplets.insert(triplets.end(), J_triplets.begin(), J_triplets.end());
            idx = faceCnt;

            for (const auto& v : mesh->vertices_interior()) {
                if(!v->lengrad_cache_valid) {
                    LenGrad<T>(v);
                }
                T p1 = v->p1_cache;
                T p2 = v->p2_cache;
                for (const auto& e0 : v->wheel()) {
                    // Jacobian of the CPlan constraint
                    // triplets.emplace_back(idx, e0->idx, 1);
        
                    // Jacobian of the CLen constraint
                    auto e = e0->next;
                    auto d = p1 * e->alpha_cos / e->alpha_sin;
                    triplets.emplace_back(vIntCnt + idx, e->idx, d);

                    e = e->next;
                    // std::cout << "The first edge: " ;
                    // b == edge
                    // p2 = T(0);
                    // auto d1 = LenGrad_2<T>(v, e1, true);
                    // std::cout << "The Second edge: " ;
                    // c == edge
                    d = -p2 * e->alpha_cos / e->alpha_sin;
                    // auto d2 = LenGrad_2<T>(v, e2, true);
                    triplets.emplace_back(vIntCnt + idx, e->idx, d);
                }
                ++idx;
            }

            Eigen::SparseMatrix<T, Eigen::RowMajor> J(faceCnt + 2 * vIntCnt, 3 * faceCnt);
            // std::cout << "face count: " << faceCnt << " vIntCnt: " << vIntCnt <<  " Edge count: " << edgeCnt << std::endl;
            J.reserve(triplets.size());
            J.setFromTriplets(triplets.begin(), triplets.end());
            // auto f_end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> f_elapsed = f_end - f_start;
            EASY_END_BLOCK;
        
            // auto std::cout << "[Profile] Initialization elapsed time..." ...
        
            auto profile_start = std::chrono::high_resolution_clock::now();
        
            EASY_BLOCK("Construct LambdaInv");
            // Lambda = diag(2/w), so we only need Lambda^-1 = diag(1/(2*w))
            triplets.clear();
            idx = 0;
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, idx, T(1) / (2 * e->weight));
                ++idx;
            }
            Eigen::SparseMatrix<T, Eigen::ColMajor> LambdaInv(edgeCnt, edgeCnt);
            LambdaInv.reserve(edgeCnt);
            LambdaInv.setFromTriplets(triplets.begin(), triplets.end());
            EASY_END_BLOCK;
        
            EASY_BLOCK("Form A and b");
            // solve Eq. 16
            EASY_BLOCK("Form A and b pt 1");
            EASY_BLOCK("Construct bstar and JLiJt pt1");
            // Decompose the original expression into intermediate steps
            SparseMatrixRM LambdaInvb1 = LambdaInv * b1;
            // First, compute temp1 = J * LambdaInv
            SparseMatrixCM JLambdaInvb1 = J * LambdaInvb1;
            // Next, compute temp2 = temp1 * b1
            // Finally, subtract b2 to obtain bstar
            SparseMatrixCM bstar = JLambdaInvb1 - b2;
            EASY_END_BLOCK;
            EASY_BLOCK("Construct bstar and JLiJt pt2");

            SparseMatrixCM JT = J.transpose();  // J^T

            SparseMatrixRM JLiJt = J * LambdaInv * JT;

            
            EASY_END_BLOCK;
        
            // TODO: check if this is sane (all or just diag) (direct inv)
            SparseMatrixRM LambdaStarInv = JLiJt.block(0, 0, faceCnt, faceCnt);
            for (int k = 0; k < LambdaStarInv.outerSize(); ++k) {
                for (typename SparseMatrixRM::InnerIterator it(LambdaStarInv, k); it; ++it) {
                    it.valueRef() = 1.F / it.value();
                }
            }

            EASY_END_BLOCK;
            EASY_BLOCK("Form A and b pt 2");
            SparseMatrixRM Jstar = JLiJt.block(faceCnt, 0, 2 * vIntCnt, faceCnt);
            auto JstarT = JLiJt.block(0, faceCnt, faceCnt, 2 * vIntCnt);
            SparseMatrixRM Jstar2 = JLiJt.block(faceCnt, faceCnt, 2 * vIntCnt, 2 * vIntCnt);
            auto bstar1 = bstar.block(0, 0, faceCnt, 1);
            SparseMatrixRM bstar2 = bstar.block(faceCnt, 0, 2 * vIntCnt, 1);
            EASY_END_BLOCK;

            EASY_BLOCK("Form A and b pt 3");
            SparseMatrixCM LambdaStarInvCM(LambdaStarInv);
            SparseMatrixCM JstarT_RM(JstarT);
            SparseMatrixCM bstar1_RM(bstar1);
            EASY_BLOCK("comp_tempA");
            SparseMatrixCM starInvJsT = LambdaStarInv * JstarT_RM;
            EASY_END_BLOCK;
            
            EASY_BLOCK("comp_A_intermediate");
            SparseMatrixRM JsLstarInvJsT = Jstar * starInvJsT;
            EASY_END_BLOCK;
            
            EASY_BLOCK("comp_A");
            SparseMatrix A = JsLstarInvJsT - Jstar2;
            EASY_END_BLOCK;
            
            EASY_BLOCK("comp_tempB");
            SparseMatrixCM tempB = LambdaStarInv * bstar1_RM;
            EASY_END_BLOCK;
            
            EASY_BLOCK("comp_b_intermediate");
            SparseMatrixRM b_intermediate = Jstar * tempB;
            EASY_END_BLOCK;
            
            EASY_BLOCK("comp_b");
            SparseMatrix b = b_intermediate - bstar2;
            EASY_END_BLOCK;
            A.makeCompressed();
            EASY_END_BLOCK;
            EASY_END_BLOCK;
            
            // auto profile_end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> elapsed = profile_end - profile_start;
            // std::cout << "[Profile] Constructing Jacobian elapsed time..." ...
            
            // solver.compute(A);
            // if (solver.info() != ... ) ...
            
            // auto deltaLambda2 = solver.solve(b);
            // if (solver.info() != ... ) ...
            
            EASY_BLOCK("Factor and Solve");

            // auto section_start = std::chrono::high_resolution_clock::now();
            
            // std::cout << "A size: " << A.rows() << " x " << A.cols() << std::endl;
            // Print matrix in a row x col grid, including zeros
            // for (int i = 0; i < A.rows(); ++i)
            // {
            //     for (int j = 0; j < A.cols(); ++j)
            //     {
            //         // 'coeff(i, j)' returns the matrix entry (zero or nonzero)
            //         // We use 'std::setw' to align columns.
            //         std::cout << std::setw(6) << A.coeff(i, j) << " ";
            //     }
            //     std::cout << std::endl;
            // }
            
            // // print r
            // for (int i = 0; i < b.size(); ++i)
            // {
            //     std::cout << std::setw(6) << b.coeff(i, 0) << " ";
            // }
            solver.compute(A);
            
            if (solver.info() != Eigen::Success) {
                #ifdef VERBOSE
                std::cerr << "Error in factorization of AtA in ABF Compute: " << solver.info() << " at iteration " << iters << " : ";
                if (solver.info() == Eigen::NumericalIssue) {
                    std::cerr << "Numerical issue..." << std::endl;
                } else if (solver.info() == Eigen::InvalidInput) {
                    std::cerr << "Invalid input" << std::endl;
                } else if (solver.info() == Eigen::NoConvergence) {
                    std::cerr << "No convergence" << std::endl;
                } else {
                    std::cerr << "Unknown error" << std::endl;
                }
                #endif
                // std::cerr << "Matrix A shape: " << A.rows() << " x " << A.cols() << std::endl;
                // std::cerr << "Vector r size: " << b.size() << std::endl;
                throw SolverException("Error in factorization of AtA in ABF compute: get status " + std::to_string(solver.info()));

            }
            // std::cout << "AtA factorized" << std::endl;
            // std::cout << "b size: " << b.rows() << " x " << b.cols() << std::endl;
            auto deltaLambda2 = solver.solve(b);
            if (solver.info() != Eigen::Success) {
                // std::cerr << "Error in solving for x" << std::endl;
                throw SolverException("Error in solving for x ABF compute: get status " + std::to_string(solver.info()));
                // std::cerr << solver.info() << std::endl;
            }
        
            EASY_END_BLOCK;

            EASY_BLOCK("Compute delta lambda 1 and 2");
            // Compute Eq. 17 -> delta_lambda_1
            auto deltaLambda1 = LambdaStarInv * (bstar1 - JstarT * deltaLambda2);
        
            // Construct deltaLambda
            DenseVector deltaLambda(deltaLambda1.rows() + deltaLambda2.rows(), 1);
            deltaLambda << DenseVector(deltaLambda1), DenseVector(deltaLambda2);
        
            // Compute Eq. 10 -> delta_alpha
            DenseVector deltaAlpha = LambdaInv * (b1 - JT * deltaLambda);
            
            EASY_END_BLOCK;
            EASY_BLOCK("Update lambda and alpha");
            EASY_BLOCK("Update: Lambda");
            // lambda += delta_lambda
            for (auto& f : mesh->faces()) {
                f->lambda_tri += deltaLambda(f->idx, 0);
            }
            for (auto& v : mesh->vertices_interior()) {
                auto intIdx = vIdx2vIntIdx.at(v->idx);
                v->lambda_plan += deltaLambda(faceCnt + intIdx, 0);
                v->lambda_len  += deltaLambda(faceCnt + vIntCnt + intIdx, 0);
            }
            EASY_END_BLOCK;

            EASY_BLOCK("Update: Alpha and Trigonometric Values");
            idx = 0;
            for (auto& e : mesh->edges()) {
                e->alpha += deltaAlpha(idx++, 0);
                e->alpha = std::min(std::max(e->alpha, T(0)), PI<T>);
                e->alpha_sin = std::sin(e->alpha);
                e->alpha_cos = std::cos(e->alpha);
            }
            EASY_END_BLOCK;


            EASY_BLOCK("Update: Recompute Gradient");
            auto newGrad = Gradient<T>(mesh);
            gradDelta = std::abs(newGrad - gradient);
            gradient = newGrad;
            EASY_END_BLOCK;
            // std::cout << "Iteration: " << iters << " Gradient: " << gradient << std::endl;
            iters++;
            EASY_END_BLOCK;
            EASY_END_BLOCK;
        }
        
    }

    /** @brief Compute parameterized interior angles */
    static void Compute(typename Mesh::Pointer& mesh)
    {
        std::size_t iters{0};
        T grad{0};
        Compute(mesh, iters, grad);
    }

private:
    /** Gradient */
    T grad_{0};
    /** Number of executed iterations */
    std::size_t iters_{0};
    /** Max iterations */
    std::size_t maxIters_{10};
};

}  // namespace OpenABF