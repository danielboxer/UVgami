#pragma once

#include <cmath>

#include <Eigen/SparseLU>

#include "OpenABF/Exceptions.hpp"
#include "OpenABF/HalfEdgeMesh.hpp"
#include "OpenABF/Math.hpp"
#include <easy/profiler.h>
#include <omp.h>  


namespace OpenABF
{

namespace traits
{
/** @brief ABF and ABFPlusPlus vertex traits */
template <typename T>
struct ABFVertexTraits : public DefaultVertexTraits<T> {
    /** Lagrange Multiplier: Planarity constraint */
    T lambda_plan{0};
    /** Lagrange Multiplier: Reconstruction constraint */
    T lambda_len{1};
};

/** @brief ABF and ABFPlusPlus edge traits */
template <typename T>
struct ABFEdgeTraits : public DefaultEdgeTraits<T> {
    /** 3D angle */
    T beta{0};
    /** Optimal (i.e. target) angle */
    T phi{0};
    /** Angle weight */
    T weight{0};
    /** Sin of alpha, because it's used a lot */
    T alpha_sin{0};
    /** Cos of alpha, because it's used a lot */
    T alpha_cos{0};
};

/** @brief ABF and ABFPlusPlus face traits */
template <typename T>
struct ABFFaceTraits : public DefaultFaceTraits<T> {
    /** Lagrange Multiplier: Triangle validity constraint */
    T lambda_tri{0};
};
}  // namespace traits

namespace detail
{

/** @brief %ABF and ABF++ implementation details */
namespace ABF
{

/** @brief A HalfEdgeMesh with the %ABF traits */
template <typename T>
using Mesh = HalfEdgeMesh<
    T,
    3,
    traits::ABFVertexTraits<T>,
    traits::ABFEdgeTraits<T>,
    traits::ABFFaceTraits<T>>;

/** @brief Initialize the %ABF angles and weights from the edge alpha values */
template <typename T, class MeshPtr>
void InitializeAnglesAndWeights(MeshPtr& m)
{
    // Initialize and bound angle properties
    static constexpr auto MinAngle = PI<T> / T(180);
    static constexpr auto MaxAngle = PI<T> - MinAngle;
    for (auto& e : m->edges()) {
        e->alpha = e->beta = e->phi =
            std::min(std::max(e->alpha, MinAngle), MaxAngle);
        e->alpha_sin = std::sin(e->alpha);
        e->alpha_cos = std::cos(e->alpha);
        e->weight = T(1) / (e->phi * e->phi);
    }

    // Update weights for interior vertices
    for (auto& v : m->vertices_interior()) {
        auto wheel = v->wheel();
        auto angle_sum = std::accumulate(
            wheel.begin(), wheel.end(), T(0),
            [](auto a, auto b) { return a + b->beta; });
        for (auto& e : wheel) {
            e->phi *= 2 * PI<T> / angle_sum;
            e->weight = T(1) / (e->phi * e->phi);
        }
    }
}

/** @brief Compute ∇CTri w.r.t LambdaTri == CTri */
template <
    typename T,
    class FacePtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto TriGrad(const FacePtr& f, bool force_recompute = false) -> T
{
    if(f->tri_grad_cache_valid && !force_recompute) {
        return f->tri_grad_cache;
    }
    T g = -PI<T>;
    for (const auto& e : *f) {
        g += e->alpha;
    }
    f->tri_grad_cache = g;
    f->tri_grad_cache_valid = true;
    return g;
}

/** @brief Compute ∇CPlan w.r.t LambdaPlan == CPlan */
template <
    typename T,
    class VertPtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto PlanGrad(const VertPtr& v, bool force_recompute = false) -> T
{
    if(v->plan_grad_cache_valid && !force_recompute) {
        return v->plan_grad_cache;
    }
    // EASY_FUNCTION("PlanGrad");
    auto edges = v->wheel();
    T g = -2 * PI<T>;
    for (const auto& e : v->wheel()) {
        g += e->alpha;
    }
    v->plan_grad_cache = g;
    v->plan_grad_cache_valid = true;
    return g;
}

/** @brief Compute ∇CLen w.r.t LambdaLen == CLen */
template <
    typename T,
    class VertPtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto LenGrad(const VertPtr& vertex, bool force_recompute = false) -> T
{
    // EASY_FUNCTION("LenGrad 1");
    if(vertex->p1_cache_valid && vertex->p2_cache_valid && !force_recompute) {
        return vertex->lengrad_cache;
    }
    T p1{1};
    T p2{1};
    for (const auto& e : vertex->wheel()) {
        p1 *= e->next->alpha_sin;
        p2 *= e->next->next->alpha_sin;
    }
    vertex->p1_cache = p1;
    vertex->p2_cache = p2;
    vertex->p1_cache_valid = true;
    vertex->p2_cache_valid = true;
    vertex->lengrad_cache = p1 - p2;
    vertex->lengrad_cache_valid = true;
    return p1 - p2;
}


/** @brief Compute ∇CLen w.r.t edge->alpha */
template <
    typename T,
    class VertPtr,
    class EdgePtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto LenGrad(const VertPtr& vertex, const EdgePtr& edge) -> T
{
    // EASY_BLOCK("LenGrad 2");

    T p1{1};
    T p2{1};
    for (const auto& a : vertex->wheel()) {
        auto b = a->next;
        if (b == edge) {
            p1 *= b->alpha_cos;
            p2 = T(0);
        } else {
            p1 *= b->alpha_sin;
        }

        auto c = a->next->next;
        if (c == edge) {
            p1 = T(0);
            p2 *= c->alpha_cos;
        } else {
            p2 *= c->alpha_sin;
        }
    }
    // // EASY_END_BLOCK;
    return p1 - p2;
}

/** @brief Compute ∇CLen w.r.t edge->alpha */
template <
    typename T,
    class VertPtr,
    class EdgePtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto LenGrad_2(const VertPtr& vertex, const EdgePtr& edge, bool verbose = false) -> T
{
//    EASY_BLOCK("LenGrad 2");

    if(!vertex->lengrad_cache_valid) {
        LenGrad<T>(vertex);
    }
    T p1 = vertex->p1_cache;
    T p2 = vertex->p2_cache;

    for (const auto& a : vertex->wheel()) {
    //    EASY_BLOCK("LenGrad 2.1");
        auto b = a->next;
        if (b == edge) {
            p1 = p1 * b->alpha_cos / b->alpha_sin;
            p2 = T(0);
            if(verbose) std::cout << "b == edge" << std::endl;
            return p1 - p2;
        } 
        // else {
        //     p1 *= b->alpha_sin;
        // }
        auto c = a->next->next;
        if (c == edge) {
            p1 = T(0);
            p2 = p2* c->alpha_cos  / c->alpha_sin ;
            if(verbose) std::cout << "c == edge" << std::endl;
            return p1 - p2;
        } 
        // else {
        //     p2 *= c->alpha_sin;
        // }
    //    EASY_END_BLOCK;
    }
    // // EASY_END_BLOCK;
    // return p1 - p2;
    // T p1{1}, p2{1};
    // for (auto* a : vertex->wheel_raw()) {
    //     auto* b = a->next.get();
    //     if (b == edge) {
    //         p1 *= b->alpha_cos;  p2 = 0;
    //     } else {
    //         p1 *= b->alpha_sin;
    //     }

    //     auto* c = b->next.get();
    //     if (c == edge) {
    //         p1 = 0;              p2 *= c->alpha_cos;
    //     } else {
    //         p2 *= c->alpha_sin;
    //     }
    // }
    // return p1 - p2;
}





/** @brief Compute ∇F w.r.t an edge's alpha */
// template <
//     typename T,
//     class EdgePtr,
//     std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
// auto AlphaGrad(const EdgePtr& edge) -> T
// {
//     // δE/δα
//     EASY_BLOCK("AlphaGrad");
//     auto g = (edge->alpha - edge->phi) * edge->weight;
//     // δCTri/δα
//     g += edge->face->lambda_tri;
//     for (const auto& e : *edge->face) {
//         // Skip boundary vertices
//         if (e->vertex->is_boundary()) {
//             continue;
//         }
//         if (e == edge) {
//             // δCPlan/δα
//             g += e->vertex->lambda_plan;
//         } else {
//             // δCLen/δα
//             auto d = LenGrad<T>(e->vertex, edge);
//             d *= e->vertex->lambda_len;
//             g += d;
//         }
//     }
//     // EASY_END_BLOCK;
//     return g;
// }

template <
    typename T,
    class EdgePtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto AlphaGrad(const EdgePtr& edge, bool force_recompute = false) -> T
{
//    EASY_FUNCTION(profiler::colors::LightBlue);
    // // EASY_BLOCK("AlphaGrad");                 // whole function

    // T g{0};

    // // ---------- δE/δα ----------
    // // EASY_BLOCK("deltaE");
    // g = (edge->alpha - edge->phi) * edge->weight;
    // // EASY_END_BLOCK;

    // // ---------- δCTri/δα ----------
    // // EASY_BLOCK("deltaCTri");
    // g += edge->face->lambda_tri;
    // // EASY_END_BLOCK;

    // // ---------- per-edge loop ----------
    // for (const auto& e : *edge->face) {

    //     // EASY_BLOCK("is_boundary");
    //     if (e->vertex->is_boundary()) {
    //         // EASY_END_BLOCK;                 // close is_boundary
    //         continue;
    //     }
    //     // EASY_END_BLOCK;                     // close is_boundary

    //     if (e == edge) {
    //         // ---------- δCPlan/δα ----------
    //         // EASY_BLOCK("deltaCPlan");
    //         g += e->vertex->lambda_plan;
    //         // EASY_END_BLOCK;
    //     } else {
    //         // ---------- δCLen/δα ----------
    //         // EASY_BLOCK("deltaCLen");
    //         auto d = LenGrad<T>(e->vertex, edge);
    //         d *= e->vertex->lambda_len;
    //         g += d;
    //         // EASY_END_BLOCK;
    //     }
    // }

    // // EASY_END_BLOCK;                          // close "AlphaGrad"
    // return g;
    if(edge->alpha_grad_cache_valid && !force_recompute) {
        return edge->alpha_grad_cache;
    }

    T g = (edge->alpha - edge->phi) * edge->weight
    + edge->face->lambda_tri;


    auto curr_e = edge;
    // // g += curr_e->vertex->lambda_plan;


    for (int i = 0; i < 3; ++i) {
        if( curr_e->vertex->is_boundary())  {
            curr_e = curr_e->next;
            continue;
        }
        if (curr_e == edge) {
            // δCPlan/δα
            g += curr_e->vertex->lambda_plan;
        } else {
            // δCLen/δα
            // std::cout << "LenGrad: " << i << std::endl;
            // auto d = LenGrad<T>(curr_e->vertex, edge);
            if(!curr_e->vertex->p1_cache_valid || !curr_e->vertex->p2_cache_valid) {
                std::cout << "LenGrad: " << i << std::endl;
                LenGrad<T>(curr_e->vertex);
            }
            auto d = T(0);
            if( i == 1) { // c == edge
                // p1 = T(0);
                // p2 = p2  * edge->alpha_cos / edge->alpha_sin;
                d = -curr_e->vertex->p2_cache * edge->alpha_cos / edge->alpha_sin;
            }else if( i == 2) { // b == edge
                // p1 = p1  * edge->alpha_cos / edge->alpha_sin;
                // p2 = T(0);
                d = curr_e->vertex->p1_cache * edge->alpha_cos / edge->alpha_sin;
            }else {
                std::cout << "i == 0, this should not happen" << std::endl;
                // auto d = LenGrad<T>(curr_e->vertex, edge);
            }
            // d = p1 - p2;
            // auto d2 = LenGrad_2<T>(curr_e->vertex, edge);
            // std::cout << "d: " << d << " d2: " << d2 << std::endl;
            // if(std::abs(d - d2) > 1e-6) {
            //     std::cout << "i: " << i << std::endl;
            //     LenGrad_2<T>(curr_e->vertex, edge, true);
            //     std::cout << "d: " << d << " d2: " << d2 <<  std::endl;
            // }
            d *= curr_e->vertex->lambda_len;
            g += d;
        }
        curr_e = curr_e->next;
    }


    // for (auto* e : edge->face->raw_edges()) {
    // //    EASY_BLOCK("is_boundary");

    //     if( e->vertex->is_boundary())  {
    //     //    EASY_END_BLOCK;
    //         continue;
    //     }
    // //    EASY_END_BLOCK;
    //     if (e == edge.get()) {
    //         g += e->vertex->lambda_plan;
    //     } else {
    //         // T d = LenGrad<T>(e->vertex.get(), edge.get());   // pass raw*
    //         // std::cout << "LenGrad<T>(e->vertex, edge)" << std::endl;
    //     //    EASY_BLOCK("deltaCLen");
    //         // auto d = LenGrad<T>(e->vertex, edge);
    //         auto d = LenGrad_2<T>(e->vertex, edge);
    //         // std::cout << "d: " << d << " d2: " << d2 << std::endl;
    //     //    EASY_END_BLOCK;
    //         g += d * e->vertex->lambda_len;
            
    //     }
    // // }
    edge->alpha_grad_cache = g;
    edge->alpha_grad_cache_valid = true;
    return g;
}


/** @brief Compute ∇F w.r.t all parameters */
template <
    typename T,
    class MeshPtr,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto Gradient(const MeshPtr& mesh) -> T
{
    // EASY_BLOCK("ABF Gradient");
    
    T g{0};
    for (const auto& v : mesh->vertices_interior()) {
        auto gPlan = PlanGrad<T>(v, true);
        g += gPlan * gPlan;

        auto gLen = LenGrad<T>(v, true);
        g += gLen * gLen;
    }
    for (const auto& f : mesh->faces()) {
        // AlphaGrad for all edges
        for (const auto& e : *f) {
            auto gAlpha = AlphaGrad<T>(e, true);
            g += gAlpha * gAlpha;
        }
        // TriGrad for all faces
        auto gTri = TriGrad<T>(f, true);
        g += gTri * gTri;
    }
    // EASY_END_BLOCK;
    // EASY_BLOCK("ABF Gradient: Interior Vertices");
    // PlanGrad and LenGrad for all interior vertices

    // EASY_END_BLOCK;
    return g;
}
}  // namespace ABF
}  // namespace detail

/**
 * @brief Compute parameterized interior angles using Angle-based flattening
 *
 * Iteratively computes a new set of interior angles which minimize the total
 * angular error of the parameterized mesh. This follows the standard
 * angled-based flattening formulation, which directly solves for the objective
 * functions and constraints. ABFPlusPlus is generally preferred as it
 * dramatically simplifies the size of the solved problem without introducing
 * more error.
 *
 * This class **does not** compute a parameterized mesh. Rather, it calculates
 * the optimal interior angles for such a mesh. To convert this information
 * into a full parameterization, pass the processed HalfEdgeMesh to
 * AngleBasedLSCM.
 *
 * Implements "Parameterization of faceted surfaces for meshing using
 * angle-based flattening" by Sheffer and de Sturler (2001)
 * \cite sheffer2001abf.
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
        Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>>,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
class ABF
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

    /** @copydoc ABF::Compute */
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

        // Initialize angles and weights
        InitializeAnglesAndWeights<T>(mesh);

        // while ||∇F(x)|| > ε
        gradient = Gradient<T>(mesh);
        if (std::isnan(gradient) or std::isinf(gradient)) {
            throw MeshException("Mesh gradient cannot be computed");
        }
        auto gradDelta = INF<T>;
        iters = 0;
        while (gradient > 0.001 and gradDelta > 0.001 and iters < maxIters) {
            if (std::isnan(gradient) or std::isinf(gradient)) {
                throw MeshException("Mesh gradient cannot be computed");
            }
            // Typedefs
            using Triplet = Eigen::Triplet<T>;
            using SparseMatrix = Eigen::SparseMatrix<T>;
            using DenseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

            // Helpful parameters
            auto vCnt = mesh->num_vertices();
            auto vIntCnt = mesh->num_vertices_interior();
            auto edgeCnt = mesh->num_edges();
            auto faceCnt = mesh->num_faces();

            //// RHS ////
            // b1 = -alpha gradient
            std::vector<Triplet> triplets;
            std::size_t idx{0};
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, 0, -AlphaGrad<T>(e));
                ++idx;
            }

            // b2 = -lambda gradient
            // lambda tri
            for (const auto& f : mesh->faces()) {
                triplets.emplace_back(idx, 0, -TriGrad<T>(f));
                ++idx;
            }
            // lambda plan and lambda len
            for (const auto& v : mesh->vertices_interior()) {
                triplets.emplace_back(idx, 0, -PlanGrad<T>(v));
                triplets.emplace_back(vIntCnt + idx, 0, -LenGrad<T>(v));
                ++idx;
            }
            SparseMatrix b(edgeCnt + faceCnt + 2 * vIntCnt, 1);
            b.reserve(triplets.size());
            b.setFromTriplets(triplets.begin(), triplets.end());

            // vertex idx -> interior vertex idx permutation
            std::map<std::size_t, std::size_t> vIdx2vIntIdx;
            std::size_t newIdx{0};
            for (const auto& v : mesh->vertices_interior()) {
                vIdx2vIntIdx[v->idx] = newIdx++;
            }

            ///// LHS /////
            // Lambda = diag(2/w)
            // v.weight == 1/w, so Lambda is diag(2*weight)
            // We only need Lambda Inverse, so this is 1 / 2*weight
            triplets.clear();
            idx = 0;
            for (const auto& e : mesh->edges()) {
                triplets.emplace_back(idx, idx, 2 * e->weight);
                ++idx;
            }

            // J
            // Jacobian of the CTri constraints
            for (idx = 0; idx < faceCnt; idx++) {
                auto row = idx + edgeCnt;
                auto col = 3 * idx;
                triplets.emplace_back(row, col, 1);
                triplets.emplace_back(row, col + 1, 1);
                triplets.emplace_back(row, col + 2, 1);

                // Jt
                triplets.emplace_back(col, row, 1);
                triplets.emplace_back(col + 1, row, 1);
                triplets.emplace_back(col + 2, row, 1);
            }
            for (const auto& v : mesh->vertices_interior()) {
                auto row = idx + edgeCnt;
                for (const auto& e0 : v->wheel()) {
                    // Jacobian of the CPlan constraint
                    triplets.emplace_back(row, e0->idx, 1);
                    triplets.emplace_back(e0->idx, row, 1);

                    // Jacobian of the CLen constraint
                    auto e1 = e0->next;
                    auto e2 = e1->next;
                    auto d1 = LenGrad<T>(v, e1);
                    auto d2 = LenGrad<T>(v, e2);
                    triplets.emplace_back(vIntCnt + row, e1->idx, d1);
                    triplets.emplace_back(vIntCnt + row, e2->idx, d2);
                    triplets.emplace_back(e1->idx, vIntCnt + row, d1);
                    triplets.emplace_back(e2->idx, vIntCnt + row, d2);
                }
                ++idx;
            }
            auto Asize = edgeCnt + faceCnt + 2 * vIntCnt;
            SparseMatrix A(Asize, Asize);
            A.reserve(triplets.size());
            A.setFromTriplets(triplets.begin(), triplets.end());

            A.makeCompressed();
            Solver solver;
            solver.compute(A);
            if (solver.info() != Eigen::ComputationInfo::Success) {
                throw SolverException(solver.lastErrorMessage());
            }
            DenseVector delta = solver.solve(b);
            if (solver.info() != Eigen::ComputationInfo::Success) {
                throw SolverException(solver.lastErrorMessage());
            }

            // alpha += delta_alpha
            // Update sin and cos
            idx = 0;
            for (auto& e : mesh->edges()) {
                e->alpha += delta(idx++, 0);
                e->alpha = std::min(std::max(e->alpha, T(0)), PI<T>);
                e->alpha_sin = std::sin(e->alpha);
                e->alpha_cos = std::cos(e->alpha);
            }

            // lambda += delta_lambda
            for (auto& f : mesh->faces()) {
                f->lambda_tri += delta(idx++, 0);
            }
            for (auto& v : mesh->vertices_interior()) {
                auto intIdx = vIdx2vIntIdx.at(v->idx);
                v->lambda_plan += delta(idx + intIdx, 0);
                v->lambda_len += delta(idx + vIntCnt + intIdx, 0);
                idx++;
            }

            // Recalculate gradient for next iteration
            auto newGrad = detail::ABF::Gradient<T>(mesh);
            gradDelta = std::abs(newGrad - gradient);
            gradient = newGrad;
            iters++;
        }
    }

    /** @copydoc ABF::Compute */
    static void Compute(typename Mesh::Pointer& mesh)
    {
        std::size_t iters{0};
        T grad{0};
        Compute(mesh, iters, grad);
    }

protected:
    /** Gradient */
    T grad_{0};
    /** Number of executed iterations */
    std::size_t iters_{0};
    /** Max iterations */
    std::size_t maxIters_{10};
};

}  // namespace OpenABF