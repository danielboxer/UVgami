#pragma once

#include <cmath>
#include <map>

#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>


#include "OpenABF/Exceptions.hpp"
#include "OpenABF/HalfEdgeMesh.hpp"
#include "OpenABF/Math.hpp"

#include "OpenABF/ABF.hpp"

namespace OpenABF
{

template<typename Scalar>
Eigen::SparseMatrix<Scalar>
removeEmptyCols(const Eigen::SparseMatrix<Scalar>& M) {
    const int rows = M.rows();
    const int cols = M.cols();

    // 1) Identify non-empty columns
    std::vector<int> keepCols;
    keepCols.reserve(cols);
    for (int j = 0; j < cols; ++j) {
        bool nonEmpty = false;
        for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(M, j); it; ++it) {
            nonEmpty = true;
            break;
        }
        if (nonEmpty) keepCols.push_back(j);
    }

    // 2) Build a new sparse matrix with only those columns
    Eigen::SparseMatrix<Scalar> R(rows, static_cast<int>(keepCols.size()));
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(M.nonZeros());

    for (int newJ = 0; newJ < static_cast<int>(keepCols.size()); ++newJ) {
        int oldJ = keepCols[newJ];
        for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(M, oldJ); it; ++it) {
            triplets.emplace_back(it.row(), newJ, it.value());
        }
    }

    R.setFromTriplets(triplets.begin(), triplets.end());
    return R;
}

// Assuming a triplet type that supports row(), col(), and value() methods.
template<typename T, typename Triplet>
void printTripletStats(const std::vector<Triplet>& tripletsA, int numFaces, int numFree) {
    std::cout << "Triplet stats:" << std::endl;
    std::cout << "  Total triplets: " << tripletsA.size() << std::endl;
    std::cout << "  Expected matrix size: " << 2 * numFaces << " x " << 2 * numFree << std::endl;

    // Count unique rows and columns, and compute other statistics
    std::set<int> uniqueRows, uniqueCols;
    T minValue = std::numeric_limits<T>::max();
    T maxValue = std::numeric_limits<T>::lowest();
    T sumValues = 0;
    int zeroCount = 0;
    int maxRow = -1;
    int maxCol = -1;

    for (const auto& triplet : tripletsA) {
        uniqueRows.insert(triplet.row());
        uniqueCols.insert(triplet.col());
        maxRow = std::max(maxRow, triplet.row());
        maxCol = std::max(maxCol, triplet.col());
        minValue = std::min(minValue, triplet.value());
        maxValue = std::max(maxValue, triplet.value());
        sumValues += triplet.value();
        if (triplet.value() == 0)
            zeroCount++;
    }

    std::cout << "  Unique rows: " << uniqueRows.size() << std::endl;
    std::cout << "  Unique columns: " << uniqueCols.size() << std::endl;
    std::cout << "  Max row index: " << maxRow << std::endl;
    std::cout << "  Max column index: " << maxCol << std::endl;
    std::cout << "  Min value: " << minValue << std::endl;
    std::cout << "  Max value: " << maxValue << std::endl;
    std::cout << "  Average value: " 
              << (tripletsA.empty() ? 0 : sumValues / tripletsA.size()) << std::endl;
    std::cout << "  Zero values: " << zeroCount << std::endl;
}



/**
 * @brief Compute parameterized mesh using Angle-based LSCM
 *
 * Computes a least-squares conformal parameterization of a mesh. Unlike the
 * original LSCM algorithm, this class ignores the 3D vertex positions and
 * instead uses the angle associated with the mesh's edge trait
 * (MeshType::EdgeTraits::alpha) to calculate the initial per-triangle edge
 * lengths. Without previously modifying the angles of the provided mesh, this
 * class produces the same result as a vertex-based LSCM implementation.
 * However, by first processing the mesh with a parameterized angle optimizer,
 * such as ABFPlusPlus, the parameterization can be improved, sometimes
 * significantly.
 *
 * Implements the angle-based variant of "Least squares conformal maps for
 * automatic texture atlas generation" by Lévy _et al._ (2002)
 * \cite levy2002lscm.
 *
 * @tparam T Floating-point type
 * @tparam MeshType HalfEdgeMesh type which implements the default mesh traits
 * @tparam Solver A solver implementing the
 * [Eigen Sparse solver
 * concept](https://eigen.tuxfamily.org/dox-devel/group__TopicSparseSystems.html)
 * and templated on Eigen::SparseMatrix<T>
 */

template <
    typename T,
    class MeshType = HalfEdgeMesh<T>,
    class Solver =
        // Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>>,
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Lower, Eigen::AMDOrdering<int>>,

    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
class AngleBasedLSCM
{
public:
    /** @brief Mesh type alias */
    using Mesh = MeshType;
    using VertPtr = typename Mesh::VertPtr;   // note the required `typename`

    /** @copydoc AngleBasedLSCM::Compute */
    int compute(typename Mesh::Pointer& mesh) const { return Compute(mesh); }

    static int Compute(typename Mesh::Pointer& mesh,
        const std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>>& vertex_pin_edges = {})
    {
        std::unordered_map<std::size_t, std::size_t> vertex_map;
        return Compute(mesh, vertex_map, vertex_pin_edges);
    }

    static int Compute_FixBoundary(typename Mesh::Pointer& mesh,
    const std::unordered_map<std::size_t, std::pair<T,T>>& pinned_uvs = {})
    {
        std::unordered_map<std::size_t, std::size_t> vertex_map;
        return Compute(mesh, vertex_map, {}, pinned_uvs);
    }

    /**
     * @brief Compute the parameterized mesh
     *
     * @throws MeshException If pinned vertex is not on boundary.
     * @throws SolverException If matrix cannot be decomposed or if solver fails
     * to find a solution.
     */
    static int Compute(typename Mesh::Pointer& mesh, 
                        std::unordered_map<std::size_t, std::size_t>& vertex_map,
                        const std::unordered_map<std::size_t, std::pair<std::size_t,std::size_t>>& vertex_pin_edges = std::unordered_map<std::size_t, std::pair<std::size_t,std::size_t>>{},
                        const std::unordered_map<std::size_t, std::pair<T,T>>& pinned_uvs = {}) // vertex_index -> pair of pinned index
    {
        using Triplet = Eigen::Triplet<T>;
        using SparseMatrix = Eigen::SparseMatrix<T>;
        using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;



        
        

        
        auto verts = mesh->vertices_outer_boundary();

        std::vector<VertPtr> pinnedVerts;              // list of pinned vertices
        std::unordered_map<std::size_t,std::size_t> pinColOffset;     // vertex-idx → base-column (2*k)


        if (verts.size() < 2) {
            // throw MeshException("Not enough vertices to pin");
            std::cerr << "Not enough vertices to pin" << std::endl;
            return -1;
        }
        auto p0 = *verts.begin();
        auto p1 = p0;
        T maxDistSq = T(-1);
        for (auto itA = verts.begin(); itA != verts.end(); ++itA) {
            for (auto itB = std::next(itA); itB != verts.end(); ++itB) {
            auto diff = (*itB)->pos - (*itA)->pos;
            auto distSq = diff.dot(diff);
            if (distSq > maxDistSq) {
                maxDistSq = distSq;
                p0 = *itA;
                p1 = *itB;
            }
            }
        }

        p0->pos = {T(0), T(0.5), T(0)};
        p1->pos = {T(1), T(0.5), T(0)};
        
        if (pinned_uvs.size() >= 2) {
            pinnedVerts.reserve(pinned_uvs.size());
            for (const auto& [vidx, uv] : pinned_uvs) {
                auto vptr = mesh->vertices()[vidx];          // assumes vertex index == position in container
                if (!vptr) continue;
                vptr->pos = {uv.first, uv.second, T(0)};
                pinColOffset[vidx] = 2 * pinnedVerts.size(); // (u,v) occupy cols 2k,2k+1
                pinnedVerts.push_back(vptr);
            }
        } else {                           // keep previous behaviour
            pinnedVerts = {p0, p1};
            pinColOffset[p0->idx] = 0;
            pinColOffset[p1->idx] = 2;
        }
        auto numFixed = pinnedVerts.size();


        // std::cout << "Pinned vertices id: " << p0->idx << " and " << p1->idx << std::endl;
        // std::cout << "Pinned vertices: " << p0->pos << " and " << p1->pos << std::endl;

        // update vertex_map with mesh.get_vertex_map()

        // Try to get vertex map from mesh if it exists 
        // and merge it with the provided vertex_map
        // std::unordered_map<std::size_t, std::size_t> updated_vertex_map = vertex_map;
        int pin_map_overlap = 0;
        try {
            auto mesh_vertex_map = mesh->get_vertex_map();
            for (const auto& [key, value] : mesh_vertex_map) {
                vertex_map[key] = value;
                if(key == p0->idx or key == p1->idx) {
                    pin_map_overlap++;
                }
            }
        } catch (const std::exception& e) {
            // Mesh doesn't have get_vertex_map() or it failed, 
            // just use the provided vertex_map
            std::cout << "Could not get vertex map from mesh: " << e.what() << std::endl;
        }

        // Print vertex map for debugging
        // std::cout << "Vertex Map Contents:" << std::endl;
        // for (const auto& [key, value] : vertex_map) {
        //     std::cout << "  " << key << " → " << value << std::endl;
        // }
        // For convenience
        auto numFaces = mesh->num_faces();
        auto numVerts = mesh->num_vertices();
        // auto numFixed = 2;

        auto numFree = numVerts - numFixed - vertex_map.size() + pin_map_overlap;


        auto all_verts = mesh->vertices();
        // Permutation for free vertices
        // This helps us find a vert's row in the solution matrix



        std::map<std::size_t, std::size_t> freeIdxTable;
        for (const auto& v : mesh->vertices()) {
            // if (v == p0 or v == p1 ) {
            //     // std::cout << "Pinned vertex: " << v->idx << std::endl;
            //     continue;
            // }
            if (pinColOffset.count(v->idx)) continue;
            if ( vertex_map.find(v->idx) != vertex_map.end()) {
                // std::cout << "mapped vertex: " << v->idx << std::endl;
                continue;
            }
            auto newIdx = freeIdxTable.size();
            freeIdxTable[v->idx] = newIdx;
        }

        // Setup pinned bFixed
        // std::vector<Triplet> tripletsB;
        // tripletsB.emplace_back(0, 0, p0->pos[0]);
        // tripletsB.emplace_back(1, 0, p0->pos[1]);
        // tripletsB.emplace_back(2, 0, p1->pos[0]);
        // tripletsB.emplace_back(3, 0, p1->pos[1]);
        // SparseMatrix bFixed(2 * numFixed, 1);
        // bFixed.reserve(tripletsB.size());

        // bFixed.setFromTriplets(tripletsB.begin(), tripletsB.end());



        std::vector<Triplet> tripletsB;
        for (std::size_t i = 0; i < pinnedVerts.size(); ++i) {
            tripletsB.emplace_back(2*i    , 0, pinnedVerts[i]->pos[0]); // u
            tripletsB.emplace_back(2*i + 1, 0, pinnedVerts[i]->pos[1]); // v
        }
        SparseMatrix bFixed(2 * numFixed, 1);
        bFixed.reserve(tripletsB.size());

        bFixed.setFromTriplets(tripletsB.begin(), tripletsB.end());
        // Setup variables matrix
        // Are only solving for free vertices, so push pins in special matrix
        std::vector<Triplet> tripletsA;
        tripletsB.clear();
        for (const auto& f : mesh->faces()) {
            auto e0 = f->head;
            auto e1 = e0->next;
            auto e2 = e1->next;
            auto sin0 = std::sin(e0->alpha);
            auto sin1 = std::sin(e1->alpha);
            auto sin2 = std::sin(e2->alpha);

            // Find the max sin idx
            std::vector<T> sins{sin0, sin1, sin2};
            auto sinMaxElem = std::max_element(sins.begin(), sins.end());
            auto sinMaxIdx = std::distance(sins.begin(), sinMaxElem);

            // Rotate the edge order of the face so last angle is largest
            if (sinMaxIdx == 0) {
                auto temp = e0;
                e0 = e1;
                e1 = e2;
                e2 = temp;
                sin0 = sins[1];
                sin1 = sins[2];
                sin2 = sins[0];
            } else if (sinMaxIdx == 1) {
                auto temp = e2;
                e2 = e1;
                e1 = e0;
                e0 = temp;
                sin0 = sins[2];
                sin1 = sins[0];
                sin2 = sins[1];
            }

            auto ratio = (sin2 == T(0)) ? T(1) : sin1 / sin2;
            auto cosine = std::cos(e0->alpha) * ratio;
            auto sine = sin0 * ratio;

            // If pin0 or pin1, put in fixedB matrix, else put in A
            auto row = 2 * f->idx;

            // Process e0 with vertex_map check
            auto e0vertex = (vertex_map.find(e0->vertex->idx) == vertex_map.end())
                      ? e0->vertex
                      : all_verts[vertex_map.at(e0->vertex->idx)];
            // std::cout << "e0vertex: " << e0vertex->idx << std::endl;

            // if (e0vertex == p0) {
            //     tripletsB.emplace_back(row, 0, cosine - T(1));
            //     tripletsB.emplace_back(row, 1, -sine);
            //     tripletsB.emplace_back(row + 1, 0, sine);
            //     tripletsB.emplace_back(row + 1, 1, cosine - T(1));
            // } else if (e0vertex == p1) {
            //     tripletsB.emplace_back(row, 2, cosine - T(1));
            //     tripletsB.emplace_back(row, 3, -sine);
            //     tripletsB.emplace_back(row + 1, 2, sine);
            //     tripletsB.emplace_back(row + 1, 3, cosine - T(1));
            // } 
            if (auto it = pinColOffset.find(e0vertex->idx); it != pinColOffset.end()) {
                std::size_t col = it->second;
                tripletsB.emplace_back(row    , col    , cosine - T(1));
                tripletsB.emplace_back(row    , col+1  , -sine);
                tripletsB.emplace_back(row + 1, col    ,  sine);
                tripletsB.emplace_back(row + 1, col+1  , cosine - T(1));
            } else {
                auto freeIdx = freeIdxTable.at(e0vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, cosine - T(1));
                tripletsA.emplace_back(row, 2 * freeIdx + 1, -sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx, sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, cosine - T(1));
            }


            // Process e1 with vertex_map check
            auto e1vertex = (vertex_map.find(e1->vertex->idx) == vertex_map.end())
                      ? e1->vertex
                      : all_verts[vertex_map.at(e1->vertex->idx)];

            // if (e1vertex == p0) {
            //     tripletsB.emplace_back(row, 0, -cosine);
            //     tripletsB.emplace_back(row, 1, sine);
            //     tripletsB.emplace_back(row + 1, 0, -sine);
            //     tripletsB.emplace_back(row + 1, 1, -cosine);
            // } else if (e1vertex == p1) {
            //     tripletsB.emplace_back(row, 2, -cosine);
            //     tripletsB.emplace_back(row, 3, sine);
            //     tripletsB.emplace_back(row + 1, 2, -sine);
            //     tripletsB.emplace_back(row + 1, 3, -cosine);
            // } else {

            if (auto it = pinColOffset.find(e1vertex->idx); it != pinColOffset.end()) {
                std::size_t col = it->second;
                tripletsB.emplace_back(row    , col    , -cosine);
                tripletsB.emplace_back(row    , col+1  , sine);
                tripletsB.emplace_back(row + 1, col    , -sine);
                tripletsB.emplace_back(row + 1, col+1  , -cosine);
            }  else {
                auto freeIdx = freeIdxTable.at(e1vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, -cosine);
                tripletsA.emplace_back(row, 2 * freeIdx + 1, sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx, -sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, -cosine);
            } 
            // std::cout << "e1vertex: " << e1vertex->idx << std::endl;
            // Process e2 with vertex_map check
            auto e2vertex = (vertex_map.find(e2->vertex->idx) == vertex_map.end())
                      ? e2->vertex
                      : all_verts[vertex_map.at(e2->vertex->idx)];


                // if (e2vertex == p0) {
                //     tripletsB.emplace_back(row, 0, T(1));
                //     tripletsB.emplace_back(row + 1, 1, T(1));
                // } else if (e2vertex == p1) {
                //     tripletsB.emplace_back(row, 2, T(1));
                //     tripletsB.emplace_back(row + 1, 3, T(1));
                if (auto it = pinColOffset.find(e2vertex->idx); it != pinColOffset.end()) {
                    std::size_t col = it->second;
                    tripletsB.emplace_back(row    , col    , T(1));
                    tripletsB.emplace_back(row + 1, col+1  , T(1));
                } else {
                    auto freeIdx = freeIdxTable.at(e2vertex->idx);
                    tripletsA.emplace_back(row, 2 * freeIdx, T(1));
                    tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, T(1));
                }
                // std::cout << "e2vertex: " << e2vertex->idx << std::endl;
            }


        SparseMatrix A(2 * numFaces, 2 * numFree);
        A.reserve(tripletsA.size());
        A.setFromTriplets(tripletsA.begin(), tripletsA.end());
        // A = removeEmptyCols(A);
        // std::cout << "A shape: " << A.rows() << " x " << A.cols() << std::endl;
    
        SparseMatrix bFree(2 * numFaces, 2 * numFixed);
        bFree.reserve(tripletsB.size());
        bFree.setFromTriplets(tripletsB.begin(), tripletsB.end());

        // Calculate rhs from free and fixed matrices
        SparseMatrix b = bFree * bFixed * -1;

        // Setup AtA and solver
        SparseMatrix AtA = A.transpose() * A;
        AtA.makeCompressed();
        Solver solver;
        // std::cout << "Using ABF++" << std::endl;

        solver.compute(AtA);
        // std::cout << "AtA factorized" << std::endl;
        if (solver.info() != Eigen::Success) {
            #ifdef VERBOSE

            std::cerr << "Error in factorization of AtA in ABF-LSCM compute" << solver.info() << std::endl;
            if (solver.info() == Eigen::NumericalIssue) {
                std::cerr << "Numerical issues in ABF." << std::endl;
            }else if (solver.info() == Eigen::InvalidInput) {
                std::cerr << "Invalid input" << std::endl;
            }else if (solver.info() == Eigen::NoConvergence) {
                std::cerr << "No convergence" << std::endl;
            }else if (solver.info() == Eigen::InvalidInput) {
                std::cerr << "Invalid input" << std::endl;
            }else {
                std::cerr << "Unknown error" << std::endl;
            }
            #endif
            // std::cerr << "Matrix A shape: " << A.rows() << " x " << A.cols() << std::endl;
            // std::cerr << "Matrix AtA shape: " << AtA.rows() << " x " << AtA.cols() << std::endl;
            // std::cerr << "Vector r size: " << b.size() << std::endl;
            // std::cerr << "Vector ATr size: " << ATr.size() << std::endl;
            // return 1;
        }
        #ifdef VERBOSE
            std::cout << "AtA factorized" << std::endl;
        #endif
    

        // Setup Atb
        SparseMatrix Atb = A.transpose() * b;

        // Solve AtAx = AtAb
        DenseMatrix x = solver.solve(Atb);

        // Assign solution to UV coordinates
        // Pins are already updated, so these are free vertices

        for (const auto& v : mesh->vertices()) {
            // if (v == p0 or v == p1) {
            //     continue;
            // }
            if (pinColOffset.count(v->idx)) continue;

            // std::cout << "v: " << v->idx << std::endl;
            std::size_t newIdx = 0;
            if (vertex_map.find(v->idx) != vertex_map.end()) {
                // std::cout << "v->idx: " << v->idx << std::endl;
                newIdx = vertex_map.at(v->idx);
                // if(newIdx == p0->idx) {
                //     v->pos[0] = p0->pos[0];
                //     v->pos[1] = p0->pos[1];
                //     v->pos[2] = T(0);
                //     continue;
                // } else if(newIdx == p1->idx) {
                //     v->pos[0] = p1->pos[0];
                //     v->pos[1] = p1->pos[1];
                //     v->pos[2] = T(0);
                //     continue;
                // }
                if (auto it = pinColOffset.find(newIdx); it != pinColOffset.end()) {
                    throw MeshException("NOT IMPLEMENTED");
                } else {
                    newIdx = 2 * freeIdxTable.at(vertex_map.at(v->idx));
                }

            } else {
                newIdx = 2 * freeIdxTable.at(v->idx);
            }
            // std::cout << "newIdx: " << newIdx << std::endl;
            v->pos[0] = x(newIdx, 0);
            v->pos[1] = x(newIdx + 1, 0);
            v->pos[2] = T(0);
            // std::cout << "v->pos: " << v->pos << std::endl;
        }


            
            
        // }

        // }
        return 0;
    }


    #if 0
    /**
     * @brief Compute the parameterized mesh
     *
     * @throws MeshException If pinned vertex is not on boundary.
     * @throws SolverException If matrix cannot be decomposed or if solver fails
     * to find a solution.
     */


    static void Compute_Constraint_Edge(typename Mesh::Pointer& mesh, 
                        const std::unordered_map<std::size_t, Vec<T, 3>>& addition_pinned_vertex_map = std::unordered_map<std::size_t, Vec<T, 3>>{},
                        const std::unordered_map<std::size_t, std::pair<std::size_t,std::size_t>>& vertex_pin_edges = std::unordered_map<std::size_t, std::pair<std::size_t,std::size_t>>{}) // vertex_index -> pair of pinned index
    {
        using Triplet = Eigen::Triplet<T>;
        using SparseMatrix = Eigen::SparseMatrix<T>;
        using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        auto verts = mesh->vertices_boundary();
        if (verts.size() < 2) {
            throw MeshException("Not enough vertices to pin");
        }
        auto p0 = *verts.begin();
        auto p1 = p0;
        T maxDistSq = T(-1);
        for (auto itA = verts.begin(); itA != verts.end(); ++itA) {
            for (auto itB = std::next(itA); itB != verts.end(); ++itB) {
            auto diff = (*itB)->pos - (*itA)->pos;
            auto distSq = diff.dot(diff);
            if (distSq > maxDistSq) {
                maxDistSq = distSq;
                p0 = *itA;
                p1 = *itB;
            }
            }
        }

        p0->pos = {T(0), T(0.5), T(0)};
        p1->pos = {T(1), T(0.5), T(0)};
        
        // std::cout << "Pinned vertices id: " << p0->idx << " and " << p1->idx << std::endl;
        // std::cout << "Pinned vertices: " << p0->pos << " and " << p1->pos << std::endl;

        // For convenience
        auto numFaces = mesh->num_faces();
        auto numVerts = mesh->num_vertices();
        auto numFixed = 2+addition_pinned_vertex_map.size();
        auto numFree = numVerts - numFixed - vertex_pin_edges.size();
        auto numConstraints = vertex_pin_edges.size();


        auto all_verts = mesh->vertices();
        // Permutation for free vertices
        // This helps us find a vert's row in the solution matrix


        for(auto& [idx, pos] : addition_pinned_vertex_map) {
            // std::cout << "Pinned vertex: " << idx << " at " << pos << std::endl;
            all_verts[idx]->pos = pos;  
        }

        std::map<std::size_t, std::size_t> freeIdxTable;
        std::map<std::size_t, std::size_t> fixIdxTable;

        for (const auto& v : mesh->vertices()) {
            if (v == p0 or v == p1 or addition_pinned_vertex_map.find(v->idx) != addition_pinned_vertex_map.end() or vertex_pin_edges.find(v->idx) != vertex_pin_edges.end()) {
                continue;
                std::cout << "Pinned vertex: " << v->idx << std::endl;
            }
            auto newIdx = freeIdxTable.size();
            freeIdxTable[v->idx] = newIdx;
        }


        auto newIdx = freeIdxTable.size();
        auto i = 0;
        for (const auto& [idx, pos] : vertex_pin_edges) {
            freeIdxTable[idx] = 2 * (newIdx) + i++;
            std::cout << "constrainted vertex: " << idx << " at " <<  freeIdxTable[idx] << std::endl;
        }

        // Setup pinned bFixed
        std::vector<Triplet> tripletsB;
        tripletsB.emplace_back(0, 0, p0->pos[0]);
        tripletsB.emplace_back(1, 0, p0->pos[1]);
        tripletsB.emplace_back(2, 0, p1->pos[0]);
        tripletsB.emplace_back(3, 0, p1->pos[1]);

        auto pin_index = 0;
        for (const auto& [idx, pos] : addition_pinned_vertex_map) {
            tripletsB.emplace_back(4+2*pin_index, 0, pos[0]);
            tripletsB.emplace_back(5+2*pin_index, 0, pos[1]);
            fixIdxTable[idx] = 4+2*pin_index;
            pin_index++;

            std::cout << "Pinned vertex: " << idx << " at " <<  fixIdxTable[idx] << std::endl;

        }

        SparseMatrix bFixed(2 * numFixed, 1);
        bFixed.reserve(tripletsB.size());
        bFixed.setFromTriplets(tripletsB.begin(), tripletsB.end());

        // Setup variables matrix
        // Are only solving for free vertices, so push pins in special matrix
        std::vector<Triplet> tripletsA;
        std::vector<Triplet> tripletsEdge;

        tripletsB.clear();
        for (const auto& f : mesh->faces()) {
            auto e0 = f->head;
            auto e1 = e0->next;
            auto e2 = e1->next;
            auto sin0 = std::sin(e0->alpha);
            auto sin1 = std::sin(e1->alpha);
            auto sin2 = std::sin(e2->alpha);

            // Find the max sin idx
            std::vector<T> sins{sin0, sin1, sin2};
            auto sinMaxElem = std::max_element(sins.begin(), sins.end());
            auto sinMaxIdx = std::distance(sins.begin(), sinMaxElem);

            // Rotate the edge order of the face so last angle is largest
            if (sinMaxIdx == 0) {
                auto temp = e0;
                e0 = e1;
                e1 = e2;
                e2 = temp;
                sin0 = sins[1];
                sin1 = sins[2];
                sin2 = sins[0];
            } else if (sinMaxIdx == 1) {
                auto temp = e2;
                e2 = e1;
                e1 = e0;
                e0 = temp;
                sin0 = sins[2];
                sin1 = sins[0];
                sin2 = sins[1];
            }

            auto ratio = (sin2 == T(0)) ? T(1) : sin1 / sin2;
            auto cosine = std::cos(e0->alpha) * ratio;
            auto sine = sin0 * ratio;

            // If pin0 or pin1, put in fixedB matrix, else put in A
            auto row = 2 * f->idx;

            // Process e0 with vertex_map check
            // auto e0vertex = (vertex_map.find(e0->vertex->idx) == vertex_map.end())
            //           ? e0->vertex
            //           : all_verts[vertex_map.at(e0->vertex->idx)];

            auto e0vertex_idx = e0->vertex->idx;
            auto e1vertex_idx = e1->vertex->idx;
            auto e2vertex_idx = e2->vertex->idx;

            // double a,b,c,d;
            

            if (addition_pinned_vertex_map.find(e0vertex_idx) != addition_pinned_vertex_map.end()) {
                tripletsB.emplace_back(row, fixIdxTable.at(e0vertex_idx), cosine - T(1));
                tripletsB.emplace_back(row, fixIdxTable.at(e0vertex_idx)+1, -sine);
                tripletsB.emplace_back(row + 1, fixIdxTable.at(e0vertex_idx), sine);
                tripletsB.emplace_back(row + 1, fixIdxTable.at(e0vertex_idx)+1, cosine - T(1));
            } else if (vertex_pin_edges.find(e0vertex_idx) != vertex_pin_edges.end()) {
                auto [idx1, idx2] = vertex_pin_edges.at(e0vertex_idx);


                
                auto posA = addition_pinned_vertex_map.at(idx1);
                auto posB = addition_pinned_vertex_map.at(idx2);

                auto a = posB[0] - posA[0];
                auto b = posA[0];
                auto c = posB[1] - posA[1];
                auto d = posA[1];


                auto A =   cosine - T(1);
                auto B = sine;

                tripletsA.emplace_back(row, freeIdxTable.at(e0vertex_idx), (A*a - B*c));
                tripletsA.emplace_back(row+1, freeIdxTable.at(e0vertex_idx), (B*a + A*c));

                tripletsEdge.emplace_back(row, 0, A*b - B*d);
                tripletsEdge.emplace_back(row+1, 0, B*b + A*d);

            } else if (e0->vertex == p0) {
                tripletsB.emplace_back(row, 0, cosine - T(1));
                tripletsB.emplace_back(row, 1, -sine);
                tripletsB.emplace_back(row + 1, 0, sine);
                tripletsB.emplace_back(row + 1, 1, cosine - T(1));
            } else if (e0->vertex == p1) {
                tripletsB.emplace_back(row, 2, cosine - T(1));
                tripletsB.emplace_back(row, 3, -sine);
                tripletsB.emplace_back(row + 1, 2, sine);
                tripletsB.emplace_back(row + 1, 3, cosine - T(1));
            } else {
                auto freeIdx = freeIdxTable.at(e0->vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, cosine - T(1));
                tripletsA.emplace_back(row, 2 * freeIdx + 1, -sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx, sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, cosine - T(1));
            }

            // Process e1 with vertex_map check
            auto e1vertex = e1->vertex;

            if (addition_pinned_vertex_map.find(e1vertex_idx) != addition_pinned_vertex_map.end()) {
                tripletsB.emplace_back(row, fixIdxTable.at(e1vertex_idx), -cosine);
                tripletsB.emplace_back(row, fixIdxTable.at(e1vertex_idx)+1, sine);
                tripletsB.emplace_back(row + 1, fixIdxTable.at(e1vertex_idx), -sine);
                tripletsB.emplace_back(row + 1, fixIdxTable.at(e1vertex_idx)+1, -cosine);
            } else if (vertex_pin_edges.find(e1vertex_idx) != vertex_pin_edges.end()) {
                auto [idx1, idx2] = vertex_pin_edges.at(e1vertex_idx);
                auto posA = addition_pinned_vertex_map.at(idx1);
                auto posB = addition_pinned_vertex_map.at(idx2);

                auto a = posB[0] - posA[0];
                auto b = posA[0];
                auto c = posB[1] - posA[1];
                auto d = posA[1];


                auto A = -cosine;
                auto B = -sine;

                tripletsA.emplace_back(row, freeIdxTable.at(e1vertex_idx), (A*a - B*c));
                tripletsA.emplace_back(row+1, freeIdxTable.at(e1vertex_idx), (B*a + A*c));
                
                tripletsEdge.emplace_back(row, 0,  A*b - B*d );
                tripletsEdge.emplace_back(row+1, 0, B*b + A*d);
            

        
            } else if (e1vertex == p0) {
                tripletsB.emplace_back(row, 0, -cosine);
                tripletsB.emplace_back(row, 1, sine);
                tripletsB.emplace_back(row + 1, 0, -sine);
                tripletsB.emplace_back(row + 1, 1, -cosine);
            } else if (e1vertex == p1) {
                tripletsB.emplace_back(row, 2, -cosine);
                tripletsB.emplace_back(row, 3, sine);
                tripletsB.emplace_back(row + 1, 2, -sine);
                tripletsB.emplace_back(row + 1, 3, -cosine);
            } else {
                auto freeIdx = freeIdxTable.at(e1vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, -cosine);
                tripletsA.emplace_back(row, 2 * freeIdx + 1, sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx, -sine);
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, -cosine);
            } 


            // Process e2 with vertex_map check
            auto e2vertex =  e2->vertex;


            if (addition_pinned_vertex_map.find(e2vertex_idx) != addition_pinned_vertex_map.end()) {
                tripletsB.emplace_back(row, fixIdxTable.at(e2vertex_idx), T(1));
                tripletsB.emplace_back(row + 1, fixIdxTable.at(e2vertex_idx)+1, T(1));
            }else if (vertex_pin_edges.find(e2vertex_idx) != vertex_pin_edges.end()) {
                auto [idx1, idx2] = vertex_pin_edges.at(e2vertex_idx);
                auto posA = addition_pinned_vertex_map.at(idx1);
                auto posB = addition_pinned_vertex_map.at(idx2);

                auto a = posB[0] - posA[0];
                auto b = posA[0];
                auto c = posB[1] - posA[1];
                auto d = posA[1];

                auto A = T(1);
                auto B = T(0);

                tripletsA.emplace_back(row, freeIdxTable.at(e2vertex_idx), (A*a - B*c));
                tripletsA.emplace_back(row+1, freeIdxTable.at(e2vertex_idx), (B*a + A*c));
                
                tripletsEdge.emplace_back(row, 0, A*b - B*d);
                tripletsEdge.emplace_back(row+1, 0, B*b + A*d);
            }
            else if (e2vertex == p0) {
                tripletsB.emplace_back(row, 0, T(1));
                tripletsB.emplace_back(row + 1, 1, T(1));
            } else if (e2vertex == p1) {
                tripletsB.emplace_back(row, 2, T(1));
                tripletsB.emplace_back(row + 1, 3, T(1));
            } else {
                auto freeIdx = freeIdxTable.at(e2vertex->idx);
                tripletsA.emplace_back(row, 2 * freeIdx, T(1));
                tripletsA.emplace_back(row + 1, 2 * freeIdx + 1, T(1));
            }
        }

        for (const auto& kv : freeIdxTable) {
            std::cout << "Vertex ID: " << kv.first << " -> Free Index: " << kv.second << std::endl;
        }

        for (const auto& triplet : tripletsA) {
            std::cout << "Row: " << triplet.row()
                      << ", Col: " << triplet.col()
                      << ", Value: " << triplet.value() << std::endl;
        }

        SparseMatrix A(2 * numFaces, 2 * numFree + numConstraints);
        std::cout << "Matrix A shape: " << A.rows() << " x " << A.cols() << std::endl;
        A.reserve(tripletsA.size());
        A.setFromTriplets(tripletsA.begin(), tripletsA.end());

        SparseMatrix bFree(2 * numFaces, 2 * numFixed);
        bFree.reserve(tripletsB.size());
        bFree.setFromTriplets(tripletsB.begin(), tripletsB.end());

        SparseMatrix bEdge(2 * numFaces, 1);
        bEdge.reserve(tripletsEdge.size());
        bEdge.setFromTriplets(tripletsEdge.begin(), tripletsEdge.end());

        std::cout << "Matrix bEdge shape: " << bEdge.rows() << " x " << bEdge.cols() << std::endl;
        for (const auto& triplet : tripletsEdge) {
            std::cout << "bEdge Row: " << triplet.row()
                      << ", Col: " << triplet.col()
                      << ", Value: " << triplet.value() << std::endl;
        }

        // Calculate rhs from free and fixed matrices
        SparseMatrix b = bFree * bFixed * -1 - bEdge;

        // Setup AtA and solver
        SparseMatrix AtA = A.transpose() * A;
        AtA.makeCompressed();
        Solver solver;
        // std::cout << "Using ABF++" << std::endl;

        solver.compute(AtA);
        #ifdef VERBOSE
        // std::cout << "AtA factorized" << std::endl;
        if (solver.info() != Eigen::Success) {
            std::cerr << "Error in factorization of AtA" << solver.info() << std::endl;
            if (solver.info() == Eigen::NumericalIssue) {
                std::cerr << "Numerical issues in ABF." << std::endl;
            }else if (solver.info() == Eigen::InvalidInput) {
                std::cerr << "Invalid input" << std::endl;
            }else if (solver.info() == Eigen::NoConvergence) {
                std::cerr << "No convergence" << std::endl;
            }else if (solver.info() == Eigen::InvalidInput) {
                std::cerr << "Invalid input" << std::endl;
            }else {
                std::cerr << "Unknown error" << std::endl;
            }
    
            std::cerr << "Matrix A shape: " << A.rows() << " x " << A.cols() << std::endl;
            // std::cerr << "Matrix AtA shape: " << AtA.rows() << " x " << AtA.cols() << std::endl;
            std::cerr << "Vector r size: " << b.size() << std::endl;
            // std::cerr << "Vector ATr size: " << ATr.size() << std::endl;
            // return 1;
        }
            std::cout << "AtA factorized" << std::endl;
        #endif
    

        // Setup Atb
        SparseMatrix Atb = A.transpose() * b;

        // Solve AtAx = AtAb
        DenseMatrix x = solver.solve(Atb);

        // Assign solution to UV coordinates
        // Pins are already updated, so these are free vertices
        for (const auto& v : mesh->vertices()) {
            if (v == p0 or v == p1 or addition_pinned_vertex_map.find(v->idx) != addition_pinned_vertex_map.end()) {
                continue;
            }
            std::size_t newIdx = 0;
            if (vertex_pin_edges.find(v->idx) != vertex_pin_edges.end()) {
                newIdx = freeIdxTable.at(v->idx);
                auto t = x(newIdx, 0);
                std::cout << "t: " << t << std::endl;
                v->pos[0] = addition_pinned_vertex_map.at(vertex_pin_edges.at(v->idx).first)[0] + t * (addition_pinned_vertex_map.at(vertex_pin_edges.at(v->idx).second)[0] - addition_pinned_vertex_map.at(vertex_pin_edges.at(v->idx).first)[0]);
                v->pos[1] = addition_pinned_vertex_map.at(vertex_pin_edges.at(v->idx).first)[1] + t * (addition_pinned_vertex_map.at(vertex_pin_edges.at(v->idx).second)[1] - addition_pinned_vertex_map.at(vertex_pin_edges.at(v->idx).first)[1]);
                v->pos[2] = T(0);
            } else {
                newIdx = 2 * freeIdxTable.at(v->idx);
                v->pos[0] = x(newIdx, 0);
                v->pos[1] = x(newIdx + 1, 0);
                v->pos[2] = T(0);
            }
        }
    }


    #endif

};
}  // namespace OpenABF