#include "AngleBasedLSCM.hpp"
#include <iostream>

namespace OpenABF
{

// --- compute() ---------------------------------------------------
template <typename T, class MeshType, class Solver, bool Enable>
void AngleBasedLSCM<T, MeshType, Solver, Enable>::compute(
    typename Mesh::Pointer& mesh) const
{
    // Simply delegate to the static Compute function.
    Compute(mesh);
}

// --- Compute() ---------------------------------------------------
template <typename T, class MeshType, class Solver, bool Enable>
void AngleBasedLSCM<T, MeshType, Solver, Enable>::Compute(
    typename Mesh::Pointer& mesh,
    const std::unordered_map<std::size_t, std::size_t>& vertex_map,
    const std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>>&
        vertex_pin_edges)
{

    // Pinned vertex selection
    // Get the end points of a boundary edge
    // auto p0 = mesh->vertices_boundary()[0];
    // auto e = p0->edge;
    // do {
    //     if (not e->pair) {
    //         break;
    //     }
    //     e = e->pair->next;
    // } while (e != p0->edge);
    // if (e == p0->edge and e->pair) {
    //     throw MeshException("Pinned vertex not on boundary");
    // }
    // auto p1 = e->next->vertex;

    // // Find pair of vertices with maximum distance
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

    // ------------------------------------------------------------
    // 1) Gather boundary vertices
    // ------------------------------------------------------------
    // auto boundaryVerts = mesh->vertices_boundary();
    // const size_t N = boundaryVerts.size();
    // if (N < 2) {
    //     throw MeshException("Not enough boundary vertices to pin");
    // }

    // // ------------------------------------------------------------
    // // 2) Precompute edge lengths around the boundary
    // // ------------------------------------------------------------
    // std::vector<T> edgeLens(N, T(0));
    // T totalLen = T(0);
    // for (size_t i = 0; i < N; ++i)
    // {
    //     size_t iNext = (i + 1) % N;
    //     auto diff = boundaryVerts[iNext]->pos - boundaryVerts[i]->pos;
    //     T length = diff.magnitude();
    //     edgeLens[i] = length;
    //     totalLen += length;
    // }

    // // ------------------------------------------------------------
    // // 3) Find p0 = "meet in the middle" (forward/backward) along boundary
    // // ------------------------------------------------------------
    // size_t i1 = 0;
    // size_t i2 = N - 1;
    // T len1 = T(0);
    // T len2 = T(0);

    // // Move i1 forward, i2 backward until they meet
    // while (i1 != i2)
    // {
    //     if (len1 < len2)
    //     {
    //         len1 += edgeLens[i1];
    //         i1 = (i1 + 1) % N;
    //     }
    //     else
    //     {
    //         i2 = (i2 + N - 1) % N; // safe backward step
    //         len2 += edgeLens[i2];
    //     }
    // }

    // auto p0 = boundaryVerts[i1];  // pinned vertex #1

    // // ------------------------------------------------------------
    // // 4) Find p1 by "meet in the middle" from opposite directions
    // //    (This matches the logic in p_chart_symmetry_pins,
    // //     reversing i1 / i2 stepping.)
    // // ------------------------------------------------------------
    // i1 = 0;
    // i2 = N - 1;
    // len1 = T(0);
    // len2 = T(0);

    // while (i1 != i2)
    // {
    //     if (len1 < len2)
    //     {
    //         // move i1 backward
    //         i1 = (i1 + N - 1) % N;
    //         len1 += edgeLens[i1];
    //     }
    //     else
    //     {
    //         // move i2 forward
    //         len2 += edgeLens[i2];
    //         i2 = (i2 + 1) % N;
    //     }
    // }

    // auto p1 = boundaryVerts[i1];  // pinned vertex #2

    // ------------------------------------------------------------
    // 5) (Optional) Align p0/p1 along a coordinate axis
    //    (Exactly as in your original code, but using p0, p1 from above)
    // ------------------------------------------------------------
    // auto pinVec = p1->pos - p0->pos;
    // auto dist = pinVec.norm();
    // pinVec /= dist;

    // // Shift p0 to origin
    // p0->pos = {T(0), T(0), T(0)};

    // // Determine principal axis to align with (the largest component)
    // // Note: if your mesh->Vertex::pos is Eigen::Matrix<T,3,1>,
    // // you can do something like pinVec[0], pinVec[1], pinVec[2].
    // // The snippet below uses std::max_element on raw data,
    // // but adapt as needed for your vector type.
    // const T* dataPtr = pinVec.data();
    // auto maxElem  = std::max_element(dataPtr, dataPtr + 3,
    //                                 [](T a, T b) { return std::abs(a) <
    //                                 std::abs(b); });
    // auto maxAxis  = std::distance(dataPtr, maxElem);
    // dist = std::copysign(dist, *maxElem);

    // // Place p1 along the identified axis
    // if (maxAxis == 0) {
    //     p1->pos = { dist, T(0), T(0) };
    // }
    // else if (maxAxis == 1) {
    //     p1->pos = { T(0), dist, T(0) };
    // }
    // else {
    //     p1->pos = { T(0), T(0), dist };
    // }

    // auto pinVec = p1->pos - p0->pos;
    // auto dist = norm(pinVec);
    // pinVec /= dist;
    // p0->pos = {T(0), T(0), T(0)};
    // auto maxElem = std::max_element(pinVec.begin(), pinVec.end());
    // auto maxAxis = std::distance(pinVec.begin(), maxElem);
    // dist = std::copysign(dist, *maxElem);
    // if (maxAxis == 0) {
    //     p1->pos = {dist, T(0), T(0)};
    // } else {
    //     p1->pos = {T(0), dist, T(0)};
    // }

    // 2) Apply the p_chart_pin_positions logic (adapted)
    // auto pinPositions = [&](auto v0, auto v1) {
    //     // if degenerate, fallback
    //     if (true) {

    //     // if (!v0 || !v1 || v0 == v1) {
    //         // For example:
    //         v0->pos[0] = T(0);
    //         v0->pos[1] = T(0.5);
    //         v0->pos[2] = T(0);
    //         v1->pos[0] = T(1);
    //         v1->pos[1] = T(0.5);
    //         v1->pos[2] = T(0);
    //         return;
    //     }
    //     auto pinVec = p1->pos - p0->pos;
    //     auto dist = norm(pinVec);

    //     auto diff =              pinVec / dist;
    //     int dirx, diry;

    //     // find largest coordinate
    //     if (diff[0] > diff[1] && diff[0] > diff[2]) {
    //         dirx = 0;
    //         diry = (diff[1] > diff[2]) ? 1 : 2;
    //     }
    //     else if (diff[1] > diff[0] && diff[1] > diff[2]) {
    //         dirx = 1;
    //         diry = (diff[0] > diff[2]) ? 0 : 2;
    //     }
    //     else {
    //         dirx = 2;
    //         diry = (diff[0] > diff[1]) ? 0 : 1;
    //     }

    //     int diru, dirv;
    //     if (dirx == 2) {
    //         diru = 1;
    //         dirv = 0;
    //     }
    //     else {
    //         diru = 0;
    //         dirv = 1;
    //     }

    //     // set pinned "UV" positions based on largest dims in 3D "pos"
    //     T vx0 = v0->pos[dirx];
    //     T vy0 = v0->pos[diry];
    //     T vx1 = v1->pos[dirx];
    //     T vy1 = v1->pos[diry];

    //     // Overwrite the same pos or store them in separate uv fields
    //     v0->pos[diru] = vx0;
    //     v0->pos[dirv] = vy0;
    //     // Zero-out the other dimension if you want purely 2D:
    //     // e.g., if diru=0, dirv=1, then set pos[2] = 0 for both
    //     // We'll do it explicitly:
    //     if (diru != 2 && dirv != 2) {
    //         v0->pos[2] = T(0);
    //     }

    //     v1->pos[diru] = vx1;
    //     v1->pos[dirv] = vy1;
    //     if (diru != 2 && dirv != 2) {
    //         v1->pos[2] = T(0);
    //     }
    // };

    // Call the adapted function
    // pinPositions(p0, p1);

    p0->pos = {T(0), T(0.5), T(0)};
    p1->pos = {T(1), T(0.5), T(0)};

    // std::cout << "Pinned vertices id: " << p0->idx << " and " << p1->idx <<
    // std::endl; std::cout << "Pinned vertices: " << p0->pos << " and " <<
    // p1->pos << std::endl;

    // For convenience
    auto numFaces = mesh->num_faces();
    auto numVerts = mesh->num_vertices();
    auto numFixed = 2;
    auto numFree = numVerts - numFixed - vertex_map.size();

    auto all_verts = mesh->vertices();
    // Permutation for free vertices
    // This helps us find a vert's row in the solution matrix

    std::map<std::size_t, std::size_t> freeIdxTable;
    for (const auto& v : mesh->vertices()) {
        if (v == p0 or v == p1 or vertex_map.find(v->idx) != vertex_map.end()) {
            continue;
            std::cout << "Pinned vertex: " << v->idx << std::endl;
        }
        auto newIdx = freeIdxTable.size();
        freeIdxTable[v->idx] = newIdx;
    }

    // Setup pinned bFixed
    std::vector<Triplet> tripletsB;
    tripletsB.emplace_back(0, 0, p0->pos[0]);
    tripletsB.emplace_back(1, 0, p0->pos[1]);
    tripletsB.emplace_back(2, 0, p1->pos[0]);
    tripletsB.emplace_back(3, 0, p1->pos[1]);
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

        if (e0vertex == p0) {
            tripletsB.emplace_back(row, 0, cosine - T(1));
            tripletsB.emplace_back(row, 1, -sine);
            tripletsB.emplace_back(row + 1, 0, sine);
            tripletsB.emplace_back(row + 1, 1, cosine - T(1));
        } else if (e0vertex == p1) {
            tripletsB.emplace_back(row, 2, cosine - T(1));
            tripletsB.emplace_back(row, 3, -sine);
            tripletsB.emplace_back(row + 1, 2, sine);
            tripletsB.emplace_back(row + 1, 3, cosine - T(1));
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

        if (e1vertex == p0) {
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
        auto e2vertex = (vertex_map.find(e2->vertex->idx) == vertex_map.end())
                            ? e2->vertex
                            : all_verts[vertex_map.at(e2->vertex->idx)];

        if (e2vertex == p0) {
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
    SparseMatrix A(2 * numFaces, 2 * numFree);
    A.reserve(tripletsA.size());
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());

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
        std::cerr << "Error in factorization of AtA" << solver.info()
                  << std::endl;
        if (solver.info() == Eigen::NumericalIssue) {
            std::cerr << "Numerical issue, check for your meshes! Are there "
                         "repetitive vertices?"
                      << std::endl;
        } else if (solver.info() == Eigen::InvalidInput) {
            std::cerr << "Invalid input" << std::endl;
        } else if (solver.info() == Eigen::NoConvergence) {
            std::cerr << "No convergence" << std::endl;
        } else if (solver.info() == Eigen::InvalidInput) {
            std::cerr << "Invalid input" << std::endl;
        } else {
            std::cerr << "Unknown error" << std::endl;
        }

        std::cerr << "Matrix A shape: " << A.rows() << " x " << A.cols()
                  << std::endl;
        // std::cerr << "Matrix AtA shape: " << AtA.rows() << " x " <<
        // AtA.cols() << std::endl;
        std::cerr << "Vector r size: " << b.size() << std::endl;
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
        if (v == p0 or v == p1) {
            continue;
        }
        std::size_t newIdx = 0;
        if (vertex_map.find(v->idx) != vertex_map.end()) {
            newIdx = 2 * freeIdxTable.at(vertex_map.at(v->idx));
        } else {
            newIdx = 2 * freeIdxTable.at(v->idx);
        }
        v->pos[0] = x(newIdx, 0);
        v->pos[1] = x(newIdx + 1, 0);
        v->pos[2] = T(0);
    }
}

// --- Compute_Constraint_Edge() -----------------------------------
template <typename T, class MeshType, class Solver, bool Enable>
void AngleBasedLSCM<T, MeshType, Solver, Enable>::Compute_Constraint_Edge(
    typename Mesh::Pointer& mesh,
    const std::unordered_map<std::size_t, Vec<T, 3>>&
        addition_pinned_vertex_map,
    const std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>>&
        vertex_pin_edges)
{
    // Omitted: full implementation.

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

    // std::cout << "Pinned vertices id: " << p0->idx << " and " << p1->idx <<
    // std::endl; std::cout << "Pinned vertices: " << p0->pos << " and " <<
    // p1->pos << std::endl;

    // For convenience
    auto numFaces = mesh->num_faces();
    auto numVerts = mesh->num_vertices();
    auto numFixed = 2 + addition_pinned_vertex_map.size();
    auto numFree = numVerts - numFixed - vertex_pin_edges.size();
    auto numConstraints = vertex_pin_edges.size();

    auto all_verts = mesh->vertices();
    // Permutation for free vertices
    // This helps us find a vert's row in the solution matrix

    for (auto& [idx, pos] : addition_pinned_vertex_map) {
        // std::cout << "Pinned vertex: " << idx << " at " << pos << std::endl;
        all_verts[idx]->pos = pos;
    }

    std::map<std::size_t, std::size_t> freeIdxTable;
    std::map<std::size_t, std::size_t> fixIdxTable;

    for (const auto& v : mesh->vertices()) {
        if (v == p0 or v == p1 or
            addition_pinned_vertex_map.find(v->idx) !=
                addition_pinned_vertex_map.end() or
            vertex_pin_edges.find(v->idx) != vertex_pin_edges.end()) {
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
        std::cout << "constrainted vertex: " << idx << " at "
                  << freeIdxTable[idx] << std::endl;
    }

    // Setup pinned bFixed
    std::vector<Triplet> tripletsB;
    tripletsB.emplace_back(0, 0, p0->pos[0]);
    tripletsB.emplace_back(1, 0, p0->pos[1]);
    tripletsB.emplace_back(2, 0, p1->pos[0]);
    tripletsB.emplace_back(3, 0, p1->pos[1]);

    auto pin_index = 0;
    for (const auto& [idx, pos] : addition_pinned_vertex_map) {
        tripletsB.emplace_back(4 + 2 * pin_index, 0, pos[0]);
        tripletsB.emplace_back(5 + 2 * pin_index, 0, pos[1]);
        fixIdxTable[idx] = 4 + 2 * pin_index;
        pin_index++;

        std::cout << "Pinned vertex: " << idx << " at " << fixIdxTable[idx]
                  << std::endl;
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
        // auto e0vertex = (vertex_map.find(e0->vertex->idx) ==
        // vertex_map.end())
        //           ? e0->vertex
        //           : all_verts[vertex_map.at(e0->vertex->idx)];

        auto e0vertex_idx = e0->vertex->idx;
        auto e1vertex_idx = e1->vertex->idx;
        auto e2vertex_idx = e2->vertex->idx;

        // double a,b,c,d;

        if (addition_pinned_vertex_map.find(e0vertex_idx) !=
            addition_pinned_vertex_map.end()) {
            tripletsB.emplace_back(
                row, fixIdxTable.at(e0vertex_idx), cosine - T(1));
            tripletsB.emplace_back(
                row, fixIdxTable.at(e0vertex_idx) + 1, -sine);
            tripletsB.emplace_back(row + 1, fixIdxTable.at(e0vertex_idx), sine);
            tripletsB.emplace_back(
                row + 1, fixIdxTable.at(e0vertex_idx) + 1, cosine - T(1));
        } else if (
            vertex_pin_edges.find(e0vertex_idx) != vertex_pin_edges.end()) {
            auto [idx1, idx2] = vertex_pin_edges.at(e0vertex_idx);

            auto posA = addition_pinned_vertex_map.at(idx1);
            auto posB = addition_pinned_vertex_map.at(idx2);

            auto a = posB[0] - posA[0];
            auto b = posA[0];
            auto c = posB[1] - posA[1];
            auto d = posA[1];

            auto A = cosine;
            auto B = sine;

            tripletsA.emplace_back(
                row, freeIdxTable.at(e0vertex_idx), (A * a - B * c));
            tripletsA.emplace_back(
                row + 1, freeIdxTable.at(e0vertex_idx), (B * a + A * c));

            tripletsEdge.emplace_back(row, 0, -(A * b + B * d));
            tripletsEdge.emplace_back(row + 1, 0, B * b + A * d);

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

        if (addition_pinned_vertex_map.find(e1vertex_idx) !=
            addition_pinned_vertex_map.end()) {
            tripletsB.emplace_back(row, fixIdxTable.at(e1vertex_idx), -cosine);
            tripletsB.emplace_back(row, fixIdxTable.at(e1vertex_idx) + 1, sine);
            tripletsB.emplace_back(
                row + 1, fixIdxTable.at(e1vertex_idx), -sine);
            tripletsB.emplace_back(
                row + 1, fixIdxTable.at(e1vertex_idx) + 1, -cosine);
        } else if (
            vertex_pin_edges.find(e1vertex_idx) != vertex_pin_edges.end()) {
            auto [idx1, idx2] = vertex_pin_edges.at(e1vertex_idx);
            auto posA = addition_pinned_vertex_map.at(idx1);
            auto posB = addition_pinned_vertex_map.at(idx2);

            auto a = posB[0] - posA[0];
            auto b = posA[0];
            auto c = posB[1] - posA[1];
            auto d = posA[1];

            auto A = -cosine;
            auto B = -sine;

            tripletsA.emplace_back(
                row, freeIdxTable.at(e1vertex_idx), (A * a - B * c));
            tripletsA.emplace_back(
                row + 1, freeIdxTable.at(e1vertex_idx), (B * a + A * c));

            tripletsEdge.emplace_back(row, 0, -(A * b + B * d));
            tripletsEdge.emplace_back(row + 1, 0, B * b + A * d);

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
        auto e2vertex = e2->vertex;

        if (addition_pinned_vertex_map.find(e2vertex_idx) !=
            addition_pinned_vertex_map.end()) {

            tripletsB.emplace_back(row, fixIdxTable.at(e2vertex_idx), T(1));
            tripletsB.emplace_back(
                row + 1, fixIdxTable.at(e2vertex_idx) + 1, T(1));
        } else if (e2vertex == p0) {
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
        std::cout << "Vertex ID: " << kv.first
                  << " -> Free Index: " << kv.second << std::endl;
    }

    for (const auto& triplet : tripletsA) {
        std::cout << "Row: " << triplet.row() << ", Col: " << triplet.col()
                  << ", Value: " << triplet.value() << std::endl;
    }

    SparseMatrix A(2 * numFaces, 2 * numFree + numConstraints);
    std::cout << "Matrix A shape: " << A.rows() << " x " << A.cols()
              << std::endl;
    A.reserve(tripletsA.size());
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());

    SparseMatrix bFree(2 * numFaces, 2 * numFixed);
    bFree.reserve(tripletsB.size());
    bFree.setFromTriplets(tripletsB.begin(), tripletsB.end());

    SparseMatrix bEdge(2 * numFaces, 1);
    bEdge.reserve(tripletsEdge.size());
    bEdge.setFromTriplets(tripletsEdge.begin(), tripletsEdge.end());

    std::cout << "Matrix bEdge shape: " << bEdge.rows() << " x " << bEdge.cols()
              << std::endl;
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
    // std::cout << "AtA factorized" << std::endl;
    if (solver.info() != Eigen::Success) {
        std::cerr << "Error in factorization of AtA" << solver.info()
                  << std::endl;
        if (solver.info() == Eigen::NumericalIssue) {
            std::cerr << "Numerical issue, check for your meshes! Are there "
                         "repetitive vertices?"
                      << std::endl;
        } else if (solver.info() == Eigen::InvalidInput) {
            std::cerr << "Invalid input" << std::endl;
        } else if (solver.info() == Eigen::NoConvergence) {
            std::cerr << "No convergence" << std::endl;
        } else if (solver.info() == Eigen::InvalidInput) {
            std::cerr << "Invalid input" << std::endl;
        } else {
            std::cerr << "Unknown error" << std::endl;
        }

        std::cerr << "Matrix A shape: " << A.rows() << " x " << A.cols()
                  << std::endl;
        // std::cerr << "Matrix AtA shape: " << AtA.rows() << " x " <<
        // AtA.cols() << std::endl;
        std::cerr << "Vector r size: " << b.size() << std::endl;
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
        if (v == p0 or v == p1 or
            addition_pinned_vertex_map.find(v->idx) !=
                addition_pinned_vertex_map.end()) {
            continue;
        }
        std::size_t newIdx = 0;
        if (vertex_pin_edges.find(v->idx) != vertex_pin_edges.end()) {
            newIdx = freeIdxTable.at(v->idx);
            auto t = x(newIdx, 0);
            std::cout << "t: " << t << std::endl;
            v->pos[0] = addition_pinned_vertex_map.at(
                            vertex_pin_edges.at(v->idx).first)[0] +
                        t * (addition_pinned_vertex_map.at(
                                 vertex_pin_edges.at(v->idx).second)[0] -
                             addition_pinned_vertex_map.at(
                                 vertex_pin_edges.at(v->idx).first)[0]);
            v->pos[1] = addition_pinned_vertex_map.at(
                            vertex_pin_edges.at(v->idx).first)[1] +
                        t * (addition_pinned_vertex_map.at(
                                 vertex_pin_edges.at(v->idx).second)[1] -
                             addition_pinned_vertex_map.at(
                                 vertex_pin_edges.at(v->idx).first)[1]);
            v->pos[2] = T(0);
        } else {
            newIdx = 2 * freeIdxTable.at(v->idx);
            v->pos[0] = x(newIdx, 0);
            v->pos[1] = x(newIdx + 1, 0);
            v->pos[2] = T(0);
        }
    }
}

//
// Explicit instantiation for T = double using default MeshType and Solver.
// (Add additional instantiations as needed.)
//
template class AngleBasedLSCM<
    double,
    HalfEdgeMesh<double>,
    Eigen::SimplicialLDLT<
        Eigen::SparseMatrix<double>,
        Eigen::Lower,
        Eigen::AMDOrdering<int>>,
    true>;

}  // namespace OpenABF
