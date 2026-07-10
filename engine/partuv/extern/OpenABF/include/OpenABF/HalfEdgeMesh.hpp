#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <cmath> // for std::fabs
#include <queue>

#include"OpenABF/Overlap.hpp"

#include "OpenABF/Exceptions.hpp"
#include "OpenABF/Vec.hpp"
#include <easy/profiler.h>

namespace OpenABF
{

namespace traits
{
/** @brief Default HalfEdgeMesh vertex traits */
template <typename T>
struct DefaultVertexTraits {
};

/** @brief Default HalfEdgeMesh edge traits */
template <typename T>
struct DefaultEdgeTraits {
    /** Edge internal angle */
    T alpha{0};
};

/** @brief Default HalfEdgeMesh face traits */
template <typename T>
struct DefaultFaceTraits {
};
}  // namespace traits

/**
 * @brief Compute the internal angles of a face
 *
 * Updates the current angle (DefaultEdgeTraits::alpha) with the internal angles
 * derived from the face's vertex positions. Useful if you want to reset a face
 * after being processed by ABF or ABFPlusPlus.
 *
 * @tparam FacePtr A Face-type pointer implementing DefaultEdgeTraits
 * @throws MeshException If interior angle is NaN or Inf
 */
template <class FacePtr>
void ComputeFaceAngles(FacePtr& face)
{
    for (auto& e : *face) {
        auto ab = e->next->vertex->pos - e->vertex->pos;
        auto ac = e->next->next->vertex->pos - e->vertex->pos;
        e->alpha = interior_angle(ab, ac);
        if (std::isnan(e->alpha) or std::isinf(e->alpha)) {
            auto msg = "Interior angle for edge " + std::to_string(e->idx) +
                       " is nan/inf";

            throw MeshException(msg);
        }
    }
}

/**
 * @brief Compute the internal angles for all faces in a mesh
 *
 * Runs ComputeFaceAngles on all faces in the mesh. Useful if you want to reset
 * a mesh after running through ABF or ABFPlusPlus.
 *
 * @tparam MeshPtr A Mesh-type pointer with faces implementing DefaultEdgeTraits
 */
template <class MeshPtr>
void ComputeMeshAngles(MeshPtr& mesh)
{
    for (auto& f : mesh->faces()) {
        ComputeFaceAngles(f);
    }
}

/** @brief Determines if mesh is open or closed */
template <class MeshPtr>
auto HasBoundary(const MeshPtr& mesh) -> bool
{
    for (const auto& v : mesh->vertices()) {
        if (v->is_boundary()) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Check if a mesh has unreferenced vertices
 *
 * @note This only checks if the vertex is associated with at least one edge.
 * A face is not currently guaranteed.
 */
template <class MeshPtr>
auto HasUnreferencedVertices(const MeshPtr& mesh) -> bool
{
    for (const auto& v : mesh->vertices()) {
        if (v->edges.empty()) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Get a list of unreferenced vertices
 *
 * @note This only checks if the vertex is associated with at least one edge.
 * A face is not currently guaranteed.
 */
template <class MeshPtr>
auto UnreferencedVertices(const MeshPtr& mesh) -> std::vector<std::size_t>
{
    std::vector<std::size_t> indices;
    for (const auto& v : mesh->vertices()) {
        if (v->edges.empty()) {
            indices.emplace_back(v->idx);
        }
    }
    return indices;
}

/** @brief Check if mesh is manifold */
template <class MeshPtr>
auto IsManifold(const MeshPtr& mesh) -> bool
{
    // insert_face won't allow non-manifold edge, so true by default
    // Check vertices for manifold
    for (const auto& v : mesh->vertices()) {
        if (not v->is_manifold()) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Half-edge mesh class
 *
 * A half-edge mesh represents each edge as two oppositely oriented half-edges.
 * There is one half-edge for each face containing the original edge. For
 * example, if two faces share the edge **AB**, this will result in two
 * half-edges, **AB** and **BA**. If an edge **BC** lies on the mesh border
 * (i.e. it is only included in a single face), there will one be a single
 * half-edge created. This data structure makes it possible to easily traverse
 * the edges and faces adjacent to a vertex (the "wheel"), as well as to
 * traverse the edges of each face.
 *
 * For more information, see Chapter 12.1 in "Fundamentals of Computer
 * Graphics, Fourth edition", Marschner and Shirley (2015)
 * \cite marschner2015fundamentals.
 *
 * @tparam T Floating-point type for coordinates
 * @tparam Dim Dimensionality of vertex coordinates
 * @tparam VertexTraits Additional traits for vertices
 * @tparam EdgeTraits Additional traits for edges
 * @tparam FaceTraits Additional traits for face
 */
template <
    typename T,
    std::size_t Dim = 3,
    typename VertexTraits = traits::DefaultVertexTraits<T>,
    typename EdgeTraits = traits::DefaultEdgeTraits<T>,
    typename FaceTraits = traits::DefaultFaceTraits<T>>
class HalfEdgeMesh
{
public:
    /** Pointer type */
    using Pointer = std::shared_ptr<HalfEdgeMesh>;

    struct Vertex;
    struct Edge;
    struct Face;



    /** @brief Vertex pointer type */
    using VertPtr = std::shared_ptr<Vertex>;
    /** @brief Edge pointer type */
    using EdgePtr = std::shared_ptr<Edge>;
    /** @brief Edge pointer type */
    using FacePtr = std::shared_ptr<Face>;


    using EdgeRaw = Edge*;
    using VertRaw = Vertex*;

    static inline bool is_boundary_raw(const EdgeRaw e) noexcept
    { return e->pair == nullptr; }
private:
    /**
     * @brief Iterator for the edges of a face
     *
     * @tparam Const If true, is a const iterator
     */
    template <bool Const = false>
    class FaceIterator
    {
    public:
        /** Difference type */
        using difference_type = std::size_t;
        /** Value type */
        using value_type = EdgePtr;
        /** Pointer type */
        using pointer =
            std::conditional_t<Const, value_type const*, value_type*>;
        /** Reference type */
        using reference =
            std::conditional_t<Const, value_type const&, value_type&>;
        /** Iterator category */
        using iterator_category = std::input_iterator_tag;

        /** Default constructor == End iterator */
        FaceIterator() = default;
        /** Construct from head of triangle and current edge */
        explicit FaceIterator(const EdgePtr& head, const EdgePtr& current)
            : head_{head}, current_{current}
        {
        }

        /** Dereference operator */
        template <bool Const_ = Const>
        std::enable_if_t<Const_, reference> operator*() const
        {
            return current_;
        }

        /** Dereference operator */
        template <bool Const_ = Const>
        std::enable_if_t<not Const_, reference> operator*()
        {
            return current_;
        }

        /** Equality operator */
        auto operator==(const FaceIterator& other) const -> bool
        {
            return current_ == other.current_;
        }
        /** Inequality operator */
        auto operator!=(const FaceIterator& other) const -> bool
        {
            return !(*this == other);
        }
        /** Increment operator */
        auto operator++() -> FaceIterator&
        {
            // Already at end
            if (current_ == nullptr) {
                return *this;
            }

            // Get the next edge
            current_ = current_->next;
            // If back at head, done iterating
            if (current_ == head_) {
                current_ = nullptr;
            }
            return *this;
        }
    private:
        /** Pointer to beginning of face */
        EdgePtr head_;
        /** Current edge pointer */
        EdgePtr current_;
    };

public:
    /** @brief %Vertex type */
    struct Vertex : public VertexTraits {
        /** @brief Default constructor */
        Vertex() = default;

        /** @brief Construct from position values */
        template <typename... Args>
        explicit Vertex(Args... args) : pos{args...}
        {
        }

        /** @brief Construct a new Vertex pointer */
        template <typename... Args>
        static auto New(Args&&... args) -> VertPtr
        {
            return std::make_shared<Vertex>(std::forward<Args>(args)...);
        }

        /**
         * @brief Get the edges of a vertex's wheel
         *
         * @throws MeshException If vertex is a boundary vertex.
         */
        // auto wheel() const -> std::vector<EdgePtr>
        // {
        //     // EASY_FUNCTION();

        //     std::vector<EdgePtr> ret;
        //     auto e = edge;
        //     do {
        //         if (not e->pair) {
        //             throw MeshException(
        //                 "Cannot enumerate wheel of boundary vertex.");
        //         }
        //         ret.push_back(e);
        //         e = e->pair->next;
        //     } while (e != edge);
        //     return ret;
        // }

        auto wheel() const -> const std::vector<EdgePtr>
        {
            // EASY_FUNCTION();
            if (!wheel_cache_valid_) {
                wheel_cache_.clear();
                auto e = edge;
                do {
                    if (!e->pair)
                        throw MeshException("Cannot enumerate wheel of boundary vertex.");
                    wheel_cache_.push_back(e);
                    e = e->pair->next;
                } while (e != edge);
                wheel_cache_valid_ = true;
                // keep wheel_raw_cache_ in sync
                // wheel_raw_cache_.resize(wheel_cache_.size());
                // std::transform(wheel_cache_.begin(), wheel_cache_.end(),
                //             wheel_raw_cache_.begin(),
                //             [](const EdgePtr& p){ return p.get(); });
            }
            return wheel_cache_;
        }


        auto wheel_raw() const -> std::vector<EdgeRaw>
        {
            // EASY_FUNCTION();

            std::vector<EdgeRaw> ret;
            EdgeRaw e = edge.get();
            do {
                if (!e->pair) throw MeshException("Boundary vertex wheel.");
                ret.push_back(e);
                e = e->pair->next.get();
            } while (e != edge.get());
            return ret;
        }

        /** @brief Returns if vertex is on mesh boundary */
        auto is_boundary() const -> bool
        {
            if (is_boundary_cache_valid_) {
                return is_boundary_cache_;
            }   
            auto e = edge;
            do {
                if (not e->pair) {
                    is_boundary_cache_ = true;
                    is_boundary_cache_valid_ = true;
                    return true;
                }
                e = e->pair->next;
            } while (e != edge);
            is_boundary_cache_ = false;
            is_boundary_cache_valid_ = true;
            return false;
        }

        /** @brief Returns if vertex is interior to mesh */
        auto is_interior() const -> bool { return not is_boundary(); }

        /** @brief Returns if vertex is unreferenced */
        auto is_unreferenced() const -> bool { return edges.empty(); }

        /** @brief Returns if vertex is manifold */
        auto is_manifold() const -> bool
        {
            std::size_t boundaryCnt{0};
            for (const auto& e : edges) {
                if (e->is_boundary()) {
                    if (++boundaryCnt > 2) {
                        return false;
                    }
                }
            }
            return boundaryCnt == 0 or boundaryCnt == 2;
        }

        void clear_grad_caches() {
            p1_cache_valid = false;
            p2_cache_valid = false;
            lengrad_cache_valid = false;
        }

        /** Insertion index */
        std::size_t idx{0};
        /** Vertex position */
        Vec<T, Dim> pos;
        /**
         * Pointer to _an_ Edge with this vertex as its head. Note that there
         * may be many such vertices.
         */
        EdgePtr edge;
        /** List of all edges with this vertex as an end point */
        std::vector<EdgePtr> edges;
        std::vector<EdgeRaw> edges_raw;



        mutable std::vector<EdgePtr> wheel_cache_;
        mutable bool wheel_cache_valid_ {false};

        mutable bool is_boundary_cache_ {false};
        mutable bool is_boundary_cache_valid_ {false};


        // grad caches
        T p1_cache{1};
        T p2_cache{1};
        mutable bool p1_cache_valid{false};
        mutable bool p2_cache_valid{false};

        T lengrad_cache{0};
        mutable bool lengrad_cache_valid{false};

        T plan_grad_cache{0};
        mutable bool plan_grad_cache_valid{false};

    };

    /** @brief %Edge type */
    struct Edge : public EdgeTraits {
        /** @brief Construct a new Edge pointer */
        template <typename... Args>
        static auto New(Args&&... args) -> EdgePtr
        {
            return std::make_shared<Edge>(std::forward<Args>(args)...);
        }

        /** @brief Returns if edge is on mesh boundary */
        auto is_boundary() const -> bool { return pair == nullptr; }

        /**
         * @brief This edge's adjacent half-edge
         *
         * If nullptr, this edge is on the boundary of the mesh (no adjacent
         * face).
         */
        EdgePtr pair;
        /** @brief The next edge in this edge's face */
        EdgePtr next;
        /** @brief The edge's vertex */
        VertPtr vertex;
        /** @brief The face containing this edge */
        FacePtr face;
        /** @brief Insertion index */
        std::size_t idx;

        bool is_manifold{true};



        T alpha_grad_cache{0};
        mutable bool alpha_grad_cache_valid{false};

    };

    /** @brief %Face type */
    struct Face : public FaceTraits {
        /** @brief Construct a new Face pointer */
        template <typename... Args>
        static auto New(Args&&... args) -> FacePtr
        {
            return std::make_shared<Face>(std::forward<Args>(args)...);
        }

        /** Face edge iterator type */
        using iterator = FaceIterator<false>;
        /** Face edge const iterator type */
        using const_iterator = FaceIterator<true>;
        /** @brief Returns an iterator over the edges of the face */
        iterator begin() { return iterator{head, head}; }
        /** @brief Returns the end iterator */
        iterator end() { return iterator(); }
        /** @brief Returns an const iterator over the edges of the face */
        const_iterator cbegin() const { return const_iterator{head, head}; }
        /** @brief Returns the const end iterator */
        const_iterator cend() const { return const_iterator(); }
        /** @brief First edge in the face */
        EdgePtr head;
        /** @brief The next face in the mesh */
        FacePtr next;
        /** @brief Insertion index */
        std::size_t idx;
        auto raw_edges() const -> std::array<EdgeRaw,3>
        {
            return { head.get(), head->next.get(), head->next->next.get() };
        }

        T tri_grad_cache{0};
        mutable bool tri_grad_cache_valid{false};
    };

private:
    /** List of vertices */
    std::vector<VertPtr> verts_;
    /** List of faces */
    /** List of edges, indexed by the vertex's insertion index */
    std::multimap<std::size_t, EdgePtr> edges_;

    // NEW: Our vertex-mapping structure
    std::unordered_map<std::size_t, std::size_t> vertex_map_;

public:
    std::vector<FacePtr> faces_;
    /** @brief Default constructor */
    HalfEdgeMesh() = default;

    /** @brief Destructor deallocating all element pointers */
    ~HalfEdgeMesh()
    {
        // Remove smart pointers from all items
        for (auto& v : verts_) {
            v->edge = nullptr;
            v->edges.clear();
        }
        for (auto& e : edges_) {
            e.second->pair = nullptr;
            e.second->next = nullptr;
            e.second->vertex = nullptr;
            e.second->face = nullptr;
        }
        for (auto& f : faces_) {
            f->head = nullptr;
            f->next = nullptr;
        }
        verts_.clear();
        edges_.clear();
        faces_.clear();
    }

    /** @brief Construct a new HalfEdgeMesh pointer */
    template <typename... Args>
    static Pointer New(Args... args)
    {
        return std::make_shared<HalfEdgeMesh>(std::forward<Args>(args)...);
    }

    /**
     * @brief Insert a new vertex
     *
     * Accepts all arguments supported by the Vertex constructor.
     */
    template <typename... Args>
    auto insert_vertex(Args... args) -> std::size_t
    {
        auto vert = Vertex::New(std::forward<Args>(args)...);
        vert->idx = verts_.size();
        verts_.push_back(vert);
        return vert->idx;
    }

    #if 0
    template <class Vector>
    auto insert_face(const Vector& vector) -> std::size_t
    {
        // Make a new face structure
        auto face = Face::New();

        // Iterate over the vertex indices
        std::size_t prevIdx{0};
        EdgePtr prevEdge;
        for (const auto& idx : vector) {
            // Make a new edge
            auto newEdge = Edge::New();
            newEdge->face = face;

            // Set the head edge for this face
            if (not face->head) {
                face->head = newEdge;
            }

            // Get the vertex by index
            auto vert = verts_.at(idx);
            newEdge->vertex = vert;
            vert->edges.push_back(newEdge);
            // std::cout << "push back new edge: " << newEdge->vertex->idx << std::endl;

            if (not vert->edge) {
                vert->edge = newEdge;
            }

            // If there's a previous edge
            if (prevEdge) {
                // Update the previous edge's successor
                prevEdge->next = newEdge;
                vert->edges.push_back(prevEdge);

                // std::cout << "push back edge: " << prevEdge->vertex->idx << std::endl;
                // Try to find a pair for prev edge using this edge's index
                auto pair = find_edge_(idx, prevIdx);
                if (pair) {
                    if (pair->pair) {
                        auto msg = "Resolved edge pair already paired. Edge (" +
                                   std::to_string(prevIdx) + ", " +
                                   std::to_string(idx) + ") is not 2-manifold.";
                        throw MeshException(msg);
                    }
                    prevEdge->pair = pair;
                    pair->pair = prevEdge;
                }
            }

            // Store the edge
            newEdge->idx = edges_.size();
            edges_.emplace(idx, newEdge);

            // Update for the next iteration
            prevIdx = idx;
            prevEdge = newEdge;


        }

        // Link back to the beginning
        prevEdge->next = face->head;
        face->head->vertex->edges.push_back(prevEdge);
        
        // if(std::find(vector.begin(), vector.end(), 920) != vector.end()){
        //     std::cout << "======================================" << std::endl;
        //     for(auto e : verts_.at(920)->edges){
        //         std::cout << e->vertex->idx << " ->  " << e->next->vertex->idx << std::endl;
        //     }
        // }

        // Try to find a pair for final edge using this edge's index
        auto pair = find_edge_(face->head->vertex->idx, prevIdx);
        if (pair) {
            if (pair->pair) {
                auto msg = "Resolved edge pair already paired. Edge (" +
                           std::to_string(prevIdx) + ", " +
                           std::to_string(face->head->vertex->idx) +
                           ") is not 2-manifold. Probably caused by PAMO \n";
                throw MeshException(msg);
            }
            prevEdge->pair = pair;
            pair->pair = prevEdge;
        }

        // Sanity check: edge lengths
        for (const auto& e : *face) {
            if (norm(e->next->vertex->pos - e->vertex->pos) == 0.0) {
                auto msg = "Zero-length edge (" +
                           std::to_string(e->vertex->idx) + ", " +
                           std::to_string(e->next->vertex->idx) + ")";
                throw MeshException(msg);
            }
        }

        // Compute angles for edges in face
        ComputeFaceAngles(face);

        // Give this face an idx and link the previous face with this one
        face->idx = faces_.size();
        if (not faces_.empty()) {
            faces_.back()->next = face;
        }
        faces_.emplace_back(face);
        return face->idx;
    }
    #endif
    /**
     * @brief Insert a face from an ordered list of Vertex indices
     *
     * Accepts an iterable supporting range-based for loops.
     *
     * @throws std::out_of_range If one of the vertex indices is out of bounds.
     * @throws MeshException (1) If one of provided edges is already paired.
     * This indicates that the mesh is not 2-manifold. (2) If an edge has
     * zero length. This means the face has zero area. (3) If an edge's interior
     * angle is NaN or Inf.
     */
    template <class Vector>
    auto insert_face_2(const Vector& vector) -> std::size_t
    {
        // Make a new face structure
        auto face = Face::New();

        // First pass: create and link edges (but do NOT pair them yet).
        EdgePtr prevEdge;
        std::size_t prevIdx{0};
        bool firstEdge = true;

        for (auto idx : vector) {
            // Create new edge
            auto newEdge = Edge::New();
            newEdge->face = face;

            // Set the head of this face if not set
            if (!face->head) {
                face->head = newEdge;
            }

            // Get the vertex by index
            auto vert = verts_.at(idx);
            newEdge->vertex = vert;


            // Link with previous edge
            if (!firstEdge) {
                vert->edges.push_back(prevEdge);
                prevEdge->next = newEdge;
            } else {
                firstEdge = false;
            }

            // Remember for next iteration
            prevEdge = newEdge;
            prevIdx   = idx;
        }

        // Close the loop: link the last edge back to face->head
        prevEdge->next = face->head;
        face->head->vertex->edges.push_back(prevEdge);

        // -------------------------------------
        //  Sanity checks on the newly-created face
        // -------------------------------------

        // (1) Zero-length edge check
        for (auto e : *face) {
            double length = norm(e->next->vertex->pos - e->vertex->pos);
            if (length == 0.0) {
                std::string msg = "Zero-length edge (" +
                                std::to_string(e->vertex->idx) + ", " +
                                std::to_string(e->next->vertex->idx) + ")";
                // std::cout << msg << std::endl;
                add_vertex_mapping(e->vertex, e->next->vertex);

                return -1;  // Or throw, depending on your style
            }
            auto tailIdx = e->vertex->idx;
            auto headIdx = e->next->vertex->idx;
            auto pair = find_edge_(headIdx, tailIdx);
            if (pair) {
                if (pair->pair) {
                    std::string msg = 
                        "Resolved edge pair already paired. Edge (" +
                        std::to_string(tailIdx) + ", " +
                        std::to_string(headIdx) + ") is not 2-manifold. insert_face_2";
                    std::cerr << msg << std::endl;
                    // throw MeshException(msg);
                    // return -1;  // Or throw, depending on your style
                }
            }
        }

        // -------------------------------------
        //  Store edges in edges_ container
        // -------------------------------------
        for (auto e : *face) {
            e->idx = edges_.size();
            auto vert = e->vertex;
            edges_.emplace(vert->idx, e);

            vert->edges.push_back(e);
            vert->edges_raw.push_back(e.get()); 
            if (!vert->edge) {
                vert->edge = e;
            }
        }

        // -------------------------------------
        //  Second pass: find and set pairs
        // -------------------------------------
        // Now that all new edges have been inserted into edges_,
        // we can safely look for pairs in each edge’s (tail->head).
        for (auto e : *face) {
            auto tailIdx = e->vertex->idx;
            auto headIdx = e->next->vertex->idx;

            // Attempt to find the existing “opposite” edge
            // i.e., an edge from headIdx -> tailIdx
            auto pair = find_edge_(headIdx, tailIdx);
            if (pair && pair->is_manifold) {
                if (pair->pair) {
                    pair->pair->pair = nullptr;
                    pair->pair->is_manifold = false;

                    pair->pair = nullptr;
                    pair->is_manifold = false;

                    e->is_manifold = false;
                }else{
                    // Set mutual pairing
                    e->pair   = pair;
                    pair->pair = e;
                }
            }
        }

        // -------------------------------------
        //  Compute angles for edges in face
        // -------------------------------------
        ComputeFaceAngles(face);

        // -------------------------------------
        //  Assign face index and link to mesh
        // -------------------------------------
        face->idx = faces_.size();
        if (!faces_.empty()) {
            faces_.back()->next = face;
        }
        faces_.emplace_back(face);

        return face->idx;
    }





    /**
     * @brief Insert a new face from an ordered list of Vertex indices
     *
     * @throws std::out_of_range If one of the vertex indices is out of bounds.
     * @throws MeshException If one of provided edges is already paired. This
     * indicates that the mesh is not 2-manifold.
     */
    template <typename... Args>
    auto insert_face(Args... args) -> std::size_t
    {
        static_assert(sizeof...(args) >= 3, "Faces require >= 3 indices");
        using Tuple = std::tuple<Args...>;
        using ElemT = typename std::tuple_element<0, Tuple>::type;
        return insert_face_2(std::initializer_list<ElemT>{args...});
    }

    /** @brief Get the list of vertices in insertion order */
    auto vertices() const -> std::vector<VertPtr> { return verts_; }

    /** @brief Get the list of edges in insertion order */
    auto edges() const -> std::vector<EdgePtr>
    {
        std::vector<EdgePtr> edges;
        for (const auto& f : faces_) {
            for (const auto& e : *f) {
                edges.emplace_back(e);
            }
        }
        return edges;
    }

    /** @brief Get the list of faces in insertion order */
    auto faces() const -> std::vector<FacePtr> { return faces_; }


    auto vertices_not_in_any_face(bool fix = false) const -> std::vector<std::size_t>
    {
        std::vector<char> used(verts_.size(), 0);
    
        for (const auto& f : faces_) {
            if (!f || !f->head) continue;
    
            // Use the face const-iterator to walk its 3 half-edges
            for (auto it = f->cbegin(); it != f->cend(); ++it) {
                const auto& e = *it;
                if (e && e->vertex) used[e->vertex->idx] = 1;
                // Defensive: also mark the 'next' head in case of oddities
                if (e && e->next && e->next->vertex) used[e->next->vertex->idx] = 1;
            }
        }
    
        std::vector<std::size_t> missing;
        missing.reserve(verts_.size());
        for (std::size_t i = 0; i < verts_.size(); ++i)
            if (!used[i]) missing.push_back(i);
    }
    


    /** @brief Get the list of interior vertices in insertion order */
    auto vertices_interior() const -> std::vector<VertPtr>
    {
        std::vector<VertPtr> ret;
        std::copy_if(
            verts_.begin(), verts_.end(), std::back_inserter(ret),
            [](auto x) { return not x->is_boundary(); });
        return ret;
    }

    /** @brief Get the list of boundary vertices in insertion order */
    auto vertices_boundary() const -> std::vector<VertPtr>
    {
        std::vector<VertPtr> ret;
        std::copy_if(
            verts_.begin(), verts_.end(), std::back_inserter(ret),
            [](auto x) { return x->is_boundary(); });
        return ret;
    }

    auto vertices_outer_boundary() const -> std::vector<VertPtr>
    {
        std::vector<VertPtr> ret;
        std::vector<std::vector<std::size_t>> loops = find_boundary_loops();
        if (loops.empty()) {
            throw MeshException("No boundary loops found");
        }

        double max_length = 0;
        auto maxLoopIdx = find_longest_boundary_loop(loops, &max_length);
        std::vector<std::size_t> loop = loops[maxLoopIdx];

        for (const auto& idx : loop) {
            ret.push_back(verts_[idx]);
        }
        return ret;

    }

    /** @brief Get the number of vertices */
    auto num_vertices() const -> std::size_t { return verts_.size(); }

    /** @brief Get the number of interior vertices */
    auto num_vertices_interior() const -> std::size_t
    {
        return std::accumulate(
            verts_.begin(), verts_.end(), std::size_t{0}, [](auto a, auto b) {
                return a + static_cast<std::size_t>(not b->is_boundary());
            });
    }

    /** @brief Get the number of edges */
    auto num_edges() const -> std::size_t { return edges_.size(); }

    /** @brief Get the number of faces */
    auto num_faces() const -> std::size_t { return faces_.size(); }
    /**
     * @brief Read-only access to the entire vertex_map_, if desired.
     */
    auto get_vertex_map() const
        -> const std::unordered_map<std::size_t, std::size_t>&
    {
        return vertex_map_;
    }


    void add_vertex_mapping(const VertPtr& a, const VertPtr& b)
    {
        std::size_t finalA = a->idx;
        std::size_t finalB = get_final_vertex(b->idx);
        if (finalA != finalB) {
            vertex_map_[finalA] = finalB;
        }
        auto previous_keys = get_previous_vertex(finalA);
        for(auto key : previous_keys){
            vertex_map_[key] = finalB;
        }
    }

    // --------------------------------------------------
    // Ear Clipping helper data structure:

    struct EarNode
    {
        // Indices into 'loop'
        int idx;          // Which polygon vertex (in the "loop" ordering) this node represents
        int prev, next;   // Pointers to previous/next in the polygon ring

        double angle;     // For the min-heap ordering

        // We'll keep a small operator< so we can store them in a priority queue
        // by ascending angle (just like the param code).
        bool operator>(const EarNode &rhs) const {
            return angle > rhs.angle;
        }
    };

    // --------------------------------------------------
    // Compute the "interior angle" at vertex i in the polygon ring. In 3D, you can
    // approximate this by looking at the angle between the two edges that meet at i.
    // For a manifold polygon lying roughly in a plane, this is often enough.
    //
    // If your polygons are very 3D, you may want to project them onto a best-fit plane
    // first or use some more robust method. We'll do a simple 3D angle measure.

    template <typename VertexPtr>
    auto compute_vertex_angle(const VertexPtr& v, const double current_angle) -> decltype(v->edge->alpha)
    {
        double  sum = 0; 
        // auto v = mesh.verts_[loop[i]];

        std::unordered_set<std::size_t> processedFaces;
        // std::cout << "vertex idx: " << v->idx << std::endl;
        for (const auto &edge : v->edges) {
            // Check that the edge is associated with a face and that we haven't already processed it.
            if (edge->face && edge->vertex->idx == v->idx && processedFaces.find(edge->face->idx) == processedFaces.end()) {
                sum += edge->alpha;
                // std::cout << "face idx: " << edge->face->idx << "with angle: " << edge->alpha << std::endl;
                processedFaces.insert(edge->face->idx);
            }
        }

        sum += current_angle;
        return current_angle * 2 * M_PI / sum;

        // return M_PI - sum;
    }


    // --------------------------------------------------
    // The actual ear‐clip routine. It returns a list of triangles (each is
    // array of 3 vertex indices). This code is a simplified version
    // reminiscent of Blender's approach, but with a priority queue of angles
    // to pick "ears" from smallest angle to largest.

    template <class Mesh>
    std::vector<std::array<std::size_t, 3>>
    triangulate_ear_clip(const std::vector<std::size_t> &loop, const Mesh &mesh)
    {
        std::vector<std::array<std::size_t, 3>> triangles;

        // If the polygon has < 3 vertices, nothing to do
        if (loop.size() < 3) {
            return triangles;
        }
        if (loop.size() == 3) {
            // Already a triangle
            triangles.push_back({loop[0], loop[1], loop[2]});
            return triangles;
        }

        // Build a doubly‐linked ring of "EarNode"
        // Each node knows its index in "loop" plus 'prev' and 'next.'
        std::vector<EarNode> nodes(loop.size());
        for (int i = 0; i < (int)loop.size(); ++i) {
            nodes[i].idx  = i;
            nodes[i].prev = (i == 0) ? (int)loop.size() - 1 : i - 1;
            nodes[i].next = (i == (int)loop.size() - 1) ? 0 : i + 1;
        }

        // Compute initial angles & store in a min‐heap
        // We'll use std::priority_queue with a custom comparator that sorts smallest angle first.
        auto cmp = [](const EarNode &a, const EarNode &b) {
            return a.angle > b.angle; // min-heap
        };
        std::priority_queue<EarNode, std::vector<EarNode>, decltype(cmp)> heap(cmp);

        // Fill each node's angle
        for (int i = 0; i < (int)loop.size(); ++i) {
            int iPrev = nodes[i].prev;
            int iNext = nodes[i].next;

            double current_angle = interior_angle(mesh.verts_[loop[iPrev]]->pos - mesh.verts_[loop[i]]->pos, mesh.verts_[loop[iNext]]->pos - mesh.verts_[loop[i]]->pos);
            double angle = compute_vertex_angle(mesh.verts_[loop[i]], current_angle);


            nodes[i].angle = angle;
            heap.push(nodes[i]);
        }
        std::priority_queue<EarNode, std::vector<EarNode>, decltype(cmp)> heap_copy = heap;

// while (!heap_copy.empty()) {
//     EarNode top = heap_copy.top();
//     heap_copy.pop();
//     std::cout << "Init Heap: " << loop[top.idx] << "(" << top.angle << ") ";
// }
        // We also keep track of which nodes are "active" in the polygon
        std::vector<bool> active(loop.size(), true);

        int remaining = (int)loop.size();

        // Keep clipping until we have fewer than 3 vertices left
        while (remaining > 3 && !heap.empty()) {

            // Pop the node with the smallest angle
            EarNode top = heap.top();
            heap.pop();

            int i   = top.idx;
            int iPrev = nodes[i].prev;
            int iNext = nodes[i].next;

            // If it's no longer active or changed neighbors, skip
            if (!active[i] || !active[iPrev] || !active[iNext]) {
                continue;
            }
            // Also re-check neighbors because the topology might have changed
            // since we first computed the angle
            if (nodes[i].prev != iPrev || nodes[i].next != iNext) {
                // Means we have old data in the heap
                continue;
            }

            // Check if it's still a valid ear
            // bool is_ear = is_ear_candidate(i, iPrev, iNext, loop, mesh);
            // std::cout << "triangle: " << loop[iPrev] << " " << loop[i] << " " << loop[iNext] << " is_ear: " << is_ear << std::endl;
            // if (is_ear_candidate(i, iPrev, iNext, loop, mesh)) {
                // Output the triangle: we want actual vertex indices from "loop"
                triangles.push_back({ loop[(size_t)iPrev],
                                    loop[(size_t)i],
                                    loop[(size_t)iNext] });
                // std::cout << "Triangle: " << loop[iPrev] << " " << loop[i] << " " << loop[iNext] << std::endl;
                // "Clip" i out of the polygon ring
                active[i] = false;
                nodes[iPrev].next = iNext;
                nodes[iNext].prev = iPrev;
                --remaining;

                // Recompute angles of the neighbors iPrev and iNext
                if (remaining > 2) {
                    // iPrev
                    int pPrev = nodes[iPrev].prev;
                    int pNext = nodes[iPrev].next;
                    // nodes[iPrev].angle = compute_vertex_angle(iPrev, pPrev, pNext, loop, mesh);
                    double p_current_angle = interior_angle(mesh.verts_[loop[iPrev]]->pos - mesh.verts_[loop[i]]->pos, mesh.verts_[loop[iNext]]->pos - mesh.verts_[loop[i]]->pos);

                    nodes[iPrev].angle = compute_vertex_angle(mesh.verts_[loop[iPrev]], p_current_angle);
                    if (active[iPrev]) {
                        heap.push(nodes[iPrev]);
                    }

                    // iNext
                    int nPrev = nodes[iNext].prev;
                    int nNext = nodes[iNext].next;
                    // nodes[iNext].angle = compute_vertex_angle(iNext, nPrev, nNext, loop, mesh);
                    double n_current_angle = interior_angle(mesh.verts_[loop[iPrev]]->pos - mesh.verts_[loop[i]]->pos, mesh.verts_[loop[iNext]]->pos - mesh.verts_[loop[i]]->pos);

                    nodes[iNext].angle = compute_vertex_angle(mesh.verts_[loop[iNext]], n_current_angle);
                    if (active[iNext]) {
                        heap.push(nodes[iNext]);
                    }

                    // print heap

                    std::priority_queue<EarNode, std::vector<EarNode>, decltype(cmp)> heap_copy = heap;
// while (!heap_copy.empty()) {
//     EarNode top = heap_copy.top();
//     heap_copy.pop();
//     std::cout << "Heap: " <<  loop[(size_t)top.idx] << "(" << top.angle << ") ";
// }
// std::cout << std::endl;
                }
        }

        // If we have exactly 3 left, output that final triangle
        if (remaining == 3) {
            // Find the triple of active nodes
            std::array<int,3> left;
            int idxA = -1;
            for (int i = 0, c=0; i < (int)loop.size(); ++i) {
                if (active[i]) {
                    left[c++] = i;
                    if (c == 3) break;
                }
            }
            // std::cout << "Triangle: "  << loop[left[0]] << " " << loop[left[1]] << " " << loop[left[2]] << std::endl;

            // left[0], left[1], left[2] -> final triangle
            triangles.push_back({ loop[(size_t)left[0]],
                                loop[(size_t)left[1]],
                                loop[(size_t)left[2]] });
        }

        return triangles;
    }


    /**
     * @brief Find all boundary loops (holes) in the mesh
     *
     * @return std::vector<std::vector<std::size_t>> List of vertex indices for each hole
     */
    std::vector<std::vector<std::size_t>> find_boundary_loops() const {
        std::vector<std::vector<std::size_t>> loops;
        std::unordered_set<std::size_t> processed_edges;


        for (const auto& edge_entry : edges_) {
            auto edge = edge_entry.second;
            // Skip non-boundary or already processed edges
            if (!edge->is_boundary() || processed_edges.count(edge->idx)) {
                continue;
            }

            std::vector<std::size_t> loop;
            auto current_edge = edge;

            // std::cout << "Finding Boundary loop from Edge: " << edge->vertex->idx << std::endl;
            
            do {
                loop.push_back(current_edge->vertex->idx);
                processed_edges.insert(current_edge->idx);

                // Find next boundary edge in the loop
                EdgePtr next_edge = nullptr;
                // Find next boundary edge by checking each condition separately for debugging
                std::vector<EdgePtr> potential_edges;
                for (const auto& e : current_edge->vertex->edges) {
                    // Debug: Print edge information
                    // std::cout << "Edge candidate: " << e->vertex->idx << " -> ";
                    if (e->next) {
                        // std::cout << e->next->vertex->idx;
                    } else {
                        // std::cout << "null";
                    }

                    if(e->vertex->idx == current_edge->vertex->idx){
                        // std::cout << " --> same vertex";
                        continue;
                    }
                    
                    // Check boundary condition
                    bool is_boundary_edge = e->is_boundary();
                    // std::cout << ", is_boundary: " << (is_boundary_edge ? "true" : "false");
                    
                    // Check if already processed
                    bool is_processed = processed_edges.count(e->idx) > 0;
                    // std::cout << ", already_processed: " << (is_processed ? "true" : "false");
                    
                    // Final decision
                    if (is_boundary_edge && !is_processed && e->vertex->idx != current_edge->vertex->idx) {
                        // std::cout << " --> SELECTED as next boundary edge" << std::endl;
                        potential_edges.push_back(e);
                    }else{
                        // std::cout << " --> skipped" << std::endl;
                    }
                    
                }

                if (potential_edges.size() == 1) {
                    next_edge = potential_edges[0];
                } else if (potential_edges.size() > 1) {
                    // std::cout << "Multiple potential edges found" << std::endl;
                    for (const auto& e : potential_edges) {
                        if(! boundary_edges_same_side(e, current_edge)){
                            next_edge = e;
                            break;
                        }
                        // std::cout << "Potential edge: " << e->vertex->idx << " -> " << e->next->vertex->idx << std::endl;
                    }
                    if (!next_edge) {
                        std::cout << "WARNING: No edge found on the different side? " << std::endl;
                        next_edge = potential_edges[0];
                    }
                    // next_edge = potential_edges[0];
                }

                if (!next_edge) {
                    break; // In case of inconsistency
                }
                current_edge = next_edge;
            } while (current_edge != edge);

            if (loop.size() >= 3) {
                loops.emplace_back(loop);
            }else{
                // std::cout << "Loop size: " << loop.size() << std::endl;
            }
        }
        return loops;
    }
    /**
     * @brief Triangulate a polygon using fan triangulation
     *
     * @param loop Ordered list of vertex indices forming the polygon
     * @return std::vector<std::array<std::size_t, 3>> List of triangles
     */
    std::vector<std::array<std::size_t, 3>> triangulate_fan(
        const std::vector<std::size_t>& loop) {
            if (loop.size() < 3) {
                return {};
            }
        std::vector<std::array<std::size_t, 3>> triangles;
        for (std::size_t i = 1; i < loop.size() - 1; ++i) {
            triangles.push_back({loop[0], loop[i], loop[i + 1]});
        }
        return triangles;
    }

/**
     * @brief Fill all holes in the mesh using fan triangulation
     *
     * @throws MeshException If triangulation fails (e.g., non-manifold edges)
     */

    bool has_overlap(bool* multiple_loops){
        auto loops = find_boundary_loops();
        if (loops.empty()) {
            return false;
        }
        for (const auto& loop : loops) {
            std::cout << "Boundary loop: ";
            for (const auto& idx : loop) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        }

        // if (loops.size() > 1) {
        //     *multiple_loops = true;
        // }
        // std::vector<float> loop_lengths;
        // for (const auto& loop : loops) {
        //     float length = 0.0f;
        //     for (std::size_t i = 0; i < loop.size(); ++i) {
        //         std::size_t j = (i + 1) % loop.size();
        //         const auto& v_i = verts_.at(loop[i])->pos;
        //         const auto& v_j = verts_.at(loop[j])->pos;
        //         length += norm(v_j - v_i); // Calculate edge length
        //     }
        //     loop_lengths.emplace_back(length);
        // }
        // for (const auto& loop : loops) {
        //     std::cout << "Ring loop: ";
        //     for (const auto& idx : loop) {
        //         std::cout << idx << " ";
        //     }
        //     std::cout << std::endl;
        // }
    
        // 3. Find the longest loop (outer boundary)
        // auto max_it = std::max_element(loop_lengths.begin(), loop_lengths.end());
        // std::size_t max_idx = std::distance(loop_lengths.begin(), max_it);
        // std::vector<std::size_t> outer_loop = loops[max_idx];

        double max_length = 0;
        auto maxLoopIdx = find_longest_boundary_loop(loops, &max_length);
        std::vector<std::size_t> outer_loop = loops[maxLoopIdx];
        return !Geometry::isSimplePolygon(outer_loop, verts_);

    }
        // 2 & 3. Find the longest boundary loop
    std::size_t find_longest_boundary_loop(std::vector<std::vector<std::size_t>>& loops, double* max_length) const {
        // 2. Compute the length of each loop
        std::vector<float> loop_lengths;
        for (const auto& loop : loops) {
            float length = 0.0f;
            for (std::size_t i = 0; i < loop.size(); ++i) {
                std::size_t j = (i + 1) % loop.size();
                const auto& v_i = verts_.at(loop[i])->pos;
                const auto& v_j = verts_.at(loop[j])->pos;
                length += norm(v_j - v_i); // Calculate edge length
            }
            loop_lengths.emplace_back(length);
        }
        
        // 3. Find the longest loop (outer boundary)
        auto max_it = std::max_element(loop_lengths.begin(), loop_lengths.end());
        std::size_t max_idx = std::distance(loop_lengths.begin(), max_it);
        if (max_length) {
            *max_length = *max_it;
        }
        return max_idx;
    }

    void fill_holes() {
        // 1. Find all boundary loops
        auto loops = find_boundary_loops();
    
        if (loops.empty()) {
            return;
        }

        // Call the function where needed
        double max_length = 0;
        std::size_t max_idx = find_longest_boundary_loop(loops, &max_length);
        
        // print the loops
        // for (const auto& loop : loops) {
        //     std::cout << "Boundary loop: ";
        //     for (const auto& idx : loop) {
        //         std::cout << idx << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // 4. Fill smaller loops (holes)
        for (std::size_t i = 0; i < loops.size(); ++i) {
            if (i != max_idx) {
                // auto triangles = triangulate_fan(loops[i]);
                auto triangles = triangulate_ear_clip(loops[i], *this);

                
                for (const auto& tri : triangles) {
                    // std::cout << "[Triangle List]: " << tri[0] << " " << tri[1] << " " << tri[2] << std::endl;
                    insert_face(tri[0], tri[1], tri[2]); // Add new faces
                    // try {
                    // } catch (const MeshException& e) {
                        
                    // }
                }
            }
        }
    }
private:
    /** Find an existing edge with the provided end points */
    auto find_edge_(std::size_t start, std::size_t end) -> EdgePtr
    {
        // Get edges with this start index
        auto range = edges_.equal_range(start);

        // Loop over potential edges
        for (auto it = range.first; it != range.second; it++) {
            const auto& e = it->second;
            if (e->next and e->next->vertex->idx == end) {
                return e;
            }
        }
        return nullptr;
    }

    // Helper function for vertex_map_
    auto get_previous_vertex(std::size_t v) -> std::vector<std::size_t>
    {

        std::vector<std::size_t> keys;
        for (const auto& pair : vertex_map_) {
            if (pair.second == v) {
                keys.push_back(pair.first);
            }
        }
        return keys;
    }
    auto get_final_vertex(std::size_t v) -> std::size_t
    {
        auto it = vertex_map_.find(v);
        if (it == vertex_map_.end()) {
            return v;
        }
        std::size_t rep = get_final_vertex(it->second);
        vertex_map_[v]  = rep;
        return rep;
    }

    bool boundary_edges_same_side(const EdgePtr& edge_a, const EdgePtr& edge_b) const
    {
        if (!edge_a->is_boundary()) {
            std::cout << "Error: edge_a must be boundary" << std::endl;
            return false;
        }
        if (!edge_b->is_boundary()) {
            std::cout << "Error: edge_b must be boundary" << std::endl;
            return false;
        }

        EdgePtr currentEdge = edge_a;
        do {
            if (currentEdge->next == edge_b) {
                return true;
            }
            currentEdge = currentEdge->next->pair;
        } while (currentEdge);

        currentEdge = edge_b;
        do {
            if (currentEdge->next == edge_a) {
                return true;
            }
            currentEdge = currentEdge->next->pair;
        } while (currentEdge);
        return false;
    }



};
}  // namespace OpenABF



namespace OpenABF
{

template<class MeshPtr>
using EdgePtrOf = typename std::remove_reference_t<decltype(*std::declval<MeshPtr>())>::EdgePtr;

template<std::size_t Dim>
struct QPos {
    std::array<long long,Dim> q;
    bool operator==(const QPos& o) const noexcept { return q == o.q; }
    bool operator< (const QPos& o) const noexcept { return q <  o.q; } // lexicographic
};

struct CombHash {
    template<typename T>
    std::size_t operator()(const T& x) const noexcept
    {
        return std::hash<T>{}(x);
    }
};

template<std::size_t Dim>
struct QPosHash {
    std::size_t operator()(const QPos<Dim>& p) const noexcept
    {
        std::size_t h = 0;
        for(auto v : p.q) h ^= CombHash{}(v) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};

template<std::size_t Dim>
struct EdgeKey {
    QPos<Dim> a, b;          // a ≤ b
    bool operator==(const EdgeKey& o) const noexcept
    { return a==o.a && b==o.b; }
};

template<std::size_t Dim>
struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey<Dim>& k) const noexcept
    {
        return QPosHash<Dim>{}(k.a) ^ (QPosHash<Dim>{}(k.b)<<1);
    }
};

// --- quantise a Vec<T,Dim> to integer grid ----------------------------------
template<typename VecT, std::size_t Dim>
static QPos<Dim> quantise(const VecT& p, double eps)
{
    QPos<Dim> qp;
    for(std::size_t i=0;i<Dim;++i)
        qp.q[i] = static_cast<long long>(std::llround(p[i]/eps));
    return qp;
}

// -----------------------------------------------------------------------------
//  ε defaults to 1e-9 * longest edge of the parent mesh (if not supplied).
// -----------------------------------------------------------------------------
template<class MeshPtrParent, class MeshPtrSub>
void propagate_alpha_by_position(const MeshPtrParent& parent,
                                 const MeshPtrSub&    sub,
                                 double eps = 1e-6)
{
    using ParentEdgePtr = EdgePtrOf<MeshPtrParent>;
    using SubEdgePtr    = EdgePtrOf<MeshPtrSub>;
    constexpr std::size_t Dim = 3;

    // --- choose ε automatically if requested --------------------------------
    if (eps < 0.0) {
        eps = 0.0;
        for (const auto& e : parent->edges())
            for(std::size_t k=0;k<Dim;++k)
                eps = std::max(eps,
                               std::fabs(e->vertex->pos[k] - e->next->vertex->pos[k]));
        eps *= 1e-9;                      // 1 nm of the longest edge
        if (eps == 0.0) eps = 1e-12;      // degenerate-mesh safeguard
    }

    // --- 1. build LUT : EdgeKey → alpha (parent) ----------------------------
    std::unordered_map<EdgeKey<Dim>, decltype(ParentEdgePtr()->alpha),
                       EdgeKeyHash<Dim>> lut;
    lut.reserve(parent->num_edges());

    // std::cout << "parent->num_edges(): " << parent->num_edges() << std::endl;

    for (const auto& e : parent->edges()) {
        auto p = quantise<decltype(e->vertex->pos),Dim>(e->vertex->pos,       eps);
        auto q = quantise<decltype(e->next->vertex->pos),Dim>(e->next->vertex->pos, eps);
        EdgeKey<Dim> key = (p < q) ? EdgeKey<Dim>{p,q} : EdgeKey<Dim>{q,p};
        lut.emplace(key, e->alpha);           // later duplicates ignored
    }


    // --- 2. transfer alpha to every sub-mesh edge ---------------------------
    for (const auto& eSub : sub->edges()) {
        auto p = quantise<decltype(eSub->vertex->pos),Dim>(eSub->vertex->pos,       eps);
        auto q = quantise<decltype(eSub->next->vertex->pos),Dim>(eSub->next->vertex->pos, eps);

        EdgeKey<Dim> key = (p < q) ? EdgeKey<Dim>{p,q} : EdgeKey<Dim>{q,p};


        auto it = lut.find(key);
        if (it == lut.end()){
            // throw MeshException("Edge defined by given positions not found in parent mesh");
            std::cout << "Edge defined by given positions not found in parent mesh" << std::endl;
            continue;
        }

        eSub->alpha = it->second;             // overwrite trait

    }
}

}