#pragma once

#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// OpenABF headers
#include "OpenABF/Exceptions.hpp"
#include "OpenABF/Vec.hpp"

// This header provides a line-sweep algorithm for testing whether a 2D polygon (given by a loop of vertex indices)
// is simple (i.e. non-self-intersecting). It is designed to work with the OpenABF half-edge mesh vertices,
// which store positions in a Vec<T,Dim> accessible via pos[0] (x) and pos[1] (y).

namespace OpenABF {
namespace Geometry {

//--------------------------------------------------------------------
// Local types for the line-sweep algorithm (avoid conflict with mesh Edge)
//--------------------------------------------------------------------
struct SweepEdge {
    std::size_t idx1;  // first vertex index (into the provided vertex array)
    std::size_t idx2;  // second vertex index
    double minX;       // minimum x value of the two endpoints
    double maxX;       // maximum x value of the two endpoints

    SweepEdge(std::size_t i1, std::size_t i2, double x1, double x2)
        : idx1(i1), idx2(i2)
    {
        minX = std::min(x1, x2);
        maxX = std::max(x1, x2);
    }
};

//--------------------------------------------------------------------
// Orientation: Compute the orientation of three vertices.
// Returns > 0 if counterclockwise, 0 if collinear, < 0 if clockwise.
// The vertices are accessed via pointers so that we can use the
// HalfEdgeMesh::Vertex type (accessing pos[0] for x and pos[1] for y).
//--------------------------------------------------------------------
template <typename VertexPtr>
inline double orientation(const VertexPtr& p1, const VertexPtr& p2, const VertexPtr& p3)
{
    return (p2->pos[1] - p1->pos[1]) * (p3->pos[0] - p2->pos[0]) -
           (p2->pos[0] - p1->pos[0]) * (p3->pos[1] - p2->pos[1]);
}

//--------------------------------------------------------------------
// doSegmentsIntersect: Checks whether the segments (p1, p2) and (p3, p4)
// intersect (excluding trivial intersections at consecutive endpoints).
//--------------------------------------------------------------------
template <typename VertexPtr>
inline bool doSegmentsIntersect(const VertexPtr& p1, const VertexPtr& p2,
                                const VertexPtr& p3, const VertexPtr& p4)
{
    double o1 = orientation(p1, p2, p3);
    double o2 = orientation(p1, p2, p4);
    double o3 = orientation(p3, p4, p1);
    double o4 = orientation(p3, p4, p2);

    // If the segments straddle each other, they intersect.
    if ((o1 * o2 < 0) && (o3 * o4 < 0))
        return true;

    return false;
}

//--------------------------------------------------------------------
// getYatX: Computes the y-coordinate on a sweep edge at a given x (sweep line)
// using linear interpolation between the two endpoints.
//--------------------------------------------------------------------
template <typename VertexPtr>
inline double getYatX(const SweepEdge& edge, double sweepX, const std::vector<VertexPtr>& verts)
{
    const auto& p1 = verts[edge.idx1];
    const auto& p2 = verts[edge.idx2];
    double dx = p2->pos[0] - p1->pos[0];
    if (std::fabs(dx) < 1e-9)
        return std::min(p1->pos[1], p2->pos[1]);
    double t = (sweepX - p1->pos[0]) / dx;
    return p1->pos[1] + t * (p2->pos[1] - p1->pos[1]);
}

//--------------------------------------------------------------------
// ActiveEdgeComparator: Orders sweep edges (by index into a SweepEdge array)
// according to the y-coordinate of the edge at the current sweep line position.
//--------------------------------------------------------------------
template <typename VertexPtr>
struct ActiveEdgeComparator {
    const std::vector<SweepEdge>* edgesPtr;
    const std::vector<VertexPtr>* vertsPtr;
    double sweepX;

    ActiveEdgeComparator(const std::vector<SweepEdge>* e,
                         const std::vector<VertexPtr>* v,
                         double x)
        : edgesPtr(e), vertsPtr(v), sweepX(x) {}

    bool operator()(std::size_t edgeA, std::size_t edgeB) const
    {
        const SweepEdge& A = (*edgesPtr)[edgeA];
        const SweepEdge& B = (*edgesPtr)[edgeB];
        double yA = getYatX<VertexPtr>(A, sweepX, *vertsPtr);
        double yB = getYatX<VertexPtr>(B, sweepX, *vertsPtr);
        if (std::fabs(yA - yB) < 1e-9)
            return edgeA < edgeB;
        return yA < yB;
    }
};

//--------------------------------------------------------------------
// Event: Represents a sweep line event (edge start or end).
//--------------------------------------------------------------------
struct Event {
    double x;          // x-coordinate of the event
    bool isStart;      // true if this event is the left endpoint of an edge
    std::size_t edgeIndex; // which sweep edge this event belongs to

    Event(double xx, bool start, std::size_t idx)
        : x(xx), isStart(start), edgeIndex(idx) {}
};

//--------------------------------------------------------------------
// isSimplePolygon: Checks whether the polygon defined by the loop (an
// ordered list of vertex indices) is simple (non-self-intersecting).
//
// The function expects a vector of vertex pointers (as defined in your mesh)
// where each vertex has a position accessible via pos[0] and pos[1].
//--------------------------------------------------------------------
template <typename VertexPtr>
bool isSimplePolygon(const std::vector<std::size_t>& loop,
                     const std::vector<VertexPtr>& verts)
{
    // Build sweep edges from the loop (assumed closed: last connects to first)
    std::vector<SweepEdge> edges;
    edges.reserve(loop.size());
    for (std::size_t i = 0; i < loop.size(); ++i) {
        std::size_t j = (i + 1) % loop.size();
        const auto& p1 = verts[loop[i]];
        const auto& p2 = verts[loop[j]];
        edges.emplace_back(loop[i], loop[j], static_cast<double>(p1->pos[0]),
                                                static_cast<double>(p2->pos[0]));
    }

    // Create events for each edge's start (minX) and end (maxX)
    std::vector<Event> events;
    events.reserve(2 * edges.size());
    for (std::size_t e = 0; e < edges.size(); ++e) {
        const auto& edge = edges[e];
        events.emplace_back(edge.minX, true, e);
        events.emplace_back(edge.maxX, false, e);
    }

    // Sort events by x; if equal, start events come before end events.
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        if (std::fabs(a.x - b.x) < 1e-9)
            return (a.isStart && !b.isStart);
        return a.x < b.x;
    });

    // Active set (balanced BST) to store indices of sweep edges currently intersecting the sweep line.
    std::set<std::size_t, ActiveEdgeComparator<VertexPtr>> active(
        ActiveEdgeComparator<VertexPtr>(&edges, &verts, 0.0));

    // Process events in order.
    for (std::size_t i = 0; i < events.size(); ++i) {
        double currentX = events[i].x;
        // Update comparator with current sweep position.
        ActiveEdgeComparator<VertexPtr> comp(&edges, &verts, currentX);
        std::set<std::size_t, ActiveEdgeComparator<VertexPtr>> newActive(comp);
        for (auto eidx : active) {
            newActive.insert(eidx);
        }
        active.swap(newActive);

        const Event& evt = events[i];
        std::size_t edgeIdx = evt.edgeIndex;

        if (evt.isStart) {
            // Insert the new edge and check for intersections with neighbors.
            auto it = active.insert(edgeIdx).first;
            auto above = std::next(it);
            auto below = (it == active.begin()) ? active.end() : std::prev(it);

            if (above != active.end()) {
                if(verts[edges[*it].idx1]->idx == 6 or 
                     verts[edges[*it].idx2]->idx == 6 or 
                     verts[edges[*above].idx1]->idx == 6 or 
                     verts[edges[*above].idx2]->idx == 6){
                      std::cout << "Edge: " << verts[edges[*it].idx1]->idx << " " << verts[edges[*it].idx2]->idx << " " << verts[edges[*above].idx1]->idx << " " << verts[edges[*above].idx2]->idx << std::endl;
                 }
                if (doSegmentsIntersect(verts[edges[*it].idx1],
                                        verts[edges[*it].idx2],
                                        verts[edges[*above].idx1],
                                        verts[edges[*above].idx2])) {
                    return false; // Intersection found
                }
            }
            if (below != active.end()) {
                if (verts[edges[*it].idx1]->idx == 6 or 
                     verts[edges[*it].idx2]->idx == 6 or 
                     verts[edges[*below].idx1]->idx == 6 or 
                     verts[edges[*below].idx2]->idx == 6){
                      std::cout << "Edge: " << verts[edges[*it].idx1]->idx << " " << verts[edges[*it].idx2]->idx << " " << verts[edges[*below].idx1]->idx << " " << verts[edges[*below].idx2]->idx << std::endl;
                 }
                if (doSegmentsIntersect(verts[edges[*it].idx1],
                                        verts[edges[*it].idx2],
                                        verts[edges[*below].idx1],
                                        verts[edges[*below].idx2])) {
                    return false; // Intersection found
                }
            }
        } else {
            // Remove the edge; check if its former neighbors now intersect.
            auto it = active.find(edgeIdx);
            if (it != active.end()) {
                auto above = std::next(it);
                auto below = (it == active.begin()) ? active.end() : std::prev(it);
                if (above != active.end() && below != active.end()) {
                    if(verts[edges[*it].idx1]->idx == 6 or 
                     verts[edges[*it].idx2]->idx == 6 or 
                     verts[edges[*above].idx1]->idx == 6 or 
                     verts[edges[*above].idx2]->idx == 6){
                      std::cout << "Edge: " << verts[edges[*it].idx1]->idx << " " << verts[edges[*it].idx2]->idx << " " << verts[edges[*above].idx1]->idx << " " << verts[edges[*above].idx2]->idx << std::endl;
                 }
                    if (doSegmentsIntersect(verts[edges[*above].idx1],
                                            verts[edges[*above].idx2],
                                            verts[edges[*below].idx1],
                                            verts[edges[*below].idx2])) {
                        return false;
                    }
                }
                active.erase(it);
            }
        }
    }

    // No intersections found; the polygon is simple.
    return true;
}

} // namespace Geometry
} // namespace OpenABF
