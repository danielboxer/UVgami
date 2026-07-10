#ifndef TRIANGLE_HELPER_H
#define TRIANGLE_HELPER_H
// #include "common.hpp"


#include <Eigen/Core>
#include <CGAL/Point_2.h>
#include <CGAL/Triangle_2.h>
#include <vector>
#include <map>
#include <list>
#include <utility>

// CGAL STUFF
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Surface_sweep_2_algorithms.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <CGAL/Bbox_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel       CGAL_Kernel;
typedef CGAL_Kernel::Point_2                                         Point_2;
typedef CGAL::Arr_segment_traits_2<CGAL_Kernel>                      Traits_2;
typedef Traits_2::Curve_2                                       Segment_2;
typedef CGAL_Kernel::Triangle_3                                       Triangle_3;
typedef CGAL_Kernel::Triangle_2                                       Triangle_2;

typedef CGAL::Simple_cartesian<double> kernel;
typedef kernel::Point_3 Point_3S;
typedef kernel::Point_2 Point_2S;
typedef kernel::Segment_3 Segment_3S;
typedef kernel::Triangle_3 Triangle_3S;
typedef kernel::Triangle_2 Triangle_2S;
typedef std::list<Triangle_3S>::iterator TIterator;
typedef CGAL::AABB_triangle_primitive<kernel, TIterator> Primitive;
typedef CGAL::AABB_traits<kernel, Primitive> AABB_triangle_traits;
// typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

// ViewerDataSimple structure (assuming minimal required members based on usage)
struct ViewerDataSimple {
    Eigen::MatrixXd V; // Vertices
    Eigen::MatrixXi F; // Faces

    // Constructor
    ViewerDataSimple(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) : V(V), F(F) {}
};

using namespace std;


extern double total_time_spent_overlap;



// A helper structure to store an undirected edge as a canonical pair (min, max).
struct Edge
{
    int v1;
    int v2;
    
    Edge(int a, int b)
    {
        // Store in a consistent order
        if(a < b)
        {
            v1 = a; 
            v2 = b;
        }
        else
        {
            v1 = b;
            v2 = a;
        }
    }
    
    // We need equality and hash for use in std::unordered_map
    bool operator==(const Edge& other) const
    {
        return (v1 == other.v1 && v2 == other.v2);
    }
};

struct EdgeHash
{
    std::size_t operator()(const Edge& e) const
    {
        // A simple combination hash for two integers
        // (there are better ways, but this is usually sufficient)
        auto h1 = std::hash<int>()(e.v1);
        auto h2 = std::hash<int>()(e.v2);
        // Combine
        // (this is the Boost::hash_combine approach)
        h1 ^= h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

// Function declarations
int orientation(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d r);

bool edgeEdgeIntersection(Eigen::Vector3d p1, Eigen::Vector3d q1, 
                         Eigen::Vector3d p2, Eigen::Vector3d q2, bool allow_common_endpoints = true);

bool onSegment(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d r);

bool pointOnTriangle(Eigen::Vector3d pt, Eigen::Vector3d v1, 
                    Eigen::Vector3d v2, Eigen::Vector3d v3);

bool pointInTriangle(Eigen::Vector3d pt, Eigen::Vector3d v1, 
                    Eigen::Vector3d v2, Eigen::Vector3d v3);


int countEdgeIntersections(const std::vector<Eigen::Vector3d> &T1, const std::vector<Eigen::Vector3d> &T2);

bool checkTriangleTriangleIntersection(std::vector<Eigen::Vector3d> T1, 
                                     std::vector<Eigen::Vector3d> T2, std::pair<int, int>* counts=nullptr);
void barycentricCoordinates(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& C, const Eigen::Vector3d& P, double& alpha, double& beta, double& gamma);

double computeOverlapingTrianglesFast(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<std::pair<int,int>>& overlappingTriangles_return);

int resolveOverlappingTriangles(Eigen::MatrixXd & V,  const Eigen::MatrixXi & F,  std::vector<std::pair<int,int>> & overlappingTriangles);  


#endif
