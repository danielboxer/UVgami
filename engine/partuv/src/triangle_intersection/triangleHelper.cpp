//2D Triangle-Triangle collisions in C++
//Release by Tim Sheerman-Chase 2016 under CC0
// https://gist.github.com/TimSC/5ba18ae21c4459275f90

// Modified by Manas Bhargava 17/12/2021
#include "triangleHelper.hpp"

#include <CGAL/box_intersection_d.h>
#include <igl/boundary_loop.h>

#include <easy/profiler.h>

typedef CGAL::Box_intersection_d::Box_d<double,2> Box;


double total_time_spent_overlap = 0.0;
// double computeOverlapingTrianglesFast2(const igl::opengl::ViewerData & mesh, std::vector<std::pair<int,int>> &overlappingTriangles_return) {
//     // mesh here is the 2D sheet mesh. and has the tree datastructure attached to it. 
//     // let's exploit it!

// }


// Global container for edge counts.
static std::unordered_map<Edge, int, EdgeHash> g_edgeCounts;

// Call this once (e.g., during initialization) when F is available.
void initializeEdgeCounts(const Eigen::MatrixXi& F) {
    g_edgeCounts.clear();
    for (int i = 0; i < F.rows(); ++i) {
        int f0 = F(i, 0), f1 = F(i, 1), f2 = F(i, 2);
        g_edgeCounts[Edge(f0, f1)]++;
        g_edgeCounts[Edge(f1, f2)]++;
        g_edgeCounts[Edge(f2, f0)]++;
    }
}

bool isBoundaryEdge(int a, int b) {
    Edge edge(a, b);
    auto it = g_edgeCounts.find(edge);
    return (it != g_edgeCounts.end() && it->second == 1);
}
/* Overlap Handling */

Eigen::Vector3d projectPointOutsideEdge(const Eigen::Vector3d &point, 
                                          const Eigen::Vector3d &edgeStart, 
                                          const Eigen::Vector3d &edgeEnd,
                                          double eps = 0) {
    // Compute the vector along the edge    
    Eigen::Vector3d edgeVector = edgeEnd - edgeStart;

    // Compute the projection scalar of (point - edgeStart) onto the edge
    double t = (point - edgeStart).dot(edgeVector) / edgeVector.dot(edgeVector);

    // Compute the projection point on the edge
    t = std::max(0.0, std::min(1.0, t));

    Eigen::Vector3d proj = edgeStart + t * edgeVector;

    Eigen::Vector3d outward;
    if ((proj - point).norm() == 0) {

        outward = Eigen::Vector3d(-edgeVector.y(), edgeVector.x(), 0);
        if (outward.norm() != 0) {
            outward.normalize();
        }
        std::cout << "Warning: point is exactly on the edge." << std::endl;
    } else {
        outward = (proj - point).normalized();
    }

    // Return the projection point offset by eps in the outward direction
    return proj + eps * outward;
}



bool resolveOverlapForTriangle(vector<Eigen::Vector3d> &T1, 
                               const vector<Eigen::Vector3d> &T2) {
    bool changed = false;
    int originalEcount = countEdgeIntersections(T1, T2);
    // Loop over each vertex of T1
    for (int i = 0; i < 3; i++) {
        if (pointInTriangle(T1[i], T2[0], T2[1], T2[2])) {
            bool resolved = false;
            // T2's edges: (T2[0],T2[1]), (T2[1],T2[2]), (T2[2],T2[0])
            for (int j = 0; j < 3; j++) {
                int j_next = (j + 1) % 3;
                // Compute candidate projection of T1[i] onto the j-th edge of T2
                Eigen::Vector3d candidate = projectPointOutsideEdge(T1[i], T2[j], T2[j_next]);
                // Build a candidate triangle by replacing T1[i] with candidate
                vector<Eigen::Vector3d> candidateT1 = T1;
                candidateT1[i] = candidate;
                // Check if this candidate resolves the intersection
                if (countEdgeIntersections(candidateT1, T2) < originalEcount) {
                    T1[i] = candidate;
                    resolved = true;
                    changed = true;
                    break; // Exit the edge loop for this vertex once resolved
                }
            }
            // Optionally, if no edge projection resolves the overlap, one might use a fallback.
            if (!resolved) {
                cout << "Warning: vertex " << i << " could not be resolved by projection." << endl;
            }
        }
    }
    return changed;
}




int handleSpecialIntersection(std::vector<Eigen::Vector3d>& T1,
                               std::vector<Eigen::Vector3d>& T2,
                               double eps =0)
{
    // Count the number of intersections each vertex is involved in.
    // (Each edge-edge intersection adds one to each of its endpoints.)
    std::vector<int> countT1(3, 0);
    std::vector<int> countT2(3, 0);
    
    for (int i = 0; i < 3; ++i) {
        int i_next = (i + 1) % 3;
        for (int j = 0; j < 3; ++j) {
            int j_next = (j + 1) % 3;
            if (edgeEdgeIntersection(T1[i], T1[i_next], T2[j], T2[j_next])) {
                countT1[i]++; countT1[i_next]++; countT2[j]++; countT2[j_next]++;
            }
        }
    }
    // In the special case exactly 4 intersections occur.
    // The "common" vertex on a triangle is the one involved in all four intersections.
    int commonT1 = -1;
    for (int i = 0; i < 3; ++i) {
        if (countT1[i] >= 3) {
            commonT1 = i;
            break;
        }
    }
    int commonT2 = -1;
    for (int j = 0; j < 3; ++j) {
        if (countT2[j] >= 3) {
            commonT2 = j;
            break;
        }
    }
    // Debug: print countT1 and countT2

    std::cout << std::endl;
    if (commonT1 == -1 || commonT2 == -1)
        return -1;
    Eigen::Vector3d mid = (T1[commonT1] + T2[commonT2]) * 0.5;
    
    // For each triangle, move its common vertex towards the midpoint by a distance of eps.
    // (If the vertex is closer than eps, we just snap it to the midpoint.)
    
    Eigen::Vector3d diff1 = mid - T1[commonT1];
    double dist1 = diff1.norm();
    T1[commonT1] += (diff1.normalized() * (dist1 + eps));
    
    
    Eigen::Vector3d diff2 = mid - T2[commonT2];
    double dist2 = diff2.norm();
    T2[commonT2] += (diff2.normalized() * (dist2 + eps));

    return 0;
}



template <typename VecType>
bool resolveOverlapForTriangle_Local(std::vector<VecType> &T1,
                                     std::vector<VecType> &T2,
                                   const Eigen::RowVectorXi &F1,
                                   const Eigen::RowVectorXi &F2
                                    ) {
    // Optionally add a static_assert if you want to restrict VecType to 2D or 3D:
    static_assert(VecType::RowsAtCompileTime == 2 || VecType::RowsAtCompileTime == 3,
                  "VecType must be either 2D or 3D.");

    auto printTriangles = [&]()
    {
        std::cout << "Triangle T0: ";
        for (const auto &v : T1)
            std::cout << "(" << std::fixed << std::setprecision(6)
                    << v(0) << ", " << v(1) << ", " << v(2) << ") ";
        std::cout << "\nTriangle T1: ";
        for (const auto &v : T2)
            std::cout << "(" << std::fixed << std::setprecision(6)
                    << v(0) << ", " << v(1) << ", " << v(2) << ") ";
        std::cout << std::endl;
    };
          
    printTriangles();


    std::vector<std::pair<int, int>> common;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if ((T1[i] - T2[j]).norm() < 1e-12) {
                common.push_back(std::pair<int, int>(i, j));
            }
        }
    }
    std::cout << "Common vertices found: " << common.size() << std::endl;
    if(checkTriangleTriangleIntersection(T1, T2))
    {
    if (common.size() != 1) {
        std::cout << "Error: common vertices count is not 1." << std::endl;
        std::pair<int, int> counts;
            std::cout << "Warning: local overlap is not a single vertex, throwing to global resort" << std::endl;
            return false;
        }
    }
    else{
        std::cout << "No intersection found, returning" << std::endl;
        return true;
    }
    
    int common_i = common[0].first;
    int common_j = common[0].second;

    std::vector<std::pair<int, int>> i_boundary;
    std::vector<std::pair<int, int>> j_boundary;
    for (int i = 1; i < 3; ++i)
    {
        int i_next = (common_i + i) % 3;
        if (isBoundaryEdge(F1(common_i), F1(i_next)))
        {
            i_boundary.push_back(std::pair<int, int>(common_i, i_next));
        }
    }

    for (int j = 1; j < 3; ++j)
    {
        int j_next = (common_j + j) % 3;
        if (isBoundaryEdge(F2(common_j), F2(j_next)))
        {
            j_boundary.push_back(std::pair<int, int>(common_j, j_next));
        }
    }

    if (i_boundary.size() !=1 || j_boundary.size() !=1)
    {
        std::cout << "Warning: local overlap boundary edge count is not 1, throwing to global resort" << std::endl;
        return false;
    }

    int i_merge = i_boundary[0].second;
    int j_merge = j_boundary[0].second;

    int i_merge_next = (i_merge + 1) % 3 == i_boundary[0].first ? (i_merge + 2) % 3 : (i_merge + 1) % 3;
    int j_merge_next = (j_merge + 1) % 3 == j_boundary[0].first ? (j_merge + 2) % 3 : (j_merge + 1) % 3;


    // if(isBoundaryEdge(F1(i_merge), F1(i_merge_next)) && isBoundaryEdge(F2(j_merge), F2(j_merge_next)))
    if(true)

    {  
        std::cout << "Merging vertices: " << i_merge << " and " << j_merge << std::endl;
        Eigen::Vector3d mid = (T1[i_merge] + T2[j_merge]) * 0.5;

        Eigen::Vector3d diff1 = mid - T1[i_merge];
        double dist1 = diff1.norm();
        T1[i_merge] += (diff1.normalized() * (dist1));

        Eigen::Vector3d diff2 = mid - T2[j_merge];
        double dist2 = diff2.norm();
        T2[j_merge] += (diff2.normalized() * (dist2));

        printTriangles();
        return true;
    }
    // else if (isBoundaryEdge(F1(i_merge), F1(i_merge_next)))
    // {
    //     T1[i_merge] = projectPointOutsideEdge(T1[i_merge], T2[j_merge], T2[common_j]);
    //     printTriangles();
    //     return true;
        
    // }else if (isBoundaryEdge(F2(j_merge), F2(j_merge_next)))
    // {
    //     T2[j_merge] = projectPointOutsideEdge(T2[j_merge], T1[i_merge], T1[common_i]);
    //     printTriangles();
    //     return true;
    // }
    else{
        std::cout << "Warning: Both local overlap triangles have only one boundary edge, this shouldn't happen" << std::endl;
        return false;
    }
}



template <typename VecType>
bool resolveOverlapForTriangle_Edges(std::vector<VecType> &T1,
                                     std::vector<VecType> &T2) {
    // Optionally add a static_assert if you want to restrict VecType to 2D or 3D:
    static_assert(VecType::RowsAtCompileTime == 2 || VecType::RowsAtCompileTime == 3,
                  "VecType must be either 2D or 3D.");

    auto printTriangles = [&]()
    {
        std::cout << "Triangle T0: ";
        for (const auto &v : T1)
            std::cout << "(" << std::fixed << std::setprecision(6)
                    << v(0) << ", " << v(1) << ", " << v(2) << ") ";
        std::cout << "\nTriangle T1: ";
        for (const auto &v : T2)
            std::cout << "(" << std::fixed << std::setprecision(6)
                    << v(0) << ", " << v(1) << ", " << v(2) << ") ";
        std::cout << "global" << std::endl;
    };
          
    printTriangles();



    for (int i = 0; i < 3; ++i)
    {
        int i_next = (i + 1) % 3;
        // Define a lambda to print triangles T1 and T2 once

        for (int j = 0; j < 3; ++j)
        {
            int j_next = (j + 1) % 3;
            if (edgeEdgeIntersection(T1[i], T1[i_next], T2[j], T2[j_next]))
            {
                if (pointInTriangle(T1[i], T2[0], T2[1], T2[2]))
                {
                    VecType candidate = projectPointOutsideEdge(T1[i], T2[j], T2[j_next]);
                    T1[i] = candidate;
                    // printTriangles();
                }
                if (pointInTriangle(T1[i_next], T2[0], T2[1], T2[2]))
                {
                    VecType candidate = projectPointOutsideEdge(T1[i_next], T2[j], T2[j_next]);
                    T1[i_next] = candidate;
                    // printTriangles();
                }
                if (pointInTriangle(T2[j], T1[0], T1[1], T1[2]))
                {
                    VecType candidate = projectPointOutsideEdge(T2[j], T1[i], T1[i_next]);
                    T2[j] = candidate;
                    // printTriangles();
                }
                if (pointInTriangle(T2[j_next], T1[0], T1[1], T1[2]))
                {
                    VecType candidate = projectPointOutsideEdge(T2[j_next], T1[i], T1[i_next]);
                    T2[j_next] = candidate;
                    // printTriangles();
                }
            }
        }
    }
    std::pair<int, int> counts;
    if (checkTriangleTriangleIntersection(T1, T2, &counts)) {
        if (counts.first == 0 && (counts.second == 4 || counts.second == 3)) {
            int success = handleSpecialIntersection(T1, T2);
            return success == 0;
        } else {
            std::cout << "Triangle T0: ";
            for (const auto &v : T1) {
                std::cout << "(" << std::fixed << std::setprecision(6)
                          << v(0) << ", " << v(1) << ", " << v(2) << ") ";
            }
            std::cout << std::endl;
            std::cout << "Triangle T1: ";
            for (const auto &v : T2) {
                std::cout << "(" << std::fixed << std::setprecision(6)
                          << v(0) << ", " << v(1) << ", " << v(2) << ") ";
            }
            std::cout << std::endl;
            std::cout << "Vertex overlap count" << counts.first << " and edge overlap count " << counts.second << std::endl;
            // return false;
        }
    }
    return true;
}

// void resolveOverlapBetweenTriangles(vector<Eigen::Vector3d> &T1, 
//                                       vector<Eigen::Vector3d> &T2) {
//     // Resolve vertices of T1 overlapping T2
//     if(!resolveOverlapForTriangle(T1, T2)) {
//         // Resolve vertices of T2 overlapping T1
//         resolveOverlapForTriangle(T2, T1);
//     }

// }





bool checkLocalOverlap(const Eigen::MatrixXi & F,
                       int ti, int tj) {
    // check if two triangle share a vertex:
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            if(F(ti, i) == F(tj, j)) {
                return true;
            }
        }
    }
    return false;
}

int getSelecetedOverlappingTriangles(const Eigen::MatrixXd & V,
                                const Eigen::MatrixXi & F,
                                const std::vector<std::pair<int,int>> & overlappingTriangles,
                                std::vector<std::pair<int,int>> & selectedTriangleTrianglePair) {

	std::vector<std::vector<int>> triangle2triangleList(F.rows());
	for(auto tp: overlappingTriangles) {
		triangle2triangleList[tp.first].push_back(tp.second);
		triangle2triangleList[tp.second].push_back(tp.first);
	}

	// std::vector<std::pair<int,int>> selectedTriangleTrianglePair; // second best pairs!!
	for(int ti = 0; ti < triangle2triangleList.size(); ti++) {
		auto tvector = triangle2triangleList[ti];
		if(tvector.size() == 1) {
			int tj = tvector[0];
			// printf("Triangle: %d is intersected only once with triangle :%d\n", ti, tj);
			if(triangle2triangleList[tj].size() == 1 and triangle2triangleList[tj][0] == ti) {
				std::pair<int,int> fpair = (ti < tj) ? std::make_pair(ti, tj) : std::make_pair(tj, ti);
				// this is the only pair that was made for each other and i have not added them into my list, so add them! They are perfect for each other 
				// they are either leaf vertexInTriangle or edge-edge intersection case! 
				if(std::find(selectedTriangleTrianglePair.begin(), selectedTriangleTrianglePair.end(), fpair) ==  selectedTriangleTrianglePair.end()) {
					selectedTriangleTrianglePair.push_back(fpair); 
				}
			}
			else if(triangle2triangleList[tj].size() == 2 and (triangle2triangleList[tj][0] == ti or triangle2triangleList[tj][1] == ti) ) { 
				// most of the point In Triangle case looks like this
				// they are mostly internal vertexInTriangle
				std::pair<int,int> fpair = (ti < tj) ? std::make_pair(ti, tj) : std::make_pair(tj, ti);
				if(std::find(selectedTriangleTrianglePair.begin(), selectedTriangleTrianglePair.end(), fpair) ==  selectedTriangleTrianglePair.end()) {
					selectedTriangleTrianglePair.push_back(fpair); 
				}
			}
		}
        // else if(tvector.size() == 2) {
        //     for(int tj: tvector) {
        //         if (triangle2triangleList[tj].size() >= 2){
        //             std::cout << "[SELECTED] Complex overlap found!!\n";
        //             return -1;
        //         }
        //     }
        // }
        // else if(tvector.size() >= 2) {
        //     std::cout << "[SELECTED] Complex > 3 overlap found!!\n";
        //     return -1;
        // }

        for(int tj: tvector) {
            if (checkLocalOverlap(F, ti, tj)) {
                std::pair<int,int> fpair = (ti < tj) ? std::make_pair(ti, tj) : std::make_pair(tj, ti);
                if(std::find(selectedTriangleTrianglePair.begin(), selectedTriangleTrianglePair.end(), fpair) ==  selectedTriangleTrianglePair.end()) {
                    // selectedTriangleTrianglePair.push_back(fpair); 
                    selectedTriangleTrianglePair.insert(selectedTriangleTrianglePair.begin(), fpair);

                }
            }
        }
	}

    return 0;


    // return selectedTriangleTrianglePair;
}


/**
 * @brief Resolves overlapping triangles in a given mesh.
 *
 * This function processes a mesh defined by its vertices (V) and face indices (F) to resolve overlaps between triangles.
 * It identifies selected overlapping triangle pairs from the provided list and attempts to resolve these overlaps
 * by adjusting the positions of the corresponding vertices in a 2D plane (using only x and y coordinates).

 * @param V Reference to an Eigen::MatrixXd containing the vertex positions. Only the x and y coordinates are used.
 * @param F Constant reference to an Eigen::MatrixXi that holds the indices for vertices composing each face.
 * @param overlappingTriangles Reference to a vector of pairs, where each pair contains the indices of two overlapping faces.
 *
 * @return int Returns 0 if all overlaps are successfully resolved.
 *             Returns 1 if a complex overlap (mismatch in expected overlapping pairs) is detected.
 *             Returns -1 if resolution for any triangle pair fails.
 */
int resolveOverlappingTriangles(Eigen::MatrixXd & V,
                                const Eigen::MatrixXi & F,
                                std::vector<std::pair<int,int>> & overlappingTriangles) {
    
    initializeEdgeCounts(F);
    std::vector<std::pair<int,int>> selectedTriangleTrianglePair;
    int success_selected  = getSelecetedOverlappingTriangles(V, F, overlappingTriangles, selectedTriangleTrianglePair);

    if(success_selected == -1) {
        // std::cout << "Complex overlap found!!\n";
        return 1;
    }

    // if(selectedTriangleTrianglePair.size() != overlappingTriangles.size()) {
    //     std::cout << "Complex overlap found!!\n";
    //     return 1;
    // }
    
    for (const auto &pair : selectedTriangleTrianglePair) {
        int face1 = pair.first;
        int face2 = pair.second;
        
        // Build triangles T1 and T2 using x and y coordinates, with z = 0.
        std::vector<Eigen::Vector3d> T1, T2;
        for (int i = 0; i < 3; i++) {
            int idx1 = F(face1, i);
            int idx2 = F(face2, i);
            
            // Create Eigen::Vector3d from the vertex row:
            // Similar to: Eigen::Vector3d vv0(mesh.V.row(v0)(0), mesh.V.row(v0)(1), 0.0)
            Eigen::Vector3d vv(V(idx1, 0), V(idx1, 1), 0.0);
            Eigen::Vector3d ww(V(idx2, 0), V(idx2, 1), 0.0);
            
            T1.push_back(vv);
            T2.push_back(ww);
            
            // std::cout << "Face1: " << face1 << " Face2: " << face2 << std::endl;
            // std::cout << "F(face1, i): " << idx1 << " F(face2, i): " << idx2 << std::endl;
        }
        
        // Resolve overlap between the two triangles
        if( checkLocalOverlap(F, face1, face2) ) {
            if (resolveOverlapForTriangle_Local(T1, T2, F.row(face1), F.row(face2))) {
                std::cout << "Resolved local overlap between faces " << face1 << " and " << face2 << std::endl;
                goto update_vertices;
            }
        }
        if (!resolveOverlapForTriangle_Edges(T1, T2)) {
            std::cout << "Could not resolve overlap between faces " << face1 << " and " << face2 << std::endl;
            return -1;
        }
        
update_vertices:
        // Update V with the new vertex positions for face face1 and face face2
        for (int i = 0; i < 3; i++) {
            int idx1 = F(face1, i);
            int idx2 = F(face2, i);
            
            // std::cout << "updating vertex " << idx1 << " to " << T1[i].transpose() << std::endl;
            // std::cout << "updating vertex " << idx2 << " to " << T2[i].transpose() << std::endl;
            
            // Write back the new positions into V:
            V(idx1, 0) = T1[i](0);
            V(idx1, 1) = T1[i](1);
            // V(idx1, 2) = T1[i](2);
            
            V(idx2, 0) = T2[i](0);
            V(idx2, 1) = T2[i](1);
            // V(idx2, 2) = T2[i](2);
        }
    }
    return 0;
}


// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
int orientation(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d r) {
    double val = (q[1] - p[1]) * (r[0] - q[0]) -
              (q[0] - p[0]) * (r[1] - q[1]);
 
    if (val == 0) return 0;  // collinear
    return (val > 0)? 1: 2; // clock or counterclock wise

}
bool edgeEdgeIntersection(Eigen::Vector3d p1, Eigen::Vector3d q1, Eigen::Vector3d p2, Eigen::Vector3d q2, bool allow_common_endpoints) {
    // p1, q1 from same edge
    // p2, q2 from same edge

    double eps = 1e-12;
    // edges share a common end point no worries allowed 
    if (allow_common_endpoints && ((p1-p2).norm() < eps or (p1-q2).norm() < eps  or (q1-p2).norm() < eps or (q1-q2).norm() < eps)){
            // printf("The edge shares a common end point ... so no intersection...\n");
        return false;
    } 

    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4) {
        // printvec(p1, "p1");
        // printvec(p2, "p2");
        // printvec(q1, "q1");
        // printvec(q2, "q2");
        // printf("%d %d %d %d\n", o1, o2, o3, o4);
        return true;
    }
 
    // Special Cases
    // p1, q1 and p2 are collinear and p2 lies on segment p1q1
    // printf("Either collinear or does not intersect\n");
    if (o1 == 0 && onSegment(p1, p2, q1)) return false; // collinear points are not allowed
 
    // p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return false;
 
    // p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return false;
 
     // p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return false;
 
    return false; // Doesn't fall in any of the above cases
}

bool onSegment(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d r) // if they are collinear points check if q lies on p and r
{
    if (q[0] <= max(p[0], r[0]) && q[0] >= min(p[0], r[0]) &&
        q[1] <= max(p[1], r[1]) && q[1] >= min(p[1], r[1]))
       return true;
 
    return false;
}

bool pointOnTriangle(Eigen::Vector3d pt, Eigen::Vector3d v1, Eigen::Vector3d v2, Eigen::Vector3d v3) {
    double alpha, beta, gamma;
    barycentricCoordinates(v1, v2, v3, pt, alpha, beta, gamma);
    return (alpha == 0 or beta == 0 or gamma == 0);
}


// Edge check to make sure that there edges also do not collide
int countEdgeIntersections(const std::vector<Eigen::Vector3d> &T1, const std::vector<Eigen::Vector3d> &T2)
{
    int countEchecks = 0;
    for (int i = 0; i < 3; ++i) {
        int i_next = (i + 1) % 3;
        for (int j = 0; j < 3; ++j) {
            int j_next = (j + 1) % 3;
            if (edgeEdgeIntersection(T1[i], T1[i_next], T2[j], T2[j_next])) {
                ++countEchecks;
            }
        }
    }
    return countEchecks;
}

bool pointInTriangle(Eigen::Vector3d pt, Eigen::Vector3d v1, Eigen::Vector3d v2, Eigen::Vector3d v3) {
    double eps = 1e-8;
    double alpha, beta, gamma;
    barycentricCoordinates(v1, v2, v3, pt, alpha, beta, gamma);
    return alpha > eps && beta > eps && gamma > eps;
}

bool checkTriangleTriangleIntersection(vector<Eigen::Vector3d> T1, vector<Eigen::Vector3d> T2, std::pair<int, int>* counts) {

    // point on triangle check
    bool checks[6];
    for (int i = 0; i < 3; ++i) {
        checks[i]   = pointInTriangle(T1[i], T2[0], T2[1], T2[2]);
        checks[i+3] = pointInTriangle(T2[i], T1[0], T1[1], T1[2]);
    }

    int vertex_checks = std::accumulate(std::begin(checks), std::end(checks), 0);
    bool isInside = vertex_checks > 0;

    // TODO take the bottom case into account as well!!!
    int countEchecks = countEdgeIntersections(T1, T2);

    if (counts) {
        *counts = std::make_pair(vertex_checks, countEchecks);
    }
     // ZW: There is no way two triangles can intersect with only one edge intersection
    isInside = isInside || (countEchecks >= 2);
    if (isInside){
            // printf("vertex check values : %d %d %d %d %d %d\n", check0, check1, check2, check3, check4, check5);
            // printf("Edge check values : %d %d %d %d %d %d %d %d %d\n", echeck0, echeck1, echeck2, echeck3, echeck4, echeck5, echeck6, echeck7, echeck8);

            // if(vertex_checks == 1 && countEchecks == 1){
            //     std::cout << "Triangle T0: ";
            //     for(int i = 0; i < 3; i++) {
            //         std::cout << "(" << T1[i][0] << ", " << T1[i][1] << ", " << T1[i][2] << ") ";
            //     }
            //     std::cout << "\nTriangle T1: ";
            //     for(int i = 0; i < 3; i++) {
            //         std::cout << "(" << T2[i][0] << ", " << T2[i][1] << ", " << T2[i][2] << ") ";
            //     }
            //     std::cout << std::endl;
            // }

            return true;
        }
    else    
        return false;
    return false;
}


double computeOverlapingTrianglesFast(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    std::vector<std::pair<int,int>> &overlappingTriangles_return)
{

    EASY_FUNCTION(profiler::colors::Green);
    auto start = std::chrono::high_resolution_clock::now();

    ViewerDataSimple mesh(V, F);
    // auto start = std::chrono::system_clock::now();
    overlappingTriangles_return.clear();

    std::vector<Point_2> points_2s(mesh.V.rows()); 
    std::map<Point_2, int> vertexMap;
    for(int vi = 0; vi < mesh.V.rows(); vi++) {
        points_2s[vi] = Point_2(mesh.V(vi,0), mesh.V(vi,1));
        vertexMap[points_2s[vi]] = vi; 
    }
    auto comparator = [](const Triangle_2& c1, const Triangle_2& c2) -> bool {
        return c1.vertex(0) < c2.vertex(0);
    };

    // std::map<Color, int, decltype(comparator)> myMap(comparator);
    std::list<Triangle_2> triangles;
    std::map<Triangle_2, int, decltype(comparator)> triangleMap((comparator)); 
    for(int fi = 0; fi < mesh.F.rows(); fi++) {    // from 0 -> 1
        int v0 = mesh.F(fi, 0), v1 = mesh.F(fi, 1), v2 = mesh.F(fi, 2);
        Triangle_2 tfi = Triangle_2(points_2s[v0], points_2s[v1], points_2s[v2]);
        triangles.push_back(tfi);
        triangleMap[tfi] = fi;
    }

    std::vector<Box> boxes;
    boxes.reserve(triangles.size());
    std::map<int, int> bbox2ind;
    int fi = 0;
    for (auto tit = triangles.begin(); tit != triangles.end(); ++tit) {
        if (!tit->is_degenerate()) {
            Box bbox = Box(tit->bbox());
            bbox2ind[bbox.id()] = fi;
            boxes.push_back(bbox);
        }
        fi++; 
    }

    auto fun = [&](const Box &a,const Box &b) {
        // std::cout << "box " << a.id() << " intersects box " << b.id() << std::endl;
        if(a.id() == b.id())
            return;

        int f0 = bbox2ind[a.id()], f1 = bbox2ind[b.id()]; 
        int v0 = mesh.F.row(f0)(0), v1 = mesh.F.row(f0)(1), v2 = mesh.F.row(f0)(2);

        auto vv00 = mesh.V.row(v0);

        Eigen::Vector3d vv0(mesh.V.row(v0)(0), mesh.V.row(v0)(1), 0.0),
                        vv1(mesh.V.row(v1)(0), mesh.V.row(v1)(1), 0.0),
                        vv2(mesh.V.row(v2)(0), mesh.V.row(v2)(1), 0.0);
        int w0 = mesh.F.row(f1)(0), w1 = mesh.F.row(f1)(1), w2 = mesh.F.row(f1)(2);
        Eigen::Vector3d ww0(mesh.V.row(w0)(0), mesh.V.row(w0)(1), 0.0),
                        ww1(mesh.V.row(w1)(0), mesh.V.row(w1)(1), 0.0),
                        ww2(mesh.V.row(w2)(0), mesh.V.row(w2)(1), 0.0);

        std::vector<Eigen::Vector3d> T0, T1;
        T0.push_back(vv0); T0.push_back(vv1); T0.push_back(vv2); 
        T1.push_back(ww0); T1.push_back(ww1); T1.push_back(ww2); 


        bool isOverlapping = checkTriangleTriangleIntersection(T0, T1);
        if(isOverlapping) {
            auto fpair = (f0 < f1) ? make_pair(f0, f1) : make_pair(f1, f0);
            overlappingTriangles_return.push_back(fpair);
            // isOverlapping = checkTriangleTriangleIntersection(T0, T1);
        }
    };
   
    CGAL::box_self_intersection_d(boxes.begin(), boxes.end(), fun);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>  elapsed_us = end - start;
    total_time_spent_overlap += elapsed_us.count();

    return overlappingTriangles_return.size();
}

void barycentricCoordinates(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& C, const Eigen::Vector3d& P, double& alpha, double& beta, double& gamma) {
    double denominator = ((B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]));
    alpha = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denominator;
    beta = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denominator;
    gamma = 1.0 - alpha - beta;
    // without using fabs.. 
}




std::vector<int> intersectVectors(vector<int> A, vector<int> B) {
    sort(A.begin(), A.end());
    sort(B.begin(), B.end());
    vector<int> C;
    set_intersection(A.begin(),A.end(),
                          B.begin(),B.end(),
                          back_inserter(C));
    return C;

}

map<int, int> flat2worldVind;
map<int, int> world2flatVind;
typedef std::pair<int, int> pi;

int findCulpritVertex(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, int f0, int f1) {
    
    int vi0 = F(f0,0), vi1 = F(f0,1), vi2 = F(f0,2); std::vector<int> vis = {vi0, vi1, vi2};
    int vj0 = F(f1,0), vj1 = F(f1,1), vj2 = F(f1,2); std::vector<int> vjs = {vj0, vj1, vj2};
    std::vector<Eigen::Vector3d> T0 = { Eigen::Vector3d(V.row(vis[0])), Eigen::Vector3d(V.row(vis[1])), Eigen::Vector3d(V.row(vis[2]))};
    std::vector<Eigen::Vector3d> T1 = { Eigen::Vector3d(V.row(vjs[0])), Eigen::Vector3d(V.row(vjs[1])), Eigen::Vector3d(V.row(vjs[2]))};

    for(int i = 0; i < 3; i++) {
        if(pointInTriangle(T0[i], T1[0], T1[1], T1[2]) or pointOnTriangle(T0[i], T1[0], T1[1], T1[2])) {
            return vis[i];
        }

        if(pointInTriangle(T1[i], T0[0], T0[1], T0[2]) or pointOnTriangle(T1[i], T0[0], T0[1], T0[2])) {
            return vjs[i];
        }
    }
    
    // they have edge in vertex intersection!! 
    return -1; 
    // printf("Something went wrong in finding the culprit vertex!! DEBUG HERE!!!!\n");
    // exit(1); 

}
