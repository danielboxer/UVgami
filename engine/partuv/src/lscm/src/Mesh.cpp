#include "Vertex.h"
#include "HalfEdge.h"
#include "Edge.h"
#include "Face.h"
#include "Mesh.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <Eigen/Core>
#include <map>
#include "FormTrait.h"
#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

using namespace MeshLib;

// #define VERBOSE

//access e->v
Vertex *Mesh::edge_vertex_1(Edge  *e) {
	assert(e->halfedge(0) != NULL);
	return e->halfedge(0)->source();
}

//access e->v
Vertex *Mesh::edge_vertex_2(Edge  *e) {
	assert(e->halfedge(0) != NULL);
	return e->halfedge(0)->target();
}

//access e->f
Face *Mesh::edge_face_1(Edge  *e) {
	assert(e->halfedge(0) != NULL);
	return e->halfedge(0)->face();
}

//access e->f
Face *Mesh::edge_face_2(Edge  *e) {
	assert(e->halfedge(1) != NULL);
	return e->halfedge(1)->face();
}

//access he->f
Face *Mesh::halfedge_face(HalfEdge  *he) {
	return he->face();
}


//access he->v
Vertex  *Mesh::halfedge_vertex(HalfEdge  *he) {
	return he->vertex();
}

bool  Mesh::is_boundary(Vertex * v) {
	return v->boundary();
}

bool  Mesh::is_boundary(Edge  *e) {
	if (e->halfedge(0) == NULL || e->halfedge(1) == NULL) return true;
	return false;
}

bool  Mesh::is_boundary(HalfEdge  *he) {
	if (he->he_sym() == NULL) return true;
	return false;
}

int Mesh::numVertices() {
	return (int)m_vertices.size();
}

int Mesh::numEdges() {
	return (int)m_edges.size();
}

int Mesh::numFaces() {
	return (int)m_faces.size();
}

//Euler operation

HalfEdge *Mesh::vertexMostClwOutHalfEdge(Vertex  *v) {
	return v->most_clw_out_halfedge();
}

HalfEdge *Mesh::vertexMostCcwOutHalfEdge(Vertex  *v) {
	return v->most_ccw_out_halfedge();
}

HalfEdge *Mesh::corner(Vertex *v, Face *f) {
	HalfEdge *he = f->halfedge();
	do{
		if (he->vertex() == v)
			return he;
		he = he->he_next();
	} while (he != f->halfedge());
	return NULL;
}

HalfEdge *Mesh::vertexNextCcwOutHalfEdge(HalfEdge  *he) {
	return he->ccw_rotate_about_source();
}

HalfEdge *Mesh::vertexNextClwOutHalfEdge(HalfEdge  *he) {
	assert(he->he_sym() != NULL);
	return he->clw_rotate_about_source();
}

HalfEdge *Mesh::vertexMostClwInHalfEdge(Vertex  *v) {
	return v->most_clw_in_halfedge();
}

HalfEdge *Mesh::vertexMostCcwInHalfEdge(Vertex  *v) {
	return v->most_ccw_in_halfedge();
}

HalfEdge *Mesh::vertexNextCcwInHalfEdge(HalfEdge  *he) {
	assert(he->he_sym() != NULL);
	return he->ccw_rotate_about_target();
}

HalfEdge *vertexNextClwInHalfEdge(HalfEdge  *he) {
	return he->clw_rotate_about_target();
}

HalfEdge *Mesh::faceNextClwHalfEdge(HalfEdge  *he) {
	return he->he_prev();
}

HalfEdge *Mesh::faceNextCcwHalfEdge(HalfEdge  *he) {
	return he->he_next();
}

HalfEdge *Mesh::faceMostCcwHalfEdge(Face  *face) {
	return face->halfedge();
}

HalfEdge *Mesh::faceMostClwHalfEdge(Face  *face) {
	return face->halfedge()->he_next();
}

Mesh::~Mesh() {
	//remove vertices
	for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); viter++) {
		Vertex * pV = *viter;
		delete pV;
	}
	m_vertices.clear();

	//remove faces
	for (std::list<Face*>::iterator fiter = m_faces.begin(); fiter != m_faces.end(); fiter++) {
		Face * pF = *fiter;

		HalfEdge *he = pF->halfedge();

		std::list<HalfEdge*> hes;
		do{
			he = he->he_next();
			hes.push_back(he);
		} while (he != pF->halfedge());

		for (std::list<HalfEdge*>::iterator hiter = hes.begin(); hiter != hes.end(); hiter++) {
			HalfEdge * pH = *hiter;
			delete pH;
		}
		hes.clear();

		delete pF;
	}
	m_faces.clear();

	//remove edges
	for (std::list<Edge*>::iterator eiter = m_edges.begin(); eiter != m_edges.end(); eiter++) {
		Edge * pE = *eiter;
		delete pE;
	}

	m_edges.clear();

	m_map_vertex.clear();
	m_map_face.clear();
	m_map_edge.clear();
}

double Mesh::edge_length(Edge *e) {
	Vertex * v1 = edge_vertex_1(e);
	Vertex * v2 = edge_vertex_2(e);
	return (v1->point() - v2->point()).norm();
}

//create new gemetric simplexes
Vertex *Mesh::create_vertex(int id) {
	Vertex *v = new Vertex();
	assert(v != NULL);
	v->id() = id;
	m_vertices.push_back(v);
	m_map_vertex.insert(std::pair<int, Vertex*>(id, v));
	return v;//insert a new vertex, with id as the key
}





void pinMultipleVertices(
    std::vector<Vertex*>& boundaryVerts,  // all boundary vertices
    Vertex* pinnedV1,                     // first pinned
    Vertex* pinnedV2,                     // second pinned
    int num_points    = 2                     // total number of points to pin
)
{
    // 1) Assign the baseline pins with known (u,v) in 2D
    pinnedV1->string() = "fixt 0.0 0.5\r";
    pinnedV2->string() = "fixt 1.0 0.5\r";

    // Store the pinned vertices in a container so we can skip them later
    std::vector<Vertex*> pinnedList;
    pinnedList.push_back(pinnedV1);
    pinnedList.push_back(pinnedV2);

    // Precompute baseline info for the line from pinnedV1->point to pinnedV2->point
    Point p1 = pinnedV1->point();
    Point p2 = pinnedV2->point();
    double bx = p2[0] - p1[0];
    double by = p2[1] - p1[1];
    double bz = p2[2] - p1[2];
    double baseLength = std::sqrt(bx*bx + by*by + bz*bz);

    // Unit direction of the baseline
    double baseDir[3] = {
        (baseLength > 0.0 ? bx / baseLength : 0.0),
        (baseLength > 0.0 ? by / baseLength : 0.0),
        (baseLength > 0.0 ? bz / baseLength : 0.0)
    };

    // For scaling the perpendicular direction "v", we need
    // the maximum distance from the line p1->p2 over ALL boundary vertices.
    // This ensures that the maximum offset is 0.5 in the final mapping.
    double maxPerpDistSqAll = 0.0;
    for (Vertex* v : boundaryVerts)
    {
        Point vp = v->point();
        double dx = vp[0] - p1[0];
        double dy = vp[1] - p1[1];
        double dz = vp[2] - p1[2];

        // Projection length t on the baseline
        double t = dx*baseDir[0] + dy*baseDir[1] + dz*baseDir[2];

        // Projected point (on line) = p1 + t * baseDir
        double proj[3] = {
            p1[0] + t*baseDir[0],
            p1[1] + t*baseDir[1],
            p1[2] + t*baseDir[2]
        };
        // Perp vector
        double px = vp[0] - proj[0];
        double py = vp[1] - proj[1];
        double pz = vp[2] - proj[2];
        double distSq = px*px + py*py + pz*pz;

        if (distSq > maxPerpDistSqAll) {
            maxPerpDistSqAll = distSq;
        }
    }
    double maxPerpDistAll = std::sqrt(maxPerpDistSqAll);
    double scaleV = (maxPerpDistAll > 0.0) ? (0.5 / maxPerpDistAll) : 0.0;

    // 2) We already pinned 2, so we want to pick (num_points - 2) more
    //    in a loop. Each iteration picks the next "farthest from pinned set" vertex.
    int needed = num_points - 2;
    while ((int)pinnedList.size() < num_points && needed > 0)
    {
        // ------------------------------------------------
        // (a) Find the boundary vertex "furthest" from the
        //     existing pinned set in 3D.  We'll use the
        //     largest *min-dist-squared* criterion.
        // ------------------------------------------------
        Vertex* furthest = nullptr;
        double bestMinDistSq = 0.0;

        for (Vertex* v : boundaryVerts)
        {
            // Skip if already pinned
            if (std::find(pinnedList.begin(), pinnedList.end(), v) != pinnedList.end())
                continue;

            // Compute minDistSq to the pinned set
            Point vp = v->point();
            
			// Compute the minimum squared distance to the pinned set.
			double minDistSq = std::numeric_limits<double>::max();
			for (Vertex* pinned : pinnedList)
			{
				Point pp = pinned->point();
				double dx = vp[0] - pp[0];
				double dy = vp[1] - pp[1];
				double dz = vp[2] - pp[2];
				double distSq = dx*dx + dy*dy + dz*dz;
				// Take the minimum distance
				if (distSq < minDistSq)
					minDistSq = distSq;
			}

			// Update if this vertex has a larger minimum distance to the pinned set
			if (minDistSq > bestMinDistSq)
			{
				bestMinDistSq = minDistSq;
				furthest = v;
			}
        }

        // If no more found, break
        if (!furthest) break;

        // ------------------------------------------------
        // (b) Triangulate its 2D coordinate relative to
        //     the *same baseline* (p1->p2).
        // ------------------------------------------------
        Point p3 = furthest->point();
        // Vector from p1 to p3
        double dx3 = p3[0] - p1[0];
        double dy3 = p3[1] - p1[1];
        double dz3 = p3[2] - p1[2];

        // "u" is the dot-product along baseDir, normalized by baseLength
        double a = dx3*baseDir[0] + dy3*baseDir[1] + dz3*baseDir[2];
        double uCoord = (baseLength != 0.0) ? (a / baseLength) : 0.0; // 0 ~ 1 across the baseline

        // Perp distance "b" to the line: first find the projected point and then the perpendicular vector
        double t = a;  // same dot product as above
        double proj[3] = {
            p1[0] + t*baseDir[0],
            p1[1] + t*baseDir[1],
            p1[2] + t*baseDir[2]
        };
        double px = p3[0] - proj[0];
        double py = p3[1] - proj[1];
        double pz = p3[2] - proj[2];
        // sign for perpendicular not used in this simplistic scheme,
        // so we just treat "b" as the magnitude
        double perpMag = std::sqrt(px*px + py*py + pz*pz);
        // vCoord is around 0.5 ± scaled perpendicular
        double vCoord = 0.5 + scaleV * perpMag;

        // Assign the final UV coordinate
        {
            std::ostringstream oss;
            oss << "fixt " << uCoord << " " << vCoord << "\r";
            furthest->string() = oss.str();
			std::cout << "Pinned point on boundary (ID): " << furthest->id() -1 << " to " 
				<< furthest->string() << std::endl;
        }

        // ------------------------------------------------
        // (c) Mark it pinned
        // ------------------------------------------------
        pinnedList.push_back(furthest);
        --needed;
    }
}



void Mesh::compute_pinned_vertices_2()
{
    // ------------------------------------------------------------
    // 1) Gather (and ensure ORDERED) boundary vertices
    // ------------------------------------------------------------
    std::vector<Vertex*> boundaryVerts;
    boundaryVerts.reserve(m_vertices.size());

    for (auto v : m_vertices)
    {
        if (v->boundary())
        {
            boundaryVerts.push_back(v);
        }
    }

    // For the "symmetry" approach to work, boundaryVerts must form
    // a continuous loop in index order. If your data doesn't already
    // guarantee that, you'd need to *build* this loop from adjacency.
    // e.g. boundaryVerts = build_ordered_boundary_loop(...);

    // If fewer than 2 boundary vertices, fallback to min/max X
    if (boundaryVerts.size() < 2)
    {
        std::cerr << "[Warning] Not enough boundary vertices found. Using min/max X fallback.\n";
        
        //  -- (same as your original fallback) --
        double minX =  std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        Vertex* vmin = nullptr;
        Vertex* vmax = nullptr;
        
        for (auto v : m_vertices)
        {
            double x = v->point()[0];
            if (x < minX) { minX = x; vmin = v; }
            if (x > maxX) { maxX = x; vmax = v; }
        }

        if (vmin && vmax && (vmin != vmax))
        {
            double dx = vmax->point()[0] - vmin->point()[0];
            double dy = vmax->point()[1] - vmin->point()[1];
            double dz = vmax->point()[2] - vmin->point()[2];
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            // Pin them
            vmin->string() = "fixt 0.0 0.5 \r";
            vmax->string() = "fixt 1.0 0.5 \r";

            if (vmin->id() >= 0 && vmax->id() >= 0)
            {
                std::cout << "Pinned fallback points (IDs): "
                          << vmin->id() << " and " << vmax->id()
                          << ", distance = " << dist << std::endl;
            }
        }
        else
        {
            std::cerr << "[Error] Could not find distinct min/max X vertices!\n";
        }
        return;
    }

    // ------------------------------------------------------------
    // 2) Measure the total perimeter of this boundary loop
    // ------------------------------------------------------------
    const size_t N = boundaryVerts.size();
    double totalLen = 0.0;
    
    // We'll store the edge lengths in an array for easy access
    std::vector<double> edgeLens(N, 0.0);

    for (size_t i = 0; i < N; ++i)
    {
        size_t iNext = (i + 1) % N;
        // Distance from boundaryVerts[i] to boundaryVerts[iNext]
        double dx = boundaryVerts[iNext]->point()[0] - boundaryVerts[i]->point()[0];
        double dy = boundaryVerts[iNext]->point()[1] - boundaryVerts[i]->point()[1];
        double dz = boundaryVerts[iNext]->point()[2] - boundaryVerts[i]->point()[2];
        double length = std::sqrt(dx * dx + dy * dy + dz * dz);

        edgeLens[i] = length;
        totalLen += length;
    }

    // If the boundary forms multiple loops or has breaks,
    // you need a more robust approach to measure each loop separately.
    // But let's assume a single continuous loop for now.

    // ------------------------------------------------------------
    // 3) Find pin1 = midpoint along the boundary (meet in the middle)
    //    This mirrors the approach from p_chart_symmetry_pins:
    //    Start from both ends of the 'longest chain' and accumulate
    //    until i1 == i2.
    // ------------------------------------------------------------
    size_t i1 = 0;         // forward pointer
    size_t i2 = (N - 1);   // backward pointer
    double len1 = 0.0;
    double len2 = 0.0;

    // Move i1 forward and i2 backward until they meet
    while (i1 != i2)
    {
        if (len1 < len2)
        {
            // move i1 forward
            len1 += edgeLens[i1];
            i1 = (i1 + 1) % N;
        }
        else
        {
            // move i2 backward
            i2 = (i2 + N - 1) % N; // safe backward step
            len2 += edgeLens[i2];
        }
    }

    Vertex* pinnedV1 = boundaryVerts[i1];  // pin1

    // ------------------------------------------------------------
    // 4) Find pin2 by "meet in the middle" outside the chain
    //    In p_chart_symmetry_pins, it re-initializes i1/i2 to the
    //    extremes and walks them in the opposite direction. 
    // ------------------------------------------------------------
    i1 = 0;
    i2 = (N - 1);
    len1 = 0.0;
    len2 = 0.0;

    while (i1 != i2)
    {
        if (len1 < len2)
        {
            // Now we move i1 *backward* 
            // (because p_chart_symmetry_pins does be1 = p_boundary_edge_prev(be1))
            i1 = (i1 + N - 1) % N;
            len1 += edgeLens[i1];
        }
        else
        {
            // move i2 forward
            // (since p_chart_symmetry_pins does be2 = p_boundary_edge_next(be2))
            len2 += edgeLens[i2];
            i2 = (i2 + 1) % N;
        }
    }

    Vertex* pinnedV2 = boundaryVerts[i1];  // pin2

    // ------------------------------------------------------------
    // 5) "Pin" them in your system
    // ------------------------------------------------------------
    pinnedV1->string() = "fixt 0.0 0.5 \r";
    pinnedV2->string() = "fixt 1.0 0.5 \r";

    // Optionally log them
    if (pinnedV1->id() >= 0 && pinnedV2->id() >= 0)
    {
        // Just to measure distance for logging:
        double dx = pinnedV1->point()[0] - pinnedV2->point()[0];
        double dy = pinnedV1->point()[1] - pinnedV2->point()[1];
        double dz = pinnedV1->point()[2] - pinnedV2->point()[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        #ifdef VERBOSE
            std::cout << "Pinned points on boundary (IDs): "
                  << pinnedV1->id() << " and " << pinnedV2->id()
                  << ", distance = " << dist << std::endl;
        #endif
    }
}




void Mesh::compute_pinned_vertices_3()
{
    // ------------------------------------------------------------
    // 1) Gather (and ensure ORDERED) boundary vertices
    // ------------------------------------------------------------
    std::vector<Vertex*> boundaryVerts;
    boundaryVerts.reserve(m_vertices.size());

    Vertex* pinnedV1 = nullptr;
    Vertex* pinnedV2 = nullptr;

    for (std::list<Edge*>::iterator eiter = m_edges.begin(); eiter != m_edges.end(); ++eiter) {
        Edge *edge = *eiter;
        HalfEdge *he[2];
        he[0] = edge->halfedge(0);
        he[1] = edge->halfedge(1);

        if (he[0] == NULL) continue;
        if (he[1] == NULL) {
            // boundary
            pinnedV1 = he[0]->vertex();
            pinnedV2 = he[0]->he_prev()->vertex();
            break;
        }
    }


    // For the "symmetry" approach to work, boundaryVerts must form
 

    // ------------------------------------------------------------
    // 5) "Pin" them in your system
    // ------------------------------------------------------------
    pinnedV1->string() = "fixt 0.0 0.5 \r";
    pinnedV2->string() = "fixt 1.0 0.5 \r";

    // Optionally log them
    if (pinnedV1->id() >= 0 && pinnedV2->id() >= 0)
    {
        // Just to measure distance for logging:
        double dx = pinnedV1->point()[0] - pinnedV2->point()[0];
        double dy = pinnedV1->point()[1] - pinnedV2->point()[1];
        double dz = pinnedV1->point()[2] - pinnedV2->point()[2];
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        #ifdef VERBOSE
            std::cout << "Pinned points on boundary (IDs): "
                  << pinnedV1->id() << " and " << pinnedV2->id()
                  << ", distance = " << dist << std::endl;
        #endif
    }
}





void Mesh::compute_pinned_vertices()
{
    // 1) Gather boundary vertices into a std::vector
    std::vector<Vertex*> boundaryVerts;
    for (auto v : m_vertices)
    {
        if (v->boundary())
        {
            boundaryVerts.push_back(v);
        }
    }

    // We'll store pinned vertex IDs (1-based, as per your mesh indexing)
    Vertex* pinnedV1; Vertex* pinnedV2;
    double maxDist = -1.0;

    // 2) If fewer than 2 boundary vertices, fallback to min-x and max-x among all
    if (boundaryVerts.size() < 2)
    {
        std::cerr << "[Warning] Not enough boundary vertices found. Using min/max X fallback.\n";
        
        // find min-x and max-x among all vertices
        double minX = std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        Vertex* vmin = nullptr;
        Vertex* vmax = nullptr;
        
        for (auto v : m_vertices)
        {
            double x = v->point()[0];
            if (x < minX)
            {
                minX = x;
                vmin = v;
            }
            if (x > maxX)
            {
                maxX = x;
                vmax = v;
            }
        }

        if (vmin && vmax && (vmin != vmax))
        {
            pinnedV1 = vmin;
            pinnedV2 = vmax;
            double dx = vmax->point()[0] - vmin->point()[0];
            double dy = vmax->point()[1] - vmin->point()[1];
            double dz = vmax->point()[2] - vmin->point()[2];
            maxDist = std::sqrt(dx * dx + dy * dy + dz * dz);

			pinnedV1->string() = "fixt 0.0 0.5 \r";
            pinnedV2->string() = "fixt 1.0 0.5 \r";

            #ifdef VERBOSE
            if (pinnedV1->id() >= 0 && pinnedV2->id() >= 0)
            {
                double dist = std::sqrt(maxDist);
                std::cout << "Pinned points on boundary (IDs): "
                          << pinnedV1->id() << " and " << pinnedV2->id()
                          << ", distance = " << dist << std::endl;
            }
            #endif
            std::cerr << "[Error] Could not find distinct min/max X vertices!\n";
        }
    }
    else
    {
        // 3) Otherwise, find the two boundary vertices that are farthest apart (brute force)
        for (size_t i = 0; i < boundaryVerts.size(); ++i)
        {
            Point pi = boundaryVerts[i]->point();
            for (size_t j = i + 1; j < boundaryVerts.size(); ++j)
            {
                Point pj = boundaryVerts[j]->point();

                double dx = pi[0] - pj[0];
                double dy = pi[1] - pj[1];
                double dz = pi[2] - pj[2];
                double distSq = dx * dx + dy * dy + dz * dz; // compare squared distances

                if (distSq > maxDist)
                {
                    maxDist = distSq;
                    pinnedV1 = boundaryVerts[i];
                    pinnedV2 = boundaryVerts[j];
                }
            }
        }

		pinnedV1->string() = "fixt 0.0 0.5 \r";
		pinnedV2->string() = "fixt 1.0 0.5 \r";

	}
	#ifdef VERBOSE
	if (pinnedV1->id() >= 0 && pinnedV2->id() >= 0)
	{
		double dist = std::sqrt(maxDist);
		std::cout << "Pinned points on boundary (IDs): "
					<< pinnedV1->id() << " and " << pinnedV2->id() 
					<< ", distance = " << dist << std::endl;
	}
	#endif
    // return std::make_pair(pinnedV1, pinnedV2);
}




// V: #V x 3 (vertex positions)
// F: #F x 3 (triangle indices, typically 0-based in IGL)
int Mesh::from_igl(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
    // 1) Clear the current mesh (if necessary).
    //    (You'll need to define clear(), or do it manually if your code doesn't have it.)
    // clear();

    // Some checks (optional)
    if (V.cols() < 3 || F.cols() < 3) {
        std::cerr << "Error: input has wrong dimensions.\n";
        return 0;
    }

    // 2) Create vertices
    //    In your read_obj code, vertices were 1-based. So let's keep that convention.
    //    We'll fill m_map_vertex[i+1] = pointer to new Vertex with ID = i+1.
    int numVerts = (int)V.rows();
    for (int i = 0; i < numVerts; ++i)
    {
        // Create a new vertex with ID = i + 1
        Vertex *v = create_vertex(i + 1);
        // Set its coordinates (Point is presumably a 3D type in your code)
        Point p;
        p[0] = V(i, 0);
        p[1] = V(i, 1);
        p[2] = V(i, 2);
        v->point() = p;
        v->id() = i + 1;
        // If you have any special strings or feature flags, set them here.
        // v->string() = ...
    }

    // 3) Create faces
    //    F is presumably #F x 3. Make sure to handle 0-based -> 1-based indexing.
    int numFaces = (int)F.rows();
    for (int i = 0; i < numFaces; ++i)
    {
        Vertex* verts[3];
        for (int j = 0; j < 3; ++j) {
            // Indices in libIGL are usually 0-based, your code uses 1-based
            int id = F(i, j) + 1;
            verts[j] = m_map_vertex[id];
        }
        create_face(verts, i + 1);
    }

    // 4) If you also have vertex normals from a separate matrix (say, Eigen::MatrixXd NV),
    //    you could assign them here, analogous to how "vn" lines are handled in the .obj file.
    //    For example:
    // for (int i = 0; i < numVerts; ++i)
    // {
    //     Vertex* v = id_vertex(i+1);
    //     Point n(NV(i, 0), NV(i, 1), NV(i, 2));
    //     v->normal() = n;
    // }

    // --- Post-processing: boundary edges, dangling vertices, etc. ---
    // This is directly copied from your read_obj:

    //Label boundary edges
    for (std::list<Edge*>::iterator eiter = m_edges.begin(); eiter != m_edges.end(); ++eiter) {
        Edge *edge = *eiter;
        HalfEdge *he[2];
        he[0] = edge->halfedge(0);
        he[1] = edge->halfedge(1);

        if (he[0] == NULL) continue;
        if (he[1] != NULL) {
            // re-orient if needed
            if (he[0]->target()->id() < he[0]->source()->id()) {
                edge->halfedge(0) = he[1];
                edge->halfedge(1) = he[0];
            }
        }
        else {
            // boundary
            he[0]->vertex()->boundary() = true;
            he[0]->he_prev()->vertex()->boundary() = true;
        }
    }

    // Find dangling vertices
    std::list<Vertex*> dangling_verts;
    for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); ++viter) {
        Vertex *v = *viter;
        if (v->halfedge() == NULL) {
            dangling_verts.push_back(v);
        }
    }

    // Remove dangling vertices
    for (Vertex* dv : dangling_verts) {
        m_vertices.remove(dv);
        delete dv;
        dv = NULL;
    }

    // Re-arrange boundary halfedges
    for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); ++viter) {
        Vertex *v = *viter;
        if (!v->boundary()) continue;

        HalfEdge *he = v->halfedge();
        if (he->he_sym() != NULL) {
            he = he->ccw_rotate_about_target();
        }
        v->halfedge() = he;
    }
	// compute_pinned_vertices_2(numPinned);
    compute_pinned_vertices();
	
    return 0; // success
}



int Mesh::read_obj(const char * filename) {
	//	TRACE("load obj file %s\n",filename);
	FILE* f = fopen(filename, "r");
	if (f == NULL) return 0;

	char cmd[1024];
	char seps[] = " ,\t\n";
	int  vid = 1;
	int  fid = 1;
	int  nid = 1;

	while (true) {
		if (fgets(cmd, 1024, f) == NULL)
			break;

		char *token = strtok(cmd, seps);

		if (token == NULL)
			continue;

		if (strcmp(token, "v") == 0) {
			Point p;
			for (int i = 0; i < 3; i++) {
				token = strtok(NULL, seps);
				p[i] = atof(token);
			}

			Vertex * v = create_vertex(vid);
			v->point() = p;
			v->id() = vid++;

			// Add feature points
            token = strtok(NULL, "\n");
            if (token == NULL) continue;

            std::string s(token);
            if (s.substr(0,3) == "fix") {
                v->string() = s;
            }
			continue;
		}

		if (strcmp(token, "vn") == 0) {
			Point p;
			for (int i = 0; i < 3; i++) {
				token = strtok(NULL, seps);
				p[i] = atof(token);

			}
			Vertex* v = id_vertex(nid);
			v->normal() = p;
			nid++;
			continue;
		}

		if (strcmp(token, "f") == 0) {
			Vertex* v[3];
			for (int i = 0; i < 3; i++)
			{
				token = strtok(NULL, seps);
				// std::cout << (void *) token;
				// char* tmp = strchr(token, '/');
				int id = atoi(token);
				// std::cout << i << ": " << id << ", ";
				v[i] = m_map_vertex[id];
			}
			create_face(v, fid++);
			// std::cout << std::endl;
		}
	}
	fclose(f);

	//Label boundary edges
	for (std::list<Edge*>::iterator eiter = m_edges.begin(); eiter != m_edges.end(); ++eiter) {
		Edge     *edge = *eiter;
		HalfEdge *he[2];

		he[0] = edge->halfedge(0);
		he[1] = edge->halfedge(1);

		assert(he[0] != NULL);

		if (he[1] != NULL) {
			assert(he[0]->target() == he[1]->source() && he[0]->source() == he[1]->target());

			if (he[0]->target()->id() < he[0]->source()->id()) {
				edge->halfedge(0) = he[1];
				edge->halfedge(1) = he[0];
			}

			assert(edge_vertex_1(edge)->id() < edge_vertex_2(edge)->id());
		}
		else {
			he[0]->vertex()->boundary() = true;
			he[0]->he_prev()->vertex()->boundary() = true;
		}
	}

	std::list<Vertex*> dangling_verts;
	//Label boundary vertices
	for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); ++viter) {
		Vertex     *v = *viter;
		if (v->halfedge() != NULL) continue;
		dangling_verts.push_back(v);
	}

	for (std::list<Vertex*>::iterator viter = dangling_verts.begin(); viter != dangling_verts.end(); ++viter) {
		Vertex *v = *viter;
		m_vertices.remove(v);
		delete v;
		v = NULL;
	}

	//Arrange the boundary half_edge of boundary vertices, to make its halfedge
	//to be the most ccw in half_edge

	for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); ++viter) {
		Vertex     *v = *viter;
		if (!v->boundary()) continue;

		HalfEdge * he = v->halfedge();
		while (he->he_sym() != NULL) {
			he = he->ccw_rotate_about_target();
		}

		v->halfedge() = he;
	}
	return 0;
}

Face *Mesh::create_face(Vertex * v[], int id) {
	Face *f = new Face();
	assert(f != NULL);
	f->id() = id;
	m_faces.push_back(f);
	m_map_face.insert(std::pair<int, Face*>(id, f));

	//create halfedges
	HalfEdge *hes[3];

	for (int i = 0; i < 3; i++) {
		hes[i] = new HalfEdge;
		assert(hes[i]);
		Vertex * vert = v[i];
		hes[i]->vertex() = vert;
		vert->halfedge() = hes[i];
	}

	//linking to each other
	for (int i = 0; i < 3; i++) {
		hes[i]->he_next() = hes[(i + 1) % 3];
		hes[i]->he_prev() = hes[(i + 2) % 3];
	}

	//linking to face
	for (int i = 0; i < 3; i++) {
		hes[i]->face() = f;
		f->halfedge() = hes[i];
	}

	//connecting with edge
	for (int i = 0; i < 3; i++) {
		Edge *e = create_edge(v[i], v[(i + 2) % 3]);
		if (e->halfedge(0) == NULL) {
			e->halfedge(0) = hes[i];
		}
		else {
			// assert(e->halfedge(1) == NULL);
			e->halfedge(1) = hes[i];
		}
		hes[i]->edge() = e;
	}

	return f;
}


//access id->v
Vertex *Mesh::id_vertex(int id) {
	return m_map_vertex[id];
}

//access v->id
int Mesh::vertex_id(Vertex  *v) {
	return v->id();
}

//access id->f
Face *Mesh::id_face(int id) {
	return m_map_face[id];
}

//acess f->id
int Mesh::face_id(Face  *f) {
	return f->id();}

Edge *Mesh::create_edge(Vertex *v1, Vertex *v2) {
	EdgeKey key(v1, v2);

	Edge *e = NULL;

	if (m_map_edge.find(key) != m_map_edge.end()) {
		e = m_map_edge[key];
		return e;
	}

	e = new Edge;

	assert(e != NULL);
	m_map_edge.insert(std::pair<EdgeKey, Edge*>(key, e));
	m_edges.push_back(e);

	return e;
}

//access vertex->edge
Edge *Mesh::vertex_edge(Vertex *v0, Vertex *v1)
{
	EdgeKey key(v0, v1);
	return m_map_edge[key];
}

//access vertex->edge
HalfEdge *Mesh::vertex_halfedge(Vertex *v0, Vertex *v1) {
	Edge *e = vertex_edge(v0, v1);
	assert(e != NULL);
	HalfEdge *he = e->halfedge(0);
	if (he->vertex() == v1 && he->he_prev()->vertex() == v0) return he;
	he = e->halfedge(1);
	assert(he->vertex() == v1 && he->he_prev()->vertex() == v0);
	return he;
}


int Mesh::write_obj(const char * output) {
	FILE * _os = fopen(output, "w");
	assert(_os);

	//remove vertices
	for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); viter++) {
		Vertex *v = *viter;

		fprintf(_os, "v");
		for (int i = 0; i < 3; i++) {
			fprintf(_os, " %g", v->point()[i]);
		}
		fprintf(_os, "\n");
	}

	for (std::list<Vertex*>::iterator viter = m_vertices.begin(); viter != m_vertices.end(); viter++) {
		Vertex *v = *viter;

		fprintf(_os, "vn");
		for (int i = 0; i < 3; i++) {
			fprintf(_os, " %g", v->normal()[i]);
		}
		fprintf(_os, "\n");
	}

	for (std::list<Face*>::iterator fiter = m_faces.begin(); fiter != m_faces.end(); fiter++) {
		Face *f = *fiter;
		fprintf(_os, "f");

		HalfEdge *he = f->halfedge();
		do {
			fprintf(_os, " %d/%d", he->target()->id(),he->target()->id());
			he = he->he_next();
		} while (he != f->halfedge());


		fprintf(_os, "\n");
	}
	fclose(_os);
	return 0;
}

