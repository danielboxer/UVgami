#ifndef _MESHLIB_MESH_H_
#define _MESHLIB_MESH_H_

#define MAX_LINE 2048

#include <assert.h>
#include <math.h>
#include "Edge.h"
#include "Face.h"
#include "HalfEdge.h"
#include "Vertex.h"
#include "Point.h"
#include <list>
#include <Eigen/Core>
#include <map>

namespace MeshLib {
	class Vertex;
	class HalfEdge;
	class Edge;
	class Face;

	class Mesh {
	public:
		typedef Vertex   * pVertex;
		typedef HalfEdge * pHalfEdge;
		typedef Edge     * pEdge;
		typedef Face     * pFace;
		typedef Mesh     * pMesh;

		Mesh(){};
		~Mesh();

		int copy(Mesh &mesh);
		int read_obj(const char *input);

		void compute_pinned_vertices();
		void compute_pinned_vertices_2();
		void compute_pinned_vertices_3();
		int from_igl(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		int write_obj(const char *output);

		int numVertices();
		int numEdges();
		int numFaces();

		bool is_boundary(Vertex  *v);
		bool is_boundary(Edge    *e);
		bool is_boundary(HalfEdge  *he);

		Vertex *id_vertex(int id);
		int vertex_id(Vertex* v);

		Face *id_face(int id);
		int face_id(Face *f);

		Edge *vertex_edge(Vertex *v1, Vertex *v2);

		HalfEdge *vertex_halfedge(Vertex *v1, Vertex *v2);
		HalfEdge *corner(Vertex *v, Face *f);

		Face *halfedge_face(HalfEdge *he);
		Vertex *halfedge_vertex(HalfEdge *he);

		Vertex *edge_vertex_1(Edge *e);
		Vertex *edge_vertex_2(Edge *e);

		Face *edge_face_1(Edge *e);
		Face *edge_face_2(Edge *e);

		//Euler operations
		HalfEdge *vertexMostClwOutHalfEdge(Vertex  *v);
		HalfEdge *vertexNextCcwOutHalfEdge(HalfEdge  *he);

		HalfEdge *vertexMostCcwOutHalfEdge(Vertex  *v);
		HalfEdge *vertexNextClwOutHalfEdge(HalfEdge  *he);

		HalfEdge *vertexMostClwInHalfEdge(Vertex  *v);
		HalfEdge *vertexNextCcwInHalfEdge(HalfEdge  *he);

		HalfEdge *vertexMostCcwInHalfEdge(Vertex  *v);
		HalfEdge *vertexNextClwInHalfEdge(HalfEdge  *he);

		HalfEdge *faceMostClwHalfEdge(Face  *f);
		HalfEdge *faceMostCcwHalfEdge(Face  *f);
		HalfEdge *faceNextCcwHalfEdge(HalfEdge  *he);
		HalfEdge *faceNextClwHalfEdge(HalfEdge  *he);

		double edge_length(Edge *e);
		std::list<Edge*> &edges() { return m_edges; };
		std::list<Face*> &faces() { return m_faces; };
		std::list<Vertex*> &vertices() { return m_vertices; };

	protected:
		std::list<Edge*> m_edges;
		std::list<Face*> m_faces;
		std::list<Vertex*> m_vertices;

		std::map<int, Vertex*> m_map_vertex;
		std::map<int, Face*> m_map_face;
		std::map<EdgeKey, Edge*> m_map_edge;
		// std::map<int, Edge> m_map_edge;
	    
	protected:
		Vertex *create_vertex(int id);
		Edge *create_edge(Vertex *v1, Vertex *v2);
		Face *create_face(Vertex *v[], int id);
	};
}

#endif
