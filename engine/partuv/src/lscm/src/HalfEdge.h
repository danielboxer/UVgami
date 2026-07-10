#ifndef _MESHLIB_HALFEDGE_H_
#define _MESHLIB_HALFEDGE_H_

#include <cassert>
#include <cmath>
#include <string>
#include "Trait.h"
#include "Edge.h"

namespace MeshLib {
	class Vertex;
	class Edge;
	class Face;

	class HalfEdge {
	public:
		HalfEdge() { m_edge = NULL; m_prev = NULL; m_next = NULL; m_face = NULL; m_trait = NULL; m_vertex = NULL; m_string = ""; };
		~HalfEdge() {};

		Edge     *& edge()    { return m_edge; };
		Vertex   *& vertex()  { return m_vertex; };
		Vertex   *& target()  { return m_vertex; };
		Vertex   *& source()  { return m_prev->vertex(); };
		HalfEdge *& he_prev() { return m_prev; };
		HalfEdge *& he_next() { return m_next; };

		HalfEdge *& he_sym()  { return m_edge->other(this); };
		Face     *& face()    { return m_face; };

		std::string &string() { return m_string; };
		Trait *&trait() { return m_trait; };

		HalfEdge *ccw_rotate_about_target();
		HalfEdge *clw_rotate_about_target();
		HalfEdge *ccw_rotate_about_source();
		HalfEdge *clw_rotate_about_source();

	private:
		Edge *m_edge;
		Face *m_face;
		Vertex *m_vertex; // target vertex
		HalfEdge *m_prev;
		HalfEdge *m_next;
		Trait * m_trait;
		std::string m_string;
	};

}
#endif