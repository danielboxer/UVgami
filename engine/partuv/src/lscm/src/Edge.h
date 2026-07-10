#ifndef _MESHLIB_EDGE_H_
#define _MESHLIB_EDGE_H_

#include <cassert>
#include <cmath>
#include <string>
#include "Trait.h"

namespace MeshLib {
	class HalfEdge;
	class Vertex;
	class Edge {
	public:
		Edge() { m_halfedge[0] = NULL; m_halfedge[1] = NULL; m_trait = NULL; m_string = ""; };
		~Edge() {};

		HalfEdge *&halfedge(int i) { assert(0 <= i && i < 2); return m_halfedge[i]; };
		bool boundary() { return (m_halfedge[0] == NULL && m_halfedge[1] != NULL) || (m_halfedge[0] != NULL && m_halfedge[1] == NULL); };
		HalfEdge *&other(HalfEdge *he) { return he != m_halfedge[0] ? m_halfedge[0] : m_halfedge[1]; };
		std::string & string() { return m_string; };
		Trait *&trait() { return m_trait; };

	private:
		HalfEdge *m_halfedge[2];
		Trait *m_trait;
		std::string m_string;
	};

	class EdgeKey {
	public:
		~EdgeKey() {};
		EdgeKey(Vertex *v1, Vertex *v2);
		bool operator<(const EdgeKey & key) const ;
		bool operator==(const EdgeKey & key) const;

	private:
		Vertex *m_vertices[2];
	};
}

#endif
