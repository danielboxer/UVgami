#ifndef _MESHLIB_FACE_H_
#define _MESHLIB_FACE_H_

#include <cassert>
#include <cmath>
#include "Trait.h"
#include "HalfEdge.h"

namespace MeshLib {
	class Edge;
	class Vertex;
	class Trait;
	class HalfEdge;
	class Point;

	class Face {
	public:
		Face() { m_id = 0; m_halfedge = NULL; m_trait = NULL; m_string = ""; };
		~Face() {};

		int &id() { return m_id; };
		const int id() const { return m_id; };
		HalfEdge *&halfedge() { return m_halfedge; };
		Trait *&trait() { return m_trait; };
		std::string &string() { return m_string; };

		bool operator==(const Face &f) const;
		bool include_vertex(Vertex *v);
		bool include_edge(Edge *e);
		Point normal();
		
	private:
		int m_id;
		HalfEdge *m_halfedge;
		Trait *m_trait;
		std::string m_string;
	};
}
#endif