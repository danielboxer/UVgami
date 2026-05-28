#ifndef _BOUNDARY_H_
#define _BOUNDARY_H_

#include <vector>
#include <list>
#include <set>

#include "Mesh.h"
#include "Iterators.h"

namespace MeshLib {
	class Loop {
	public:
		Loop(Mesh *mesh, HalfEdge *he);
		~Loop();

		std::list<HalfEdge*> &halfedges() { return m_halfedges; };
		double length() { return m_length(); };

	protected:
		Mesh *m_mesh;
		HalfEdge *m_halfedge;
		double m_length();
		std::list<HalfEdge*> m_halfedges;
	};

	class Boundary {
	public:
		Boundary(Mesh *mesh);
		~Boundary();
		std::list<Loop*> &loops() { return m_loops; };

	protected:
		Mesh *m_mesh;
		std::list<Loop*> m_loops;
	};
}

#endif
