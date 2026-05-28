#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <queue>
#include <vector>
#include <set>

#include "Mesh.h"
#include "Boundary.h"
#include "Iterators.h"
#include "FormTrait.h"

namespace MeshLib {
	class Graph {
	public:
		Graph(Mesh *mesh);
		~Graph();
		void insert(Edge *e);
		void breadth_first_search();
		void locate_loops();
		void propagation();
		void prune();
		std::list<std::list<HalfEdge*>*> & loops() {
			return m_loops;
		};
	protected:
		Mesh *m_mesh;
		std::set<Edge*> m_edges;
		std::set<Vertex*> m_vertices;
		std::list<Edge*> m_left_edges;
		std::list<std::list<HalfEdge*>*> m_loops;
		Vertex *m_root;

		void trace_to_root(Vertex *leaf, std::vector<Vertex*> &path);
	};
}

#endif
