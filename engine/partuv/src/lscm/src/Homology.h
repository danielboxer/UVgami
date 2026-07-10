#ifndef _HOMOLOGY_H_
#define _HOMOLOGY_H_

#include <queue>
#include <vector>

#include "Mesh.h"
#include "Boundary.h"
#include "Iterators.h"
#include "FormTrait.h"
#include "Graph.h"

namespace MeshLib {
	class Homology {
	public:
		Homology(Mesh * mesh);
		~Homology();
		void compute_cut_graph(const char * prefix);
		void compute_basis();
		void output(const char * prefix);

	protected:
		Mesh * m_mesh;
		Graph      * m_graph;
		std::list<std::list<HalfEdge*>*> m_loops;
	};
}

#endif
