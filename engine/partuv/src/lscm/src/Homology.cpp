#include "Homology.h"
#include <iostream>

using namespace MeshLib;

Homology::Homology(Mesh *mesh) {
	m_mesh = mesh;
	m_graph = new Graph(m_mesh);
}

Homology::~Homology() {
	delete m_graph;

	for (std::list<std::list<HalfEdge*>*>::iterator liter = m_loops.begin(); liter != m_loops.end(); liter++) {
		std::list<HalfEdge*>* l = *liter;
		delete l;
	}
	m_loops.clear();
}

void Homology::compute_basis() {
	m_graph->propagation();
	for (MeshEdgeIterator me(m_mesh); !me.end(); me++) {
		Edge *e = *me;
		if (!e_sharp(e)) continue;
		m_graph->insert(e);
	}
	m_graph->breadth_first_search();
	m_graph->locate_loops();

	std::list<std::list<HalfEdge*>*> &loops = m_graph->loops();

	for (std::list<std::list<HalfEdge*>*>::iterator llh = loops.begin(); llh != loops.end(); llh++) {
		std::list<HalfEdge*>* l = *llh;
		std::list<HalfEdge*>* loop = new std::list < HalfEdge* > ;
		assert(loop);
		*loop = *l;
		m_loops.push_back(loop);
	}
}

