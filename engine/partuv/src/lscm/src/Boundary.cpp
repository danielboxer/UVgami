#include "Boundary.h"

using namespace MeshLib;

Loop::Loop(Mesh *mesh, HalfEdge *he) {
	// An arbitrary loop starting wtih a halfedge and ends with it
	HalfEdge *he1 = he;
	do {
		Vertex *v = he1->target();
		he1 = m_mesh->vertexMostClwOutHalfEdge(v); // Along the boundary if there is 
		m_halfedges.push_back(he);
	} while (he1 != m_halfedge);
}

Loop::~Loop() {
	m_halfedges.clear();
}

Boundary::Boundary(Mesh *mesh) {
	std::set<HalfEdge*> boundary_hes;
	for (MeshEdgeIterator me(mesh); !me.end(); me++) {
		Edge *e = *me;
		if (!mesh->is_boundary(e)) continue; // Find a boundary edge

		HalfEdge *he = e->halfedge(0); // The halfedge of the boundary edge
		boundary_hes.insert(he); // Stores all the halfedges of all the boundaries
		// No duplicate halfedges will be stored because of "std::set"
	}

	while (!boundary_hes.empty()) {
		std::set<HalfEdge*>::iterator she = boundary_hes.begin(); // Start with a halfedge
		HalfEdge *he = *she;
		Loop *l = new Loop(mesh, he); // Build a loop, must be a boundary because "MostClwOut"
		assert(l);
		m_loops.push_back(l); // Store a Loop/boundary

		// Remove all the halfedges in the current Loop
		for (std::list<HalfEdge*>::iterator lhe = l->halfedges().begin(); lhe != l->halfedges().end(); lhe++) {
			HalfEdge *he = *lhe;
			she = boundary_hes.find(he);
			if (she == boundary_hes.end()) continue;
			boundary_hes.erase(she);
		}
	}
}
