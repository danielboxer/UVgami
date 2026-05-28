#include <assert.h>
#include <iostream>
#include "Edge.h"
#include "Vertex.h"

using namespace MeshLib;

EdgeKey::EdgeKey(Vertex * v1, Vertex* v2) {
	assert(v1->id() != v2->id());
	
	if (v1->id() < v2->id()) {
		m_vertices[0] = v1;
		m_vertices[1] = v2;
	}
	else {
		m_vertices[0] = v2;
		m_vertices[1] = v1;
	}
}

bool EdgeKey::operator<(const EdgeKey & key) const {
	if (m_vertices[0]->id() < key.m_vertices[0]->id()) return true;
	if (m_vertices[0]->id() > key.m_vertices[0]->id()) return false;

	if (m_vertices[1]->id() < key.m_vertices[1]->id()) return true;
	if (m_vertices[1]->id() > key.m_vertices[1]->id()) return false;

	return false;
}

bool EdgeKey::operator==(const EdgeKey & key) const {
	if (m_vertices[0]->id() != key.m_vertices[0]->id()) return false;
	if (m_vertices[1]->id() != key.m_vertices[1]->id()) return false;

	return true;
}
