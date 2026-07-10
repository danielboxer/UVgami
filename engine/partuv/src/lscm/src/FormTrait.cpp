#include "FormTrait.h"

using namespace MeshLib;

FormTrait::FormTrait(Mesh * mesh) {
	m_mesh = mesh;

	allocate_vertex_trait();
	allocate_edge_trait();
	allocate_halfedge_trait();
	allocate_face_trait();
}

FormTrait::~FormTrait() {
	dellocate_vertex_trait();
	dellocate_edge_trait();
	dellocate_halfedge_trait();
	dellocate_face_trait();
}


void FormTrait::allocate_vertex_trait() {
	m_vertex_traits = new VertexTrait[m_mesh->numVertices()];
	assert(m_vertex_traits);

	int id = 0;
	for (std::list<Vertex*>::iterator viter = m_mesh->vertices().begin(); viter != m_mesh->vertices().end(); viter++) {
		Vertex* v = *viter;
		v->trait() = (Trait*)&m_vertex_traits[id++];
	}
}

void FormTrait::allocate_edge_trait() {
	m_edge_traits = new EdgeTrait[m_mesh->numEdges()];
	assert(m_edge_traits);

	int id = 0;
	for (std::list<Edge*>::iterator eiter = m_mesh->edges().begin(); eiter != m_mesh->edges().end(); eiter++) {
		Edge* e = *eiter;
		e->trait() = (Trait*)&m_edge_traits[id++];
	}
}

void FormTrait::allocate_halfedge_trait() {
	m_halfedge_traits = new HalfEdgeTrait[m_mesh->numFaces() * 3];
	assert(m_halfedge_traits);

	int id = 0;
	for (std::list<Face*>::iterator fiter = m_mesh->faces().begin(); fiter != m_mesh->faces().end(); fiter++) {
		Face* f = *fiter;
		HalfEdge* he = f->halfedge();
		for (int k = 0; k < 3; k++) {
			he->trait() = (Trait*)&m_halfedge_traits[id++];
			he = he->he_next();
		}
	}
}

void FormTrait::allocate_face_trait() {
	m_face_traits = new FaceTrait[m_mesh->numFaces()];
	assert(m_face_traits);

	int id = 0;
	for (std::list<Face*>::iterator fiter = m_mesh->faces().begin(); fiter != m_mesh->faces().end(); fiter++) {
		Face* f = *fiter;
		f->trait() = (Trait*)&m_face_traits[id++];
	}
}

void FormTrait::dellocate_vertex_trait() {
	for (std::list<Vertex*>::iterator viter = m_mesh->vertices().begin(); viter != m_mesh->vertices().end(); viter++) {
		Vertex* v = *viter;
		v->trait() = NULL;
	}
	delete[] m_vertex_traits;
}

void FormTrait::dellocate_edge_trait() {
	for (std::list<Edge*>::iterator eiter = m_mesh->edges().begin(); eiter != m_mesh->edges().end(); eiter++) {
		Edge* e = *eiter;
		e->trait() = NULL;
	}
	delete[] m_edge_traits;
}

void FormTrait::dellocate_halfedge_trait() {
	for (std::list<Face*>::iterator fiter = m_mesh->faces().begin(); fiter != m_mesh->faces().end(); fiter++) {
		Face* f = *fiter;
		HalfEdge* he = f->halfedge();
		for (int k = 0; k < 3; k++) {
			he->trait() = NULL;
			he = he->he_next();
		}
	}
	delete[] m_halfedge_traits;
}

void FormTrait::dellocate_face_trait() {
	for (std::list<Face*>::iterator fiter = m_mesh->faces().begin(); fiter != m_mesh->faces().end(); fiter++) {
		Face* f = *fiter;
		f->trait() = NULL;
	}
	delete[] m_face_traits;
}
