#include "Face.h"
#include "HalfEdge.h"
#include "Vertex.h"
#include "Point.h"

using namespace MeshLib;

bool Face::include_edge(Edge *e) {
	HalfEdge *he = m_halfedge;
	if (he->edge() == e || he->he_next()->edge() == e || he->he_prev()->edge() == e)
		return true;
	return false;
}

bool Face::include_vertex(Vertex *v) {
	HalfEdge *he = m_halfedge;
	if (he->target() == v || he->source() == v || he->he_next()->target() == v)
		return true;
	return false;
}

Point Face::normal() {
	HalfEdge *he = m_halfedge;
	Point p1 = he->target()->point() - he->source()->point();
	Point p2 = he->he_next()->target()->point() - he->target()->point();
	Point n = p1^p2;
	n /= n.norm();
	return n;
}
