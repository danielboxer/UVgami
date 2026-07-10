#include "HalfEdge.h"

using namespace MeshLib;

HalfEdge *HalfEdge::ccw_rotate_about_target() {
	HalfEdge * he_dual = he_sym();
	if (he_dual == NULL) return NULL;
	return he_dual->he_prev();
}

HalfEdge *HalfEdge::clw_rotate_about_target() {
	HalfEdge * he = he_next()->he_sym();
	return he;
}

HalfEdge *HalfEdge::ccw_rotate_about_source() {
	HalfEdge * he = he_prev()->he_sym();
	return he;
}

HalfEdge *HalfEdge::clw_rotate_about_source() {
	HalfEdge * he = he_sym();
	if (he == NULL) return NULL;
	return he->he_next();
}
