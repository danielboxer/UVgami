#ifndef _MESHLIB_VERTEX_H_
#define _MESHLIB_VERTEX_H_

#include <cassert>
#include <cmath>
#include <string>
#include "Point.h"
#include "Face.h"
#include "Trait.h"
#include "HalfEdge.h"

namespace MeshLib {
	class HalfEdge;

	class Vertex {
	public:
		Vertex() { m_halfedge = NULL; m_boundary = false; m_trait = NULL; };
		~Vertex(){};

		Point &point() { return m_point; };
		Point &normal() { return m_normal; };
		Point2 &uv() { return m_uv; };

		HalfEdge * & halfedge() { return m_halfedge; };
		std::string &string() { return m_string; };
		int &id(){ return m_id; };
		bool &boundary(){ return m_boundary; };

		Trait *&trait(){ return m_trait; };

		HalfEdge *most_ccw_in_halfedge();
		HalfEdge *most_clw_in_halfedge() ;
		HalfEdge *most_ccw_out_halfedge();
		HalfEdge *most_clw_out_halfedge();

	private:
		int m_id;

		Point m_point;
		Point m_normal;
		Point2 m_uv;

		HalfEdge *m_halfedge;
		bool m_boundary;
		Trait *m_trait;
		std::string m_string;
	};
}

#endif
