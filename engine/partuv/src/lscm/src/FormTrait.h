#ifndef  _FORMTRAIT_H_
#define  _FORMTRAIT_H_

#include <map>
#include <vector>
#include <iostream>

#include "Mesh.h"
#include "Trait.h"
#include "Point.h"

namespace MeshLib {
	class FaceTrait : public Trait {
	public:
		bool m_touched;
		FaceTrait() { m_touched = false; };
		~FaceTrait() {};
	};

	class VertexTrait : public  Trait {
	public:
		int m_index;
		int m_valence;
		Point2 m_uv;
		Vertex* m_father;
		bool m_touched;
		VertexTrait() { m_index = 0; m_valence = 0; m_touched = false; };
		~VertexTrait() {};
	};

	class EdgeTrait : public  Trait {
	public:
		double m_length;
		double m_weight;
		int m_mark;
		bool m_sharp;
		std::string m_string;
		EdgeTrait() { m_length = 0.0; m_weight = 0.0; m_sharp = false; m_mark = 0; };
		~EdgeTrait(){};

		void read() {
			std::string s(m_string);
			string_token_iterator str_iter(s, " \n");

			for (; str_iter != string_token_iterator(); str_iter++){
				std::string str = *str_iter;
				if (str == "sharp") {
					m_sharp = true;
					break;
				}
			}
		};
	};

	class HalfEdgeTrait : public  Trait {
	public:
		double m_angle;
		HalfEdge * m_next;
		HalfEdge * m_prev;
		Point m_s;
		HalfEdgeTrait() { m_angle = 0.0; m_prev = NULL; m_next = NULL; };
		~HalfEdgeTrait() {};
	};

	inline bool & v_touched(Vertex* v) {
		VertexTrait * pVT = (VertexTrait*)v->trait();
		return pVT->m_touched;
	}

	inline Vertex* & v_father(Vertex* v) {
		VertexTrait * pVT = (VertexTrait*)v->trait();
		return pVT->m_father;
	}


	inline int & v_idx(Vertex* v) {
		VertexTrait * pVT = (VertexTrait*)v->trait();
		return pVT->m_index;
	}

	inline int & v_valence(Vertex* v) {
		VertexTrait * pVT = (VertexTrait*)v->trait();
		return pVT->m_valence;
	}

	inline int & e_mark(Edge* e) {
		EdgeTrait * pET = (EdgeTrait*)e->trait();
		return pET->m_mark;
	}

	inline double & e_l(Edge* e) {
		EdgeTrait * pET = (EdgeTrait*)e->trait();
		return pET->m_length;
	}

	inline double & e_w(Edge* e) {
		EdgeTrait * pET = (EdgeTrait*)e->trait();
		return pET->m_weight;
	}

	inline bool & e_sharp(Edge* e) {
		EdgeTrait * pET = (EdgeTrait*)e->trait();
		return pET->m_sharp;
	}

	inline std::string & e_string(Edge* e) {
		EdgeTrait * pET = (EdgeTrait*)e->trait();
		return pET->m_string;
	}

	inline HalfEdge* & c_next(HalfEdge * c) {
		HalfEdgeTrait * pHT = (HalfEdgeTrait*)c->trait();
		return pHT->m_next;
	}

	inline HalfEdge* & c_prev(HalfEdge * c) {
		HalfEdgeTrait * pHT = (HalfEdgeTrait*)c->trait();
		return pHT->m_prev;
	}

	inline double & c_a(HalfEdge * c) {
		HalfEdgeTrait * pHT = (HalfEdgeTrait*)c->trait();
		return pHT->m_angle;
	}

	inline Point & c_s(HalfEdge * c) {
		HalfEdgeTrait * pHT = (HalfEdgeTrait*)c->trait();
		return pHT->m_s;
	}

	inline Point2 & v_uv(Vertex* v) {
		VertexTrait * pVT = (VertexTrait*)v->trait();
		return pVT->m_uv;
	}

	inline bool & f_touched(Face* f) {
		FaceTrait * pFT = (FaceTrait*)f->trait();
		return pFT->m_touched;
	}

	class FormTrait {
	public:
		FormTrait(Mesh * mesh);
		~FormTrait();

	protected:
		Mesh* m_mesh;

		VertexTrait* m_vertex_traits;
		EdgeTrait * m_edge_traits;
		HalfEdgeTrait* m_halfedge_traits;
		FaceTrait* m_face_traits;

	private:
		//allocate and dellocate traits

		void allocate_vertex_trait();
		void allocate_edge_trait();
		void allocate_halfedge_trait();
		void allocate_face_trait();

		void dellocate_vertex_trait();
		void dellocate_edge_trait();
		void dellocate_halfedge_trait();
		void dellocate_face_trait();
	};
}

#endif
