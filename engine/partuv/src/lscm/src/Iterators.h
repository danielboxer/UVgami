#ifndef  _ITERATORS_H_
#define  _ITERATORS_H_

#include "Mesh.h"

namespace MeshLib{

	class Vertex;
	class HalfEdge;
	class Edge;
	class Face;
	class Mesh;

	//sequencial iterators

	//v->out halfedge
	class VertexOutHalfedgeIterator {
	public:
		VertexOutHalfedgeIterator(Mesh *  pMesh, Vertex *  v) {
			m_pMesh = pMesh; m_vertex = v; m_halfedge = m_pMesh->vertexMostClwOutHalfEdge(v);
		};

		~VertexOutHalfedgeIterator(){};
		void operator++() {
			assert(m_halfedge != NULL);
			if (m_halfedge == m_pMesh->vertexMostCcwOutHalfEdge(m_vertex))
				m_halfedge = NULL;
			else
				m_halfedge = m_pMesh->vertexNextCcwOutHalfEdge(m_halfedge);
		};

		HalfEdge * value() { return m_halfedge; };
		bool end(){ return m_halfedge == NULL; };
		HalfEdge * operator*() { return value(); };

	private:
		Mesh *        m_pMesh;
		Vertex *      m_vertex;
		HalfEdge * m_halfedge;
	};

	//v->in halfedge
	class VertexInHalfedgeIterator {
	public:
		VertexInHalfedgeIterator(Mesh *  pMesh, Vertex * v) {
			m_pMesh = pMesh; m_vertex = v; m_halfedge = m_pMesh->vertexMostClwInHalfEdge(v);
		};

		~VertexInHalfedgeIterator(){};
		void operator++() {
			assert(m_halfedge != NULL);
			if (m_halfedge == m_pMesh->vertexMostCcwInHalfEdge(m_vertex))
				m_halfedge = NULL;
			else
				m_halfedge = m_pMesh->vertexNextCcwInHalfEdge(m_halfedge);
		};

		HalfEdge * value() { return m_halfedge; };
		bool end(){ return m_halfedge == NULL; };
		HalfEdge * operator*() { return value(); };

	private:
		Mesh *        m_pMesh;
		Vertex *      m_vertex;
		HalfEdge * m_halfedge;
	};

	//v -> v
	class VertexVertexIterator {
	public:

		VertexVertexIterator(Vertex *  v) {
			m_vertex = v;
			m_halfedge = m_vertex->most_clw_out_halfedge();
		};

		~VertexVertexIterator(){};

		void operator++() {
			assert(m_halfedge != NULL);

			if (!m_vertex->boundary()) {
				if (m_halfedge != m_vertex->most_ccw_out_halfedge()) {
					m_halfedge = m_halfedge->ccw_rotate_about_source();
				}
				else {
					m_halfedge = NULL;
				}
				return;
			}

			if (m_vertex->boundary()) {
				if (m_halfedge == m_vertex->most_ccw_in_halfedge()) {
					m_halfedge = NULL;
					return;
				}

				HalfEdge * he = m_halfedge->ccw_rotate_about_source();

				if (he == NULL) {
					m_halfedge = m_vertex->most_ccw_in_halfedge();
				}
				else {
					m_halfedge = he;
				}
			}

			return;
		};


		Vertex * value() {
			if (!m_vertex->boundary()) {
				return m_halfedge->target();
			}

			if (m_halfedge != m_vertex->most_ccw_in_halfedge()) {
				return m_halfedge->target();
			}

			if (m_halfedge == m_vertex->most_ccw_in_halfedge()) {
				return m_halfedge->source();
			}
			return NULL;
		};

		Vertex * operator*() { return value(); };

		bool end(){ return m_halfedge == NULL; };

		void reset()	{ m_halfedge = m_vertex->most_clw_out_halfedge(); };

	private:
		Vertex *   m_vertex;
		HalfEdge * m_halfedge;
	};

	//v -> edge
	class VertexEdgeIterator {
	public:

		VertexEdgeIterator(Vertex *  v) {
			m_vertex = v;
			m_halfedge = m_vertex->most_clw_out_halfedge();
		};

		~VertexEdgeIterator(){};

		void operator++() {
			assert(m_halfedge != NULL);

			if (!m_vertex->boundary()) {
				if (m_halfedge != m_vertex->most_ccw_out_halfedge()) {
					m_halfedge = m_halfedge->ccw_rotate_about_source();
				}
				else {
					m_halfedge = NULL;
				}
				return;
			}

			if (m_vertex->boundary()) {
				if (m_halfedge == m_vertex->most_ccw_in_halfedge()) {
					m_halfedge = NULL;
					return;
				}

				HalfEdge * he = m_halfedge->ccw_rotate_about_source();

				if (he == NULL) {
					m_halfedge = m_vertex->most_ccw_in_halfedge();
				}
				else {
					m_halfedge = he;
				}
			}

			return;
		};

		Edge * value() {
			assert(m_halfedge != NULL);
			return m_halfedge->edge();
		};

		Edge * operator*() { return value(); };
		bool end(){ return m_halfedge == NULL; };
		void reset()	{ m_halfedge = m_vertex->most_clw_out_halfedge(); };

	private:
		Vertex *   m_vertex;
		HalfEdge * m_halfedge;
	};

	// v->face
	class VertexFaceIterator {
	public:
		VertexFaceIterator(Vertex * & v) {
			m_vertex = v;
			m_halfedge = m_vertex->most_clw_out_halfedge();
		};

		~VertexFaceIterator(){};

		void operator++() {
			assert(m_halfedge != NULL);

			if (m_halfedge == m_vertex->most_ccw_out_halfedge()) {
				m_halfedge = NULL;
				return;
			}
			m_halfedge = m_halfedge->ccw_rotate_about_source();
		};

		Face * value() { return m_halfedge->face(); };
		Face * operator*() { return value(); };
		bool end(){ return m_halfedge == NULL; };
		void reset()	{ m_halfedge = m_vertex->most_clw_out_halfedge(); };

	private:
		Vertex *   m_vertex;
		HalfEdge * m_halfedge;
	};

	// f -> halfedge
	class FaceHalfedgeIterator {
	public:

		FaceHalfedgeIterator(Face * f) {
			m_face = f;
			m_halfedge = f->halfedge();
		};

		~FaceHalfedgeIterator(){};

		void operator++() {
			assert(m_halfedge != NULL);
			m_halfedge = m_halfedge->he_next();

			if (m_halfedge == m_face->halfedge())
			{
				m_halfedge = NULL;
				return;
			};
		}

		HalfEdge * value() { return m_halfedge; };
		HalfEdge * operator*() { return value(); };

		bool end(){ return m_halfedge == NULL; };

	private:
		Face *        m_face;
		HalfEdge * m_halfedge;
	};

	// f -> edge
	class FaceEdgeIterator {
	public:

		FaceEdgeIterator(Face * f) {
			m_face = f;
			m_halfedge = f->halfedge();
		};

		~FaceEdgeIterator(){};

		void operator++() {
			assert(m_halfedge != NULL);
			m_halfedge = m_halfedge->he_next();

			if (m_halfedge == m_face->halfedge()) {
				m_halfedge = NULL;
				return;
			};
		}

		Edge * value() { return m_halfedge->edge(); };
		Edge * operator*() { return value(); };

		bool end(){ return m_halfedge == NULL; };

	private:
		Face  *       m_face;
		HalfEdge * m_halfedge;
	};

	// f -> vertex
	class FaceVertexIterator {
	public:

		FaceVertexIterator(Face * f) {
			m_face = f;
			m_halfedge = f->halfedge();
		};

		~FaceVertexIterator(){};

		void operator++() {
			assert(m_halfedge != NULL);
			m_halfedge = m_halfedge->he_next();

			if (m_halfedge == m_face->halfedge()) {
				m_halfedge = NULL;
				return;
			};
		}

		Vertex * value() { return m_halfedge->target(); };
		Vertex * operator*() { return value(); };

		bool end(){ return m_halfedge == NULL; };

	private:
		Face         * m_face;
		HalfEdge * m_halfedge;
	};

	// soild vertices
	class MeshVertexIterator {
	public:
		MeshVertexIterator(Mesh * pMesh) {
			m_pMesh = pMesh;
			m_iter = m_pMesh->vertices().begin();
		}

		Vertex * value() { return *m_iter; };

		void operator++()	 { ++m_iter; };
		void operator++(int) { ++m_iter; };

		bool end() { return m_iter == m_pMesh->vertices().end(); }

		Vertex * operator*(){ return value(); };

	private:
		Mesh * m_pMesh;
		std::list<Vertex*> ::iterator m_iter;
	};

	// soild faces
	class MeshFaceIterator {
	public:
		MeshFaceIterator(Mesh * pMesh) {
			m_pMesh = pMesh;
			m_iter = pMesh->faces().begin();
		}

		Face * value() { return *m_iter; };

		void operator++() { ++m_iter; };
		void operator++(int) { ++m_iter; };

		bool end() { return m_iter == m_pMesh->faces().end(); }

		Face * operator*(){ return value(); };

	private:
		Mesh * m_pMesh;
		std::list<Face*>::iterator  m_iter;
	};

	// soild edges
	class MeshEdgeIterator {
	public:
		MeshEdgeIterator(Mesh * pMesh) {
			m_pMesh = pMesh;
			m_iter = m_pMesh->edges().begin();
		}

		Edge * value() { return *m_iter; };

		void operator++() { ++m_iter; };
		void operator++(int) { m_iter++; };

		bool end() { return m_iter == m_pMesh->edges().end(); }

		Edge * operator*(){ return value(); };

	private:
		Mesh * m_pMesh;
		std::list<Edge*>::iterator m_iter;
	};

	// soild halfedges
	class MeshHalfEdgeIterator {
	public:
		MeshHalfEdgeIterator(Mesh * pMesh) {
			m_pMesh = pMesh;
			m_iter = m_pMesh->edges().begin();
			m_id = 0;
		}

		HalfEdge * value() { Edge * e = *m_iter; return e->halfedge(m_id); };

		void operator++() {
			++m_id;

			switch (m_id) {
			case 1: {
				Edge * e = *m_iter;
				if (e->halfedge(m_id) == NULL) {
					m_id = 0;
					++m_iter;
				}
			}
				break;
			case 2:
				m_id = 0;
				++m_iter;
				break;
			}
		};

		bool end() { return m_iter == m_pMesh->edges().end(); }
		HalfEdge * operator*(){ return value(); };

	private:
		// HalfEdge * m_he;
		Mesh *	 m_pMesh;
		std::list<Edge*>::iterator m_iter;
		int  m_id;
	};
} //solid 

#endif
