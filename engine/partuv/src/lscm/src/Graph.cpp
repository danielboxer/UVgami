#include "Graph.h"
#include <iostream>
#include <fstream>

using namespace MeshLib;

Graph::Graph(Mesh *mesh) {
	m_mesh = mesh;
	m_root = NULL;
}

Graph::~Graph() {
	for (std::list<std::list<HalfEdge*>*>::iterator lhe = m_loops.begin(); lhe != m_loops.end(); lhe++) {
		std::list<HalfEdge*> *l = *lhe;
		delete l;
	}
}

void Graph::insert(Edge *e) {
	m_edges.insert(e);

	Vertex *v1 = m_mesh->edge_vertex_1(e);
	Vertex *v2 = m_mesh->edge_vertex_2(e);

	m_vertices.insert(v1);
	m_vertices.insert(v2);
}

void Graph::breadth_first_search() {
	for (std::set<Vertex*>::iterator vg = m_vertices.begin(); vg != m_vertices.end(); vg++) {
		Vertex *v = *vg;
		v_touched(v) = false;
		v_father(v) = NULL;
		m_root = v;
	}

	// Only "touch" the last one
	std::queue<Vertex*> q;
	q.push(m_root);
	v_touched(m_root) = true;
	v_father(m_root) = NULL;

	while (!q.empty()) {
		Vertex *head = q.front();
		q.pop();

		// Search around Vertex "head"
		// Assign father-to-child relation
		for (VertexEdgeIterator ve(head); !ve.end(); ++ve) {
			Edge *e = *ve;
			std::set<Edge*>::iterator se = m_edges.find(e);
			if (se == m_edges.end()) continue;
			Vertex *w = (m_mesh->edge_vertex_1(e) != head) ? m_mesh->edge_vertex_1(e) : m_mesh->edge_vertex_2(e);
			if (v_touched(w)) continue;
			v_father(w) = head;
			q.push(w);
			v_touched(w) = true;
		}
	}

	std::set<Edge*> edges = m_edges;
	for (std::set<Vertex*>::iterator sv = m_vertices.begin(); sv != m_vertices.end(); sv++) {
		Vertex *v = *sv;
		Vertex *w = v_father(v);
		if (w == NULL) continue;
		Edge *e = m_mesh->vertex_edge(v, w);
		std::set<Edge*>::iterator epos = edges.find(e);
		assert(epos != edges.end());
		edges.erase(epos);
	}

	for (std::set<Edge*>::iterator se = edges.begin(); se != edges.end(); se++) {
		Edge *e = *se;
		m_left_edges.push_back(e);
	}
	
}

void Graph::trace_to_root(Vertex *leaf, std::vector<Vertex*> &path) {
	path.push_back(leaf);
	Vertex *father = v_father(leaf);
	while (father != NULL) {
		path.push_back(father);
		father = v_father(father);
	}
}

void Graph::locate_loops() {
	int id = 1;
	for (std::list<Edge*>::iterator le = m_left_edges.begin(); le != m_left_edges.end(); le++) {
		// Loop starts with the edge
		Edge *e = *le;
		Vertex *v1 = m_mesh->edge_vertex_1(e);
		Vertex *v2 = m_mesh->edge_vertex_2(e);

		std::vector<Vertex*> trace1;
		std::vector<Vertex*> trace2;

		trace_to_root(v1, trace1);
		trace_to_root(v2, trace2);

		int s1 = trace1.size() - 1;
		int s2 = trace2.size() - 1;

		// Find the indices where two traces join together
		while (trace1[s1] == trace2[s2]) {
			s1--; s2--;
		}
		s1++; s2++;

		std::vector<Vertex*> path1;
		std::vector<Vertex*> path2;

		// Path until the first common vertex
		for (int i = 0; i < s1 + 1; i++) path1.push_back(trace1[i]);
		for (int i = 0; i < s2 + 1; i++) path2.push_back(trace2[i]);

		// Merge two paths to form a loop
		// From end of path1 to start of path1 and from start of path2 to end of path2

		std::list<HalfEdge*> *loop = new std::list < HalfEdge* > ;
		assert(loop);
		m_loops.push_back(loop);

		// From end of path1 to start of path1
		for (int i = path1.size() - 1; i > 0; i--) {
			Vertex* s = path1[i];
			Vertex* t = path1[i - 1];

			Edge* e = m_mesh->vertex_edge(s, t);
			// he from s to t

			HalfEdge *he = (e->halfedge(0)->target() == t) ? e->halfedge(0) : e->halfedge(1);
			assert(he->target() == t && he->source() == s);
			loop->push_back(he);
		}

		// Add the edge to the loop 
		HalfEdge *he = e->halfedge(0)->target() == v2 ? e->halfedge(0) : e->halfedge(1);
		loop->push_back(he);

		for (int i = 0; i < (int)path2.size() - 1; i++) {
			Vertex *s = path2[i];
			Vertex *t = path2[i + 1];
			Edge *e = m_mesh->vertex_edge(s, t);
			HalfEdge *he = e->halfedge(0)->target() == t ? e->halfedge(0) : e->halfedge(1);
			assert(he->source() == s);
			loop->push_back(he);
		}
		
		char graph1[1024];
		sprintf(graph1, "graph_%d.txt", id++);
		std::ofstream myfile;
		myfile.open(graph1);
		for (std::list<HalfEdge*>::iterator hiter = loop->begin(); hiter != loop->end(); hiter++) {
			HalfEdge * he = *hiter;
			Vertex *v = he->source();
			Point p = v -> point();
			myfile << p.x() << " " <<p.y() << " " << p.z() << "\n";
		}
		myfile.close();
	}
}

// Propagate through all the faces
void Graph::propagation() {
	for (MeshEdgeIterator me(m_mesh); !me.end(); me++) {
		Edge* e = *me;
		e_sharp(e) = false;
	}
	Face *head = NULL;
	for (MeshFaceIterator mf(m_mesh); !mf.end(); mf++) {
		Face* f = *mf;
		f_touched(f) = false;
		head = f;
	}
	assert(head != NULL);
	f_touched(head) = true;

	std::queue<Face*> q;
	q.push(head);

	while (!q.empty()) {
		Face* head = q.front();
		q.pop();

		HalfEdge *he = head->halfedge();
		for (int i = 0; i < 3; i++) {
			HalfEdge *sh = he->he_sym();
			if (sh != NULL) {
				Face *sf = sh->face();
				if (!f_touched(sf)) {
					q.push(sf);
					f_touched(sf) = true;
					e_sharp(he->edge()) = true;
				}
			}
			he = he->he_next();
		}
	}

	for (MeshEdgeIterator eiter(m_mesh); !eiter.end(); eiter++) {
		Edge * e = *eiter;
		e_sharp(e) = !e_sharp(e);
	}

	prune();

	for (MeshEdgeIterator me(m_mesh); !me.end(); me++) {
		Edge *e = *me;
		if (!e_sharp(e)) continue;
		e->string() += "sharp";
	}
}

void Graph::prune() {
	std::set<Vertex*> nodes;

	for (MeshVertexIterator viter(m_mesh); !viter.end(); viter++) {
		Vertex * pv = *viter;
		v_valence(pv) = 0;
		v_touched(pv) = false;
	}

	Vertex * head = NULL;
	for (MeshEdgeIterator eiter(m_mesh); !eiter.end(); eiter++) {
		Edge * e = *eiter;
		e_mark(e) = 0;

		if (!e_sharp(e)) continue;
		e_mark(e) = 1;

		Vertex * v1 = m_mesh->edge_vertex_1(e);
		head = v1;
	}

	Vertex* root = head;

	std::queue<Vertex*> vqueue;
	v_touched(head) = true;
	vqueue.push(head);
	v_father(head) = NULL;

	while (!vqueue.empty()) {
		head = vqueue.front();
		vqueue.pop();

		for (VertexEdgeIterator veiter(head); !veiter.end(); ++veiter) {
			Edge * e = *veiter;
			if (e_mark(e) == 1) {
				Vertex * v1 = m_mesh->edge_vertex_1(e);
				Vertex * v2 = m_mesh->edge_vertex_2(e);
				Vertex * tail = (v1 != head) ? v1 : v2;
				if (v_touched(tail)) continue;
				vqueue.push(tail);
				v_touched(tail) = true;
				v_father(tail) = head;
				e_mark(e) = 2;
			}
		}
	}

	std::list<Edge*> ecuts;

	for (MeshEdgeIterator eiter(m_mesh); !eiter.end(); eiter++) {
		Edge* e = *eiter;
		if (e_mark(e) == 1) {
			ecuts.push_back(e);
		}
		e_sharp(e) = false;
	}

	for (std::list<Edge*>::iterator eiter = ecuts.begin(); eiter != ecuts.end(); eiter++) {
		Edge* e = *eiter;
		e_sharp(e) = true;

		Vertex* v1 = m_mesh->edge_vertex_1(e);
		Vertex* v2 = m_mesh->edge_vertex_2(e);

		Vertex * pV = v1;
		while (v_father(pV) != NULL) {
			Vertex * fv = v_father(pV);
			Edge* e = m_mesh->vertex_edge(pV, fv);
			e_sharp(e) = true;
			pV = fv;
		}
		pV = v2;
		while (v_father(pV) != NULL) {
			Vertex * fv = v_father(pV);
			Edge * e = m_mesh->vertex_edge(pV, fv);
			e_sharp(e) = true;
			pV = fv;
		}
	}

	while (true) {
		Edge* link = NULL;
		int valence = 0;
		for (VertexEdgeIterator veiter(root); !veiter.end(); ++veiter) {
			Edge* e = *veiter;
			if (!e_sharp(e)) continue;
			valence++;
			link = e;
		}
		if (valence != 1) break;
		e_sharp(link) = false;

		Vertex* v1 = m_mesh->edge_vertex_1(link);
		Vertex* v2 = m_mesh->edge_vertex_2(link);

		root = (root != v1) ? v1 : v2;
	}
}
