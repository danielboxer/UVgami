#include "LSCM.h"
#include <assert.h>
#include <math.h>
#include <float.h>
#include <Eigen/Sparse>
#include <iostream>
#include <sstream>
#include <Eigen/IterativeLinearSolvers>
// #include <Eigen/PardisoSupport>
#include <Eigen/Eigenvalues>

#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>
#include <chrono>
#include <Eigen/Sparse>
// #include <Eigen/PardisoSupport>
#include <Eigen/Cholesky>

// #define VERBOSE
using namespace MeshLib;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::VectorXd VectorXd;
// typedef Eigen::SparseMatrix<double> SpMat;
// typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrixRM;

LSCM::LSCM(Mesh * mesh) {
	m_mesh = mesh;
}

LSCM::~LSCM(){}



// Function to check if a sparse matrix is symmetric
bool isSymmetric(const SpMat& mat, double tol = 1e-10) {
    if (mat.rows() != mat.cols()) {
        std::cerr << "Symmetry Check Failed: Matrix is not square (" 
                  << mat.rows() << "x" << mat.cols() << ").\n";
        return false;
    }

    // Compute the difference between the matrix and its transpose
    Eigen::SparseMatrix<double, Eigen::ColMajor> mat_Transposed = mat.transpose();

    Eigen::SparseMatrix<double> diff = mat - mat_Transposed;
    double norm = diff.norm();

    // std::cout << "Symmetry Check: ||A - A^T|| = " << norm << "\n";

    if (norm > tol) {
        std::cerr << "Symmetry Check Failed: Matrix is not symmetric within tolerance " 
                  << tol << ".\n";
        return false;
    }

    // std::cout << "Symmetry Check Passed: Matrix is symmetric.\n";
    return true;
}

// Function to check if a sparse matrix is positive definite
bool isPositiveDefinite(const SpMat& mat) {
    // Attempt a Cholesky decomposition
    Eigen::SimplicialLLT<SpMat> llt;
    llt.compute(mat);

    if (llt.info() == Eigen::Success) {
        // std::cout << "Positive Definiteness Check Passed: Matrix is positive definite.\n";
        return true;
    } else {
        std::cerr << "Positive Definiteness Check Failed: Matrix is not positive definite.\n";
        return false;
    }
}

void LSCM::set_coefficients() {
	for (MeshEdgeIterator eiter(m_mesh); !eiter.end(); ++eiter) {
		Edge * e = *eiter;
		e_l(e) = m_mesh->edge_length(e);
	}

	for (MeshFaceIterator fiter(m_mesh); !fiter.end(); ++fiter) {
		Point  p[3];
		Face *f = *fiter;

		double l[3];
		HalfEdge * he = f->halfedge();
		for (int j = 0; j < 3; j++) {
			Edge * e = he->edge();
			l[j] = e_l(e);
			he = he->he_next();
		}

		double a = acos((l[0]*l[0] + l[2]*l[2] - l[1]*l[1]) / (2*l[0]*l[2]));

		p[0] = Point(0, 0, 0);
		p[1] = Point(l[0], 0, 0);
		p[2] = Point(l[2]*cos(a), l[2]*sin(a), 0);

		Point n = (p[1]-p[0]) ^ (p[2]-p[0]);
		double area = n.norm() / 2.0;
		n /= area;

		he = f->halfedge();
		for (int j = 0; j < 3; j++) {
			Point s = (n ^ (p[(j + 1) % 3] - p[j])) / sqrt(area);
			c_s(he) = s;
			he = he->he_next();
		}
	}
}


void LSCM::set_coefficients_faster()
{
    // 1) Precompute and cache all edge lengths in a vector
    //    (so we don't call m_mesh->edge_length() repeatedly).
    //    Also gather faces into a face vector for parallel iteration.

    std::vector<Edge*> edgeVec;
    edgeVec.reserve(m_mesh->numEdges());
    for (MeshEdgeIterator eiter(m_mesh); !eiter.end(); ++eiter)
    {
        edgeVec.push_back(*eiter);
    }
    // Compute and store lengths
    for (auto e : edgeVec)
    {
        e_l(e) = m_mesh->edge_length(e);
    }

    std::vector<Face*> faceVec;
    faceVec.reserve(m_mesh->numFaces());
    for (MeshFaceIterator fiter(m_mesh); !fiter.end(); ++fiter)
    {
        faceVec.push_back(*fiter);
    }

    // 2) For each face, compute the local 2D embedding and store c_s(he).
    //    We remove acos/cos/sin calls by using:
    //
    //      x2 = ( l0^2 + l2^2 - l1^2 ) / ( 2*l0 )
    //      y2 = sqrt( l2^2 - x2^2 )
    //
    //    This avoids expensive trig calls.

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < (int)faceVec.size(); ++i)
    {
        Face* f = faceVec[i];

        // Gather lengths of the 3 edges
        double l[3];
        HalfEdge* he = f->halfedge();
        for (int j = 0; j < 3; j++)
        {
            Edge* e = he->edge();
            l[j]    = e_l(e);
            he      = he->he_next();
        }

        // Now compute local coordinates in 2D:
        //
        //   p[0] = (0,    0)
        //   p[1] = (l[0], 0)
        //   p[2] = (x2,   y2)

        double x2 = (l[0]*l[0] + l[2]*l[2] - l[1]*l[1]) / (2.0 * l[0]);
        // Guard against tiny negative rounding inside sqrt
        double temp = std::max(l[2]*l[2] - x2*x2, 0.0);
        double y2   = std::sqrt(temp);

        Point p[3];
        p[0] = Point(0.0,     0.0, 0.0);
        p[1] = Point(l[0],    0.0, 0.0);
        p[2] = Point(x2,      y2,  0.0);

        // Compute face normal in 3D to get area
        //   n = (p1 - p0) x (p2 - p0)
        // area = 0.5 * ||n||
        // Then scale n by area in the same step
        Point v01  = p[1] - p[0];
        Point v02  = p[2] - p[0];
        Point n    = v01 ^ v02;
        double area = n.norm() * 0.5;

        // If area is extremely small, continue or set c_s(he) to zero
        if (area < 1e-14) {
            // Degenerate face; handle gracefully
            he = f->halfedge();
            for (int j = 0; j < 3; j++)
            {
                c_s(he) = Point(0.0, 0.0, 0.0);
                he      = he->he_next();
            }
            continue;
        }

        // (Optionally) normalize n by area for convenience
        n /= (area); // So now n has length ~2

        // For each halfedge, c_s(he) = ( n x (p[(j+1)%3] - p[j]) ) / sqrt(area)
        // We do sqrt(area) once, though we must re-check degenerate faces
        double sqrtArea = std::sqrt(area);

        he = f->halfedge();
        for (int j = 0; j < 3; j++)
        {
            Point edgeVec2D = p[(j + 1) % 3] - p[j];
            Point s         = (n ^ edgeVec2D) / sqrtArea;
            c_s(he)         = s;
            he              = he->he_next();
        }
    }
}

void LSCM::set_coefficients_parallel()
{
    // 1) Gather all edges into a std::vector
    std::vector<Edge*> edges;
    edges.reserve(m_mesh->numEdges());
    for (MeshEdgeIterator eiter(m_mesh); !eiter.end(); ++eiter) {
        edges.push_back(*eiter);
    }

    // 2) Gather all faces into a std::vector
    std::vector<Face*> faces;
    faces.reserve(m_mesh->numFaces());
    for (MeshFaceIterator fiter(m_mesh); !fiter.end(); ++fiter) {
        faces.push_back(*fiter);
    }

    // ==============
    // Parallel section for edges
    // ==============
    // We assume e_l(e) is something like a std::map<Edge*, double> or
    // a custom array. You must ensure it is thread-safe.
#pragma omp parallel for
    for (int i = 0; i < (int)edges.size(); i++) {
        Edge* e = edges[i];
        e_l(e) = m_mesh->edge_length(e);
    }

    // ==============
    // Parallel section for faces
    // ==============
#pragma omp parallel for
    for (int i = 0; i < (int)faces.size(); i++) {
        Face* f = faces[i];

        // The following is almost the same as your original code
        // but we do it inside a parallel region.  You need to be sure
        // c_s(he) is safe to write in parallel.
        Point p[3];
        double l[3];

        // First, compute side lengths l[0..2]
        HalfEdge* he = f->halfedge();
        for (int j = 0; j < 3; j++) {
            Edge* e = he->edge();
            l[j] = e_l(e);
            he = he->he_next();
        }

        // Next, compute the angle
        double a = acos((l[0]*l[0] + l[2]*l[2] - l[1]*l[1]) / (2*l[0]*l[2]));

        // Set up triangle corners
        p[0] = Point(0, 0, 0);
        p[1] = Point(l[0], 0, 0);
        p[2] = Point(l[2]*cos(a), l[2]*sin(a), 0);

        // Normal and area
        Point n = (p[1]-p[0]) ^ (p[2]-p[0]);
        double area = n.norm() / 2.0;
        n /= area;  // "normalized" with respect to area

        // Traverse the halfedges again
        he = f->halfedge();
        for (int j = 0; j < 3; j++) {
            // The cross-product used in your original code
            Point s = (n ^ (p[(j + 1) % 3] - p[j])) / sqrt(area);

            // Write out to c_s(he)
            c_s(he) = s;

            he = he->he_next();
        }
    }
}

int LSCM::project( Eigen::MatrixXd & V_UV ) {
	auto start = std::chrono::high_resolution_clock::now();
	set_coefficients();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
    #ifdef VERBOSE
        std::cout << "set_coefficients() took " << elapsed.count() << " seconds." << std::endl;
    #endif

	std::vector<Vertex*> vertices;
    std::vector<Face*> faces;

    int i = 0;
	for (MeshVertexIterator viter(m_mesh); !viter.end(); ++viter){
		Vertex * v = *viter;
        i++;
		if (v->string().substr(0,4) != "fixt") {
			vertices.push_back(v);
		}
		else {
			m_fix_vertices.push_back(v);
		}

	}

    if(m_fix_vertices.size() < 2){
        std::cerr << "Error: Less than 2 fixed vertices. Aborting." << std::endl;
        return -1;
    }
	assert(m_fix_vertices.size()>=2);

	for (int k = 0; k < (int)vertices.size(); k++ ){
		v_idx(vertices[k]) = k;
	}
	for (int k = 0; k < (int)m_fix_vertices.size(); k++) {
		v_idx(m_fix_vertices[k]) = k;
		Vertex *v = m_fix_vertices[k];
		std::string tmp;
		double uv0, uv1;
		std::stringstream(v->string()) >> tmp >> uv0 >> uv1;
		v_uv(v) = Point2(uv0, uv1);
	}

	int fn = m_mesh->numFaces();
	int	vfn = m_fix_vertices.size();
	int	vn = m_mesh->numVertices() - vfn;
	
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList1;
	std::vector<T> tripletList2;
	tripletList1.reserve(fn);
	tripletList2.reserve(fn);
	VectorXd b(vfn * 2);

	int fid = 0;
	auto start_coeff = std::chrono::high_resolution_clock::now();
	for (MeshFaceIterator fiter(m_mesh); !fiter.end(); ++fiter, ++fid) {
		Face *f = *fiter;
		HalfEdge *he = f->halfedge();

		for (int j = 0; j < 3; j++) {
			Point s = c_s(he);

			Vertex * v = he->he_next()->target();
			int vid = v_idx(v);

			if (v->string().substr(0,4) != "fixt") {
				tripletList1.push_back(T(fid,vid,s[0]));
				tripletList1.push_back(T(fn + fid, vn + vid, s[0]));
				tripletList1.push_back(T(fid, vn + vid, -s[1]));
				tripletList1.push_back(T(fn + fid, vid, s[1]));
			}
			else {
				Point2 uv = v_uv(v);
                // std::cout << "uv: " << uv[0] << ", " << uv[1] << std::endl;
				tripletList2.push_back(T(fid, vid, s[0]));
				tripletList2.push_back(T(fn + fid, vfn + vid, s[0]));
				tripletList2.push_back(T(fid, vfn + vid, -s[1]));
				tripletList2.push_back(T(fn + fid, vid, s[1]));

				b[vid] = uv[0];
				b[vfn + vid] = uv[1];
			}
			he = he->he_next();
		}
	}
    auto end_coeff = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_coeff = end_coeff - start_coeff;
    #ifdef VERBOSE
    std::cout << "Coefficient setup took " << elapsed_coeff.count() << " seconds." << std::endl;
    #endif

    auto start_matrix = std::chrono::high_resolution_clock::now();
    SpMat A(2*fn, 2*vn);
    SpMat B(2*fn, 2*vfn);

    A.setFromTriplets(tripletList1.begin(), tripletList1.end());
    B.setFromTriplets(tripletList2.begin(), tripletList2.end());
    #ifdef VERBOSE
    auto end_matrix = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_matrix = end_matrix - start_matrix;
    std::cout << "Matrix setup took " << elapsed_matrix.count() << " seconds." << std::endl;
    #endif

    VectorXd r, x;
    r = B * b;
    r = r * -1;

    auto start_ata = std::chrono::high_resolution_clock::now();
    
    // Convert input matrix to row-major if it isn't already
    SparseMatrixRM A_row(A);
    
    // Pre-allocate the result matrix with estimated non-zeros
    // This avoids reallocation during multiplication
    Eigen::SparseMatrix<double> AtA(A.cols(), A.cols());
    AtA.reserve(Eigen::VectorXi::Constant(A.cols(), 20));  // Estimate 20 non-zeros per column
    
    // Use optimized sparse multiplication
    Eigen::SparseMatrix<double> A_T = A_row.transpose();
    AtA = A_T * A_row;


    #ifdef False
    if (!AtA.isCompressed()) {
        AtA.makeCompressed();
        std::cout << "Matrix was not compressed. Compressed it.\n";
    } 

    if (!isSymmetric(AtA)) {
        std::cerr << "Error: The matrix is not symmetric. Aborting.\n";
        return -1;
    }

    if (!isPositiveDefinite(AtA)) {

        // bool A_PD  = isPositiveDefinite(A);
        // std::cout << "A is positive definite: " << A_PD << std::endl;

        // bool AtA_PD = isPositiveDefinite(AtA);
        // std::cout << "AtA is positive definite: " << AtA_PD << std::endl;

        
        Eigen::MatrixXd A_dense(A);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_dense.transpose() * A_dense);
        if (es.info() == Eigen::Success) {
            std::cout << "Eigenvalues of A^T * A: " 
                      << es.eigenvalues().transpose() << std::endl;
        } else {
            std::cerr << "Failed to compute eigenvalues of A.\n";
        }



        // std::cout << "A:" << A << std::endl;
        // std::cout <<"ATA:" << AtA << std::endl;
        
        std::cerr << "Error: The matrix is not positive definite. Aborting.\n";
        // return;
    }
    #endif

    #ifdef VERBOSE
    auto end_ata = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ata = end_ata - start_ata;
    std::cout << "Forming AtA took " << elapsed_ata.count() << " seconds." << std::endl;
    #endif

    auto start_atr = std::chrono::high_resolution_clock::now();
    // 2) Form ATr
    Eigen::VectorXd ATr;
    ATr.noalias() = A_T * r;
    #ifdef VERBOSE
    auto end_atr = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_atr = end_atr - start_atr;
    std::cout << "Forming ATr took " << elapsed_atr.count() << " seconds." << std::endl;
    #endif

    auto start_factor = std::chrono::high_resolution_clock::now();
    // 3) Factor AtA.  We'll use PardisoLLT here, but you could also use
    //    PardisoLDLT or Eigen::SimplicialLDLT, etc., depending on your matrix.
    // Eigen::PardisoLLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> solver;


    solver.compute(AtA);
    // std::cout << "AtA factorized" << std::endl;
    if (solver.info() != Eigen::Success) {
        #ifdef VERBOSE

        std::cerr << "Error in factorization of AtA in LSCM compute" << solver.info() << std::endl;
        if (solver.info() == Eigen::NumericalIssue) {
            std::cerr << "Numerical issue in LSCM." << std::endl;
        }else if (solver.info() == Eigen::InvalidInput) {
            std::cerr << "Invalid input" << std::endl;
        }else if (solver.info() == Eigen::NoConvergence) {
            std::cerr << "No convergence" << std::endl;
        }else if (solver.info() == Eigen::InvalidInput) {
            std::cerr << "Invalid input" << std::endl;
        }else {
            std::cerr << "Unknown error" << std::endl;
        }

        std::cerr << "Matrix A shape: " << A.rows() << " x " << A.cols() << std::endl;
        std::cerr << "Matrix AtA shape: " << AtA.rows() << " x " << AtA.cols() << std::endl;
        std::cerr << "Vector r size: " << r.size() << std::endl;
        std::cerr << "Vector ATr size: " << ATr.size() << std::endl;
        #endif
        
        return 1;
    }
    #ifdef VERBOSE
        std::cout << "AtA factorized" << std::endl;
    #endif
    // 4) Solve for x in AtA x = ATr
    x = solver.solve(ATr);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Error in solving for x" << std::endl;
        std::cout << solver.info() << std::endl;
        return 1;
    }
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_factor = end_factor - start_factor;
    #ifdef VERBOSE
        std::cout << "Factorizing AtA and solving for x took " << elapsed_factor.count() << " seconds." << std::endl;
    #endif


	// for (int i = 0; i < std::min(20, (int)x.size()); ++i) {
	// 	std::cout << "x[" << i << "] = " << x[i] << std::endl;
	// }
	auto start_uv = std::chrono::high_resolution_clock::now();
	// for (int i = 0; i < vn; i++) {
	// 	Vertex * v = vertices[i];
	// 	v_uv(v) = Point2(x[i], x[i + vn]);
	// }
	// for (MeshVertexIterator viter(m_mesh); !viter.end(); viter++) {
	// 	Vertex * v = *viter;
	// 	Point2 p = v_uv(v);
	// 	Point p3(p[0], p[1], 0);
	// 	v->point() = p3;
	// }

    V_UV.resize(m_mesh->numVertices(), 2);

    int iV = 0; 
    for (MeshVertexIterator viter(m_mesh); !viter.end(); ++viter, ++iV)
    {
        Vertex * v = *viter;

        // If this vertex is not one of the "fixed" ones, it's in 'x':
        if (v->string().substr(0,4) != "fixt") 
        {
            // Use v_idx(v) to find its index in the solution vector x,
            // then copy x-coord and y-coord out of x (x stores all free verts).
            int idx = v_idx(v);
            V_UV(iV, 0) = x[idx];
            V_UV(iV, 1) = x[idx + vn];
        }
        else 
        {
            // A fixed vertex.  We already know its UV is stored in v_uv(v)
            // (read from the vertex string earlier).
            Point2 uv = v_uv(v);
            V_UV(iV, 0) = uv[0];
            V_UV(iV, 1) = uv[1];
        }
    }
    if (V_UV.rows() == 0 && V_UV.cols() == 0){
        std::cout << "Matrix is 0x0 (empty)." << std::endl;
        return 1;
    }

	auto end_uv = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_uv = end_uv - start_uv;
    #ifdef VERBOSE
        std::cout << "Assigning UV coordinates took " << elapsed_uv.count() << " seconds." << std::endl;
    #endif
    return 0;
}


