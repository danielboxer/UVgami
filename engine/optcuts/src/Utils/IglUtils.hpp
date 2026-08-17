//  Created by Minchen Li on 8/30/17.

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include "TriMesh.hpp"

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>
#include <set>
#include <vector>

namespace uvgami {

// a static class implementing basic geometry processing operations that are not
// provided in libIgl
class IglUtils {
  public:
    // graph laplacian with half-weighted boundary edge, the computation is also
    // faster
    static void computeUniformLaplacian(const Eigen::MatrixXi &F,
                                        Eigen::SparseMatrix<double> &graphL);

    static void computeMVCMtr(const Eigen::MatrixXd &V,
                              const Eigen::MatrixXi &F,
                              Eigen::SparseMatrix<double> &MVCMtr);

    static void fixedBoundaryParam_MVC(Eigen::SparseMatrix<double> A,
                                       const Eigen::VectorXi &bnd,
                                       const Eigen::MatrixXd &bnd_uv,
                                       Eigen::MatrixXd &UV_Tutte);

    static void mapTriangleTo2D(const Eigen::Vector3d v[3],
                                Eigen::Vector2d u[3]);
    static void computeDeformationGradient(const Eigen::Vector3d v[3],
                                           const Eigen::Vector2d u[3],
                                           Eigen::Matrix2d &F);

    // to a circle with the perimeter equal to the length of the boundary on the
    // mesh
    static void map_vertices_to_circle(const Eigen::MatrixXd &V,
                                       const Eigen::VectorXi &bnd,
                                       Eigen::MatrixXd &UV);

    static void reportDistortion(const Eigen::VectorXd &scalar,
                                 double lowerBound, double upperBound);

    static void addBlockToMatrix(Eigen::SparseMatrix<double> &mtr,
                                 Eigen::Ref<const Eigen::MatrixXd> block,
                                 Eigen::Ref<const Eigen::VectorXi> index,
                                 int dim);
    // writes into presized V, I, J starting at tripletInd so callers can
    // fill disjoint slices in parallel
    static void addBlockToMatrix(Eigen::Ref<const Eigen::MatrixXd> block,
                                 Eigen::Ref<const Eigen::VectorXi> index, int dim,
                                 Eigen::VectorXd *V, Eigen::VectorXi *I,
                                 Eigen::VectorXi *J, int tripletInd);
    static void addDiagonalToMatrix(Eigen::Ref<const Eigen::VectorXd> diagonal,
                                    Eigen::Ref<const Eigen::VectorXi> index,
                                    int dim, Eigen::VectorXd *V,
                                    Eigen::VectorXi *I = NULL,
                                    Eigen::VectorXi *J = NULL);
    static void addBlockToMatrix(Eigen::Ref<const Eigen::MatrixXd> block,
                                 Eigen::Ref<const Eigen::VectorXi> index, int dim,
                                 Eigen::MatrixXd &mtr);
    static void addDiagonalToMatrix(Eigen::Ref<const Eigen::VectorXd> diagonal,
                                    Eigen::Ref<const Eigen::VectorXi> index,
                                    int dim, Eigen::MatrixXd &mtr);

    // nearest positive semidefinite matrix to a triangle's 6x6 uv hessian,
    // negative eigenvalues clamped to zero. the energy is translation
    // invariant, so the two uniform shifts are a null space and the eigen
    // solve happens in the 4-dimensional complement (Q spans it, so
    // H == Q (Q^T H Q) Q^T). the 6x6 solve was half the cpu of a run
    static void makePDTriangleHessian(Eigen::Matrix<double, 6, 6> &symMtr) {
        static const Eigen::Matrix<double, 6, 4> Q = [] {
            Eigen::Matrix<double, 6, 6> shiftsFirst =
                Eigen::Matrix<double, 6, 6>::Zero();
            shiftsFirst(0, 0) = shiftsFirst(2, 0) = shiftsFirst(4, 0) = 1.0;
            shiftsFirst(1, 1) = shiftsFirst(3, 1) = shiftsFirst(5, 1) = 1.0;
            shiftsFirst.block<6, 4>(0, 2) =
                Eigen::Matrix<double, 6, 4>::Identity();
            Eigen::HouseholderQR<Eigen::Matrix<double, 6, 6>> qr(shiftsFirst);
            const Eigen::Matrix<double, 6, 6> full = qr.householderQ();
            return Eigen::Matrix<double, 6, 4>(full.rightCols<4>());
        }();
        const Eigen::Matrix4d reduced = Q.transpose() * symMtr * Q;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigenSolver(reduced);
        if (eigenSolver.eigenvalues()[0] >= 0.0)
            return;
        Eigen::Vector4d clamped = eigenSolver.eigenvalues();
        for (int i = 0; i < 4 && clamped[i] < 0.0; ++i)
            clamped[i] = 0.0;
        const Eigen::Matrix<double, 6, 4> QV = Q * eigenSolver.eigenvectors();
        symMtr = QV * clamped.asDiagonal() * QV.transpose();
    }

    static double computeRotAngle(const Eigen::RowVector2d &from,
                                  const Eigen::RowVector2d &to);

    // test whether 2D segments ab intersect with cd
    static bool Test2DSegmentSegment(const Eigen::RowVector2d &a,
                                     const Eigen::RowVector2d &b,
                                     const Eigen::RowVector2d &c,
                                     const Eigen::RowVector2d &d,
                                     double eps = 0.0);

    // true if any two non-adjacent UV boundary edges cross, i.e. the input UV
    // islands self-intersect or overlap each other. spatial-hash broad phase,
    // Test2DSegmentSegment narrow phase. does not catch full containment.
    // with crossingVerts, every crossing is reported through it instead of
    // returning at the first one, so a caller can tell which charts are at
    // fault. transversalOnly skips touching and collinear overlap, so the
    // exactly coincident runs of a mid-zip stitch don't count as crossings
    static bool checkUVBoundaryOverlap(
        const Eigen::MatrixXd &UV,
        const std::vector<std::vector<int>> &bnd_all,
        std::set<int> *crossingVerts = nullptr, bool transversalOnly = false);

    static void smoothVertField(const TriMesh &mesh, Eigen::VectorXd &field);
};
} // namespace uvgami

#endif
