#include <Eigen/IterativeLinearSolvers>

#include "math/EquationSolver.h"
#include "math/Mat.h"
#include "math/SparseMatrix.h"
#include "math/Vec.h"
#include "math/eigen_support.h"
#include "sys.h"

using nla3d::math::BiCGSTAB_EquationSolver;
using nla3d::math::ConjugateGradientEquationSolver;
using nla3d::math::Mat;
using nla3d::math::SparseMatrix;
using nla3d::math::SparseSymMatrix;
using nla3d::math::Vec;
using Mat33 = Mat<3, 5>;
using Vec3 = Vec<3>;

static SparseSymMatrix dummySparseSymMatrix() {
    SparseSymMatrix matrix{3, 3};
    matrix.addEntry(1, 1);
    matrix.addEntry(3, 1);
    matrix.addEntry(2, 2);
    matrix.addEntry(1, 3);
    matrix.addEntry(3, 3);
    matrix.compress();
    matrix(1, 1) = 4;
    matrix(3, 1) = 1;
    matrix(2, 2) = 2;
    matrix(1, 3) = 1;
    matrix(3, 3) = 3;

    return matrix;
}

static SparseMatrix dummySparseMatrix() {
    SparseMatrix matrix = {3, 5};
    matrix.addEntry(1, 3);
    matrix.addEntry(2, 2);
    matrix.addEntry(3, 3);
    matrix.addEntry(2, 4);
    matrix.addEntry(1, 5);
    matrix.compress();
    matrix(1, 3) = 1;
    matrix(2, 2) = 2;
    matrix(3, 3) = 3;
    matrix(2, 4) = 4;
    matrix(1, 5) = 5;

    return matrix;
}

int main() {
    {
        LOG(INFO) << "Test `view_mat`";
        Mat33 matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        auto view = nla3d::math::view_mat(matrix);

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 5; j++) {
                CHECK_EQTH(matrix[i][j], view(i, j), 1e-7);
            }
        }
    }
    {
        LOG(INFO) << "Test `view_vec`";
        Vec3 vec = {1, 2, 3};
        auto view = nla3d::math::view_vec(vec);

        for (size_t i = 0; i < 3; i++) {
            CHECK_EQTH(vec[i], view(i), 1e-7);
        }
    }
    {
        LOG(INFO) << "Test `useSparseMat`";
        auto matrix = dummySparseMatrix();

        using View = nla3d::math::internal::traits<SparseMatrix>::MappedEigenEquivalent;
        nla3d::math::useSparseMat(matrix, [](const View& view) {
            CHECK_EQTH(view.coeff(0, 2), 1, 1e-7);
            CHECK_EQTH(view.coeff(1, 1), 2, 1e-7);
            CHECK_EQTH(view.coeff(2, 2), 3, 1e-7);
            CHECK_EQTH(view.coeff(1, 3), 4, 1e-7);
            CHECK_EQTH(view.coeff(0, 4), 5, 1e-7);
        });
    }
    {
        LOG(INFO) << "Test `useSparseMat` with direct product";
        auto matrix = dummySparseSymMatrix();

        using View = nla3d::math::internal::traits<SparseSymMatrix>::MappedEigenEquivalent;
        nla3d::math::useSparseMat(matrix, [](const View& view) {
            const Eigen::MatrixXd product = view * Eigen::Vector3d{2.0 / 11.0, 1.0 / 2.0, 3.0 / 11.0};
            CHECK(product.isApproxToConstant(1)) << "Actual value: " << product;
        });
    }
    {
        LOG(INFO) << "Test eigen sparse eq solver infrastructure";
        Eigen::SparseMatrix<double, Eigen::RowMajor> matrix{3, 3};
        matrix.coeffRef(0, 0) = 4;
        matrix.coeffRef(2, 0) = 1;
        matrix.coeffRef(1, 1) = 2;
        matrix.coeffRef(0, 2) = 1;
        matrix.coeffRef(2, 2) = 3;
        matrix.makeCompressed();

        double input[3] = {1, 1, 1};
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> factorization{matrix};
        const Eigen::Map<Eigen::Matrix<double, -1, 1>> rhs{input, 3};
        const auto solution = factorization.solve(rhs);

        CHECK(solution.isApprox(Eigen::Vector3d{2.0 / 11.0, 1.0 / 2.0, 3.0 / 11.0})) << "Actual value: " << solution;
    }
    {
        LOG(INFO) << "Test `ConjugateGradientEquationSolver`";
        auto matrix = dummySparseSymMatrix();

        double input[3] = {1, 1, 1}, output[3] = {0, 0, 0};
        ConjugateGradientEquationSolver solver;
        solver.solveEquations(&matrix, input, output);
        Eigen::Map<Eigen::Vector3d> output_mapping{output};

        CHECK(output_mapping.isApprox(Eigen::Vector3d{2.0 / 11.0, 1.0 / 2.0, 3.0 / 11.0}))
            << "Actual values: " << output_mapping;
    }
    {
        LOG(INFO) << "Test `BiCSTAB_EquationSolver`";
        auto matrix = dummySparseSymMatrix();

        double input[3] = {1, 1, 1}, output[3] = {0, 0, 0};
        BiCGSTAB_EquationSolver solver;
        solver.solveEquations(&matrix, input, output);
        Eigen::Map<Eigen::Vector3d> output_mapping{output};

        CHECK(output_mapping.isApprox(Eigen::Vector3d{2.0 / 11.0, 1.0 / 2.0, 3.0 / 11.0}))
            << "Actual values: " << output_mapping;
    }
}