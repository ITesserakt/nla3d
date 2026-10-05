#pragma once

#include "SparseMatrix.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "math/Mat.h"
#include "math/Vec.h"
#include "sys.h"

namespace nla3d {
namespace math {
namespace internal {

template <typename T> struct traits {};

template <uint16 M, uint16 N> struct traits<Mat<M, N>> {
    using EigenEquivalent = Eigen::Matrix<double, M, N, Eigen::RowMajor>;
    using MappedEigenEquivalent = Eigen::Map<EigenEquivalent>;

    static MappedEigenEquivalent unsafeWrap(Mat<M, N>& object) {
        auto array = object.ptr();
        return MappedEigenEquivalent{array};
    }
};

template <uint16 M> struct traits<Vec<M>> {
    using EigenEquivalent = Eigen::Matrix<double, M, 1, Eigen::ColMajor>;
    using MappedEigenEquivalent = Eigen::Map<EigenEquivalent>;

    static MappedEigenEquivalent unsafeWrap(Vec<M>& object) {
        auto array = object.ptr();
        return MappedEigenEquivalent{array};
    }
};

template <> struct traits<SparseMatrix> {
    using EigenEquivalent = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using MappedEigenEquivalent = Eigen::Map<EigenEquivalent>;

    static MappedEigenEquivalent unsafeWrap(SparseMatrix& object) {
        auto* iofeirArray = reinterpret_cast<int32*>(object.getIofeirArray());
        auto* colsArray = reinterpret_cast<int32*>(object.getColumnsArray());

        return Eigen::Map<EigenEquivalent>{object.nRows(), object.nColumns(), object.nValues(),
                                           iofeirArray,    colsArray,         object.getValuesArray()};
    }
};

template <> struct traits<SparseSymMatrix> {
    using EigenEquivalent = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using MappedEigenEquivalent = Eigen::SparseSelfAdjointView<Eigen::Map<EigenEquivalent>, Eigen::Upper>;

    static MappedEigenEquivalent unsafeWrap(SparseSymMatrix& object) {
        auto* iofeirArray = reinterpret_cast<int32*>(object.getIofeirArray());
        auto* colsArray = reinterpret_cast<int32*>(object.getColumnsArray());
        auto&& mapping = Eigen::Map<EigenEquivalent>{object.nRows(), object.nColumns(), object.nValues(),
                                                     iofeirArray,    colsArray,         object.getValuesArray()};
        return Eigen::SparseSelfAdjointView<Eigen::Map<EigenEquivalent>, Eigen::Upper>{mapping};
    }
};
} // namespace internal

using internal::traits;

/// Returns a mutable view of an original matrix with Eigen support
template <uint16 dimM, uint16 dimN>
typename traits<Mat<dimM, dimN>>::MappedEigenEquivalent view_mat(Mat<dimM, dimN>& matrix) {
    static_assert(sizeof(Mat<dimM, dimN>) == sizeof(double[dimM * dimN]), "Matrix is not continuous");
    return traits<Mat<dimM, dimN>>::unsafeWrap(matrix);
}

/// Returns a mutable view of an original vector with Eigen support
template <uint16 dimM> typename traits<Vec<dimM>>::MappedEigenEquivalent view_vec(Vec<dimM>& vec) {
    static_assert(sizeof(Vec<dimM>) == sizeof(double[dimM]), "Vec is not continuous");
    return traits<Vec<dimM>>::unsafeWrap(vec);
}

template <typename T, typename F> void useSparseMat(T& mat, F callback) {
    const auto rows = mat.nRows();
    const auto cols = mat.nColumns();
    const auto nonZeros = mat.nValues();

    auto* colsArray = reinterpret_cast<int32*>(mat.getColumnsArray());
    for (uint32 i = 0; i < nonZeros; i++)
        colsArray[i] -= 1;
    auto* iofeirArray = reinterpret_cast<int32*>(mat.getIofeirArray());
    for (uint32 i = 0; i < rows + 1; i++)
        iofeirArray[i] -= 1;

    auto&& wrapper = traits<T>::unsafeWrap(mat);
    callback(wrapper);

    for (uint32 i = 0; i < nonZeros; i++)
        colsArray[i] += 1;
    for (uint32 i = 0; i < rows + 1; i++)
        iofeirArray[i] += 1;
}

} // namespace math
} // namespace nla3d
