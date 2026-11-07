mod accessor;
mod mat;
mod mat_sym;

pub use accessor::*;
pub use mat_sym::*;

use crate::dense::mat::ArrayStorage;
use generic_array::typenum::Const;

pub type Vec<const N: usize, T = f64> = Mat<N, 1, T>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mat<const N: usize, const M: usize, T = f64>(ArrayStorage<T, N, M>);
pub struct MatSym<const N: usize, T = f64, Kind = Lower>(SymArrayStorage<T, Const<N>, Kind>)
where
    Const<N>: MatSymLength;

pub type DMat<T = f64, Rows = usize, Cols = usize> = faer::Mat<T, Rows, Cols>;

#[derive(Debug, Clone, Copy)]
pub struct SingularMatrix;

impl std::fmt::Display for SingularMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Matrix is singular")
    }
}

impl std::error::Error for SingularMatrix {}

pub trait MatMul<Rhs> {
    type Output;

    fn matmul(self, rhs: Rhs) -> Self::Output;
}

pub trait MatMulAssign<Rhs> {
    type Output;

    fn matmul_assign(self, rhs: Rhs, out: Self::Output);
}

#[cfg(test)]
mod tests {
    use crate::dense::{Lower, Mat, MatSym, SymKind, Upper};
    use crate::num_traits::Zero;

    type M = MatSym<3>;

    #[test]
    fn test_sym_mat_size() {
        assert_eq!(size_of::<M>(), 6 * size_of::<f64>());
    }

    #[test]
    fn test_sym_mat_new() {
        let mut mat = MatSym::<3>::zero();
        mat[(1, 0)] = 3.0;

        assert_eq!(mat[(1, 0)], 3.0);
        assert_eq!(mat[(0, 1)], 3.0);
    }

    #[test]
    fn test_sym_mat_eq() {
        let mut mat = MatSym::<3>::zero();
        mat[(1, 0)] = 3.0;

        let mut mat2 = Mat::<3, 3>::zero();
        mat2[(1, 0)] = 3.0;
        mat2[(0, 1)] = 3.0;

        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_upper_indexing() {
        let n = 5;
        for i in 0..n {
            for j in 0..n {
                let k = Upper::flat_index(n, i, j);
                let (i_, j_) = Upper::wide_index(n, k);

                println!("{k}, {i_}, {j_}");
                assert!(i == i_ && j == j_ || i == j_ && j == i_);
            }
        }
    }

    #[test]
    fn test_lower_indexing() {
        let n = 5;
        for i in 0..n {
            for j in 0..n {
                let k = Lower::flat_index(n, i, j);
                let (i_, j_) = Lower::wide_index(n, k);

                assert!(i == i_ && j == j_ || i == j_ && j == i_);
            }
        }
    }
}
