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
    use generic_array::typenum::Const;
    use matrixcompare::assert_scalar_eq;
    use rand::SeedableRng;

    use crate::dense::{Lower, Mat, MatSym, MatSymLength, SymKind, Upper};
    use crate::num_traits::Zero;
    use crate::zip::RandomConstructible;

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

    fn test_determinants<const N: usize>(expected: f64)
    where
        Const<N>: MatSymLength,
    {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let m = MatSym::<_, f64, Lower>::random_uniform((N, N), &mut rng).to_mat();
        let mu = MatSym::<_, _, Upper>::from_mat(m);
        let ml = MatSym::<_, _, Lower>::from_mat(m);

        let det = m.determinant();
        assert_scalar_eq!(det, expected, comp = abs, tol = 1e-7);
        assert_scalar_eq!(det, mu.determinant(), comp = abs, tol = 1e-9);
        assert_scalar_eq!(det, ml.determinant(), comp = abs, tol = 1e-9);
    }

    #[test]
    fn test_determinant_0() {
        test_determinants::<0>(1.0);
    }

    #[test]
    fn test_determinant_1() {
        test_determinants::<1>(0.8143051451229099);
    }

    #[test]
    fn test_determinant_2() {
        test_determinants::<2>(-0.3971094091837122);
    }

    #[test]
    fn test_determinant_3() {
        test_determinants::<3>(-0.172172412694047);
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_determinant_4() {
        test_determinants::<4>(-0.28809505590157414);
    }
}
