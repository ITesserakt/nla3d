use std::ops::Index;

use faer::traits::ComplexField;
use num_traits::One;

use crate::zip::ext::Fill;
use crate::zip::{KnownShape, Producer, Zip};

pub trait MatMul<Rhs>: Sized {
    type Scalar;
    type Output;
    type Wrapper<T>;

    #[inline]
    fn matmul(self, rhs: Rhs) -> Self::Wrapper<Self::Output>
    where
        Self::Scalar: One,
    {
        self.matmul_scalar(rhs, One::one())
    }

    fn matmul_scalar(self, rhs: Rhs, scalar: Self::Scalar) -> Self::Wrapper<Self::Output>;
}

pub trait MatMulAssign<Lhs, Rhs>: Sized
where
    Lhs: MatMul<Rhs, Output = Self>,
{
    fn matmul_assign_scalar(&mut self, lhs: Lhs, rhs: Rhs, scalar: Lhs::Scalar)
    -> Lhs::Wrapper<()>;

    #[inline]
    fn matmul_assign(&mut self, lhs: Lhs, rhs: Rhs) -> Lhs::Wrapper<()>
    where
        Lhs::Scalar: One,
    {
        self.matmul_assign_scalar(lhs, rhs, Lhs::Scalar::one())
    }
}

#[inline]
pub fn manual_matmul_inplace<'a, T, P>(
    inner_size: usize,
    dst: P,
    scalar: T,
    lhs: &impl Index<(usize, usize), Output = T>,
    rhs: &impl Index<(usize, usize), Output = T>,
) where
    T: ComplexField + 'a,
    P: Producer<Item = &'a mut T, Shape: KnownShape>,
{
    Zip::from(dst)
        .indexed()
        .and(Fill(scalar))
        .for_each(|y, (i, j), scalar| {
            for k in 0..inner_size {
                *y += scalar.clone() * lhs[(i, k)].clone() * rhs[(k, j)].clone();
            }
        });
}

mod implementation_traits {
    use std::ops::Index;

    use faer::mat::{AsMatMut, AsMatRef};
    use faer::sparse::{FaerError, SparseColMatRef};
    use faer::traits::ComplexField;
    use faer::{Accum, Par};
    use generic_array::typenum::Const;
    use num_traits::Zero;

    use crate::dense::{DMat, Mat};
    use crate::matmul::{MatMul, MatMulAssign, manual_matmul_inplace};
    use crate::sparse;
    use crate::sparse::SparsityInfo;

    pub struct Dense;
    pub struct Sparse;

    struct IndexWrapper<T>(T);

    pub trait Operand {
        type Kind;
    }
    impl<T: Operand + ?Sized> Operand for &T {
        type Kind = T::Kind;
    }

    pub trait DenseOperand: Operand<Kind = Dense> {
        type Scalar;
        type View<'a>: AsMatRef<T = Self::Scalar, Rows = usize, Cols = usize>
        where
            Self: 'a;
        type Rows: Materialize;
        type Cols: Materialize;

        fn dense(&self) -> Self::View<'_>;
        fn rows(&self) -> Self::Rows;
        fn cols(&self) -> Self::Cols;
        fn index(&self, i: usize, j: usize) -> &Self::Scalar;
    }

    pub trait SparseOperand: Operand<Kind = Sparse> {
        type Scalar;

        fn sparse(&self) -> SparseColMatRef<'_, usize, Self::Scalar>;
        fn rows(&self) -> usize;
        fn cols(&self) -> usize;
    }

    impl<T> DenseOperand for &T
    where
        T: DenseOperand,
    {
        type Scalar = T::Scalar;
        type View<'a>
            = T::View<'a>
        where
            Self: 'a;
        type Rows = T::Rows;
        type Cols = T::Cols;

        #[inline]
        fn dense(&self) -> T::View<'_> {
            T::dense(self)
        }

        #[inline]
        fn rows(&self) -> Self::Rows {
            T::rows(self)
        }

        #[inline]
        fn cols(&self) -> Self::Cols {
            T::cols(self)
        }

        #[inline]
        fn index(&self, i: usize, j: usize) -> &Self::Scalar {
            T::index(self, i, j)
        }
    }

    impl<T: DenseOperand> Index<(usize, usize)> for IndexWrapper<T> {
        type Output = T::Scalar;

        #[inline]
        fn index(&self, index: (usize, usize)) -> &Self::Output {
            DenseOperand::index(&self.0, index.0, index.1)
        }
    }

    pub trait Product<L, R> {
        type Scalar;
        type Out;
        type Wrapper<T>;

        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Self::Wrapper<Self::Out>;
        fn product_into(
            out: &mut Self::Out,
            lhs: &L,
            rhs: &R,
            scalar: Self::Scalar,
            accum: Accum,
            par: Par,
        ) -> Self::Wrapper<()>;
    }

    #[diagnostic::on_unimplemented(
        message = "inner dimensions do not match: `{Self}` x `{Other}`",
        note = "static dimensions must be equal at compile time"
    )]
    pub trait Conforms<Other> {}

    pub trait Materialize {
        const VALUE: Option<usize>;

        fn materialize(self) -> usize;
    }

    pub trait HaveOutput<Rhs> {
        type Matrix<T>: AsMatMut<T = T, Rows = usize, Cols = usize>;

        fn zeroed<T: Zero + Clone>(rows: Self, cols: Rhs) -> Self::Matrix<T>;
    }

    impl<L, R, S> Product<L, R> for (Dense, Dense)
    where
        L: DenseOperand<Scalar = S>,
        R: DenseOperand<Scalar = S>,
        L::Cols: Conforms<R::Rows>,
        L::Rows: HaveOutput<R::Cols>,
        S: ComplexField + Zero,
    {
        type Scalar = S;
        type Out = <L::Rows as HaveOutput<R::Cols>>::Matrix<S>;
        type Wrapper<T> = T;

        #[inline]
        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Self::Out {
            assert_eq!(lhs.cols().materialize(), rhs.rows().materialize());
            let mut out = L::Rows::zeroed(lhs.rows(), rhs.cols());
            Self::product_into(&mut out, lhs, rhs, scalar, Accum::Replace, Par::Seq);
            out
        }

        #[inline]
        fn product_into(
            out: &mut Self::Out,
            lhs: &L,
            rhs: &R,
            scalar: Self::Scalar,
            accum: Accum,
            par: Par,
        ) {
            const SMALL_LIMIT: usize = 128;
            const fn manual_override<N: Materialize, K: Materialize, M: Materialize>() -> bool {
                match (N::VALUE, K::VALUE, M::VALUE) {
                    (Some(n), Some(k), Some(m)) => n * k * m <= SMALL_LIMIT,
                    _ => false,
                }
            }

            if const { manual_override::<L::Rows, L::Cols, R::Cols>() } {
                manual_matmul_inplace(lhs.cols().materialize(), out.as_mat_mut(), scalar, &IndexWrapper(lhs), &IndexWrapper(rhs));
                return;
            }

            faer::linalg::matmul::matmul(out, accum, lhs.dense(), rhs.dense(), scalar, par);
        }
    }

    impl<L, R, S> Product<L, R> for (Sparse, Sparse)
    where
        L: SparseOperand<Scalar = S>,
        R: SparseOperand<Scalar = S>,
        S: ComplexField,
    {
        type Scalar = S;
        type Out = sparse::Mat<S>;
        type Wrapper<T> = Result<T, FaerError>;

        #[inline]
        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Result<Self::Out, FaerError> {
            assert_eq!(lhs.cols(), rhs.rows());
            let output = faer::sparse::linalg::matmul::sparse_sparse_matmul(
                lhs.sparse(),
                rhs.sparse(),
                scalar,
                Par::Seq,
            )?;

            let (sparsity, values) = output.into_parts();
            let info = SparsityInfo::from_descriptor(sparsity);
            Ok(sparse::Mat::from_parts(info, values.into_boxed_slice()))
        }

        #[inline]
        fn product_into(
            out: &mut Self::Out,
            lhs: &L,
            rhs: &R,
            scalar: Self::Scalar,
            _: Accum,
            _: Par,
        ) -> Result<(), FaerError> {
            *out = Self::product(lhs, rhs, scalar)?;
            Ok(())
        }
    }

    impl<L, R, S> Product<L, R> for (Dense, Sparse)
    where
        L: DenseOperand<Scalar = S>,
        R: SparseOperand<Scalar = S>,
        S: ComplexField,
    {
        type Scalar = S;
        type Out = DMat<S>;
        type Wrapper<T> = T;

        #[inline]
        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Self::Out {
            assert_eq!(lhs.cols().materialize(), rhs.rows());
            let mut out = DMat::zeros(lhs.rows().materialize(), rhs.cols());
            Self::product_into(&mut out, lhs, rhs, scalar, Accum::Replace, Par::Seq);
            out
        }

        #[inline]
        fn product_into(
            out: &mut Self::Out,
            lhs: &L,
            rhs: &R,
            scalar: Self::Scalar,
            accum: Accum,
            par: Par,
        ) {
            let lhs = lhs.dense();
            faer::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut(),
                accum,
                lhs.as_mat_ref(),
                rhs.sparse(),
                scalar,
                par,
            );
        }
    }

    impl<L, R, S> Product<L, R> for (Sparse, Dense)
    where
        L: SparseOperand<Scalar = S>,
        R: DenseOperand<Scalar = S>,
        S: ComplexField,
    {
        type Scalar = S;
        type Out = DMat<S>;
        type Wrapper<T> = T;

        #[inline]
        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Self::Out {
            assert_eq!(lhs.cols(), rhs.rows().materialize());
            let mut out = DMat::zeros(lhs.rows(), rhs.cols().materialize());
            Self::product_into(&mut out, lhs, rhs, scalar, Accum::Replace, Par::Seq);
            out
        }

        fn product_into(
            out: &mut Self::Out,
            lhs: &L,
            rhs: &R,
            scalar: Self::Scalar,
            accum: Accum,
            par: Par,
        ) {
            let rhs = rhs.dense();
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut(),
                accum,
                lhs.sparse(),
                rhs.as_mat_ref(),
                scalar,
                par,
            );
        }
    }

    type K<L, R> = (<L as Operand>::Kind, <R as Operand>::Kind);
    impl<L: Operand, R: Operand, S, O, W> MatMul<R> for L
    where
        K<L, R>: Product<L, R, Scalar = S, Out = O, Wrapper<O> = W>,
    {
        type Scalar = S;
        type Output = O;
        type Wrapper<T> = <K<L, R> as Product<L, R>>::Wrapper<T>;

        fn matmul_scalar(self, rhs: R, scalar: Self::Scalar) -> W {
            <K<L, R>>::product(&self, &rhs, scalar)
        }
    }

    impl<L: Operand, R: Operand, S, O, W> MatMulAssign<L, R> for O
    where
        K<L, R>: Product<L, R, Scalar = S, Out = O, Wrapper<()> = W>,
    {
        fn matmul_assign_scalar(&mut self, lhs: L, rhs: R, scalar: S) -> W {
            <K<L, R>>::product_into(self, &lhs, &rhs, scalar, Accum::Add, Par::Seq)
        }
    }

    impl<const N: usize> Materialize for Const<N> {
        const VALUE: Option<usize> = Some(N);

        #[inline]
        fn materialize(self) -> usize {
            N
        }
    }

    impl Materialize for usize {
        const VALUE: Option<usize> = None;

        #[inline]
        fn materialize(self) -> usize {
            self
        }
    }

    impl<const N: usize> Conforms<Const<N>> for Const<N> {}
    impl<const N: usize> Conforms<Const<N>> for usize {}
    impl<const N: usize> Conforms<usize> for Const<N> {}
    impl Conforms<usize> for usize {}

    impl<const N: usize, const M: usize> HaveOutput<Const<M>> for Const<N> {
        type Matrix<T> = Mat<N, M, T>;

        #[inline]
        fn zeroed<T: Zero + Clone>(_: Const<N>, _: Const<M>) -> Self::Matrix<T> {
            Mat::zero()
        }
    }

    impl<const N: usize> HaveOutput<Const<N>> for usize {
        type Matrix<T> = DMat<T>;

        #[inline]
        fn zeroed<T: Zero + Clone>(rows: usize, _: Const<N>) -> Self::Matrix<T> {
            DMat::full(rows, N, T::zero())
        }
    }

    impl<const N: usize> HaveOutput<usize> for Const<N> {
        type Matrix<T> = DMat<T>;

        #[inline]
        fn zeroed<T: Zero + Clone>(_: Self, cols: usize) -> Self::Matrix<T> {
            DMat::full(N, cols, T::zero())
        }
    }

    impl HaveOutput<usize> for usize {
        type Matrix<T> = DMat<T>;

        #[inline]
        fn zeroed<T: Zero + Clone>(rows: Self, cols: usize) -> Self::Matrix<T> {
            DMat::full(rows, cols, T::zero())
        }
    }
}

mod implementation {
    use faer::MatRef;
    use faer::prelude::SparseColMatRef;
    use generic_array::typenum::Const;

    use super::implementation_traits::*;
    use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
    use crate::sparse;

    impl<const N: usize, const M: usize, T> Operand for Mat<N, M, T> {
        type Kind = Dense;
    }

    impl<const N: usize, const M: usize, T> DenseOperand for Mat<N, M, T> {
        type Scalar = T;
        type View<'a>
            = MatRef<'a, T>
        where
            Self: 'a;
        type Rows = Const<N>;
        type Cols = Const<M>;

        #[inline]
        fn dense(&self) -> Self::View<'_> {
            self.view()
        }

        #[inline]
        fn rows(&self) -> Const<N> {
            Const
        }

        #[inline]
        fn cols(&self) -> Const<M> {
            Const
        }

        #[inline]
        fn index(&self, i: usize, j: usize) -> &Self::Scalar {
            std::ops::Index::index(self, (i, j))
        }
    }

    impl<T> Operand for DMat<T> {
        type Kind = Dense;
    }

    impl<T> DenseOperand for DMat<T> {
        type Scalar = T;
        type View<'a>
            = MatRef<'a, T>
        where
            Self: 'a;
        type Rows = usize;
        type Cols = usize;

        #[inline]
        fn dense(&self) -> Self::View<'_> {
            self.as_ref()
        }

        #[inline]
        fn rows(&self) -> Self::Rows {
            self.nrows()
        }

        #[inline]
        fn cols(&self) -> Self::Cols {
            self.ncols()
        }

        #[inline]
        fn index(&self, i: usize, j: usize) -> &Self::Scalar {
            std::ops::Index::index(self, (i, j))
        }
    }

    impl<const N: usize, T, K: SymKind> Operand for MatSym<N, T, K>
    where
        Const<N>: MatSymLength,
    {
        type Kind = Dense;
    }

    impl<const N: usize, T, K: SymKind> DenseOperand for MatSym<N, T, K>
    where
        Const<N>: MatSymLength,
        T: Clone,
    {
        type Scalar = T;
        type View<'a>
            = Mat<N, N, T>
        where
            Self: 'a;
        type Rows = Const<N>;
        type Cols = Const<N>;

        #[inline]
        fn dense(&self) -> Self::View<'_> {
            self.to_mat()
        }

        #[inline]
        fn rows(&self) -> Self::Rows {
            Const
        }

        #[inline]
        fn cols(&self) -> Self::Cols {
            Const
        }

        #[inline]
        fn index(&self, i: usize, j: usize) -> &Self::Scalar {
            std::ops::Index::index(self, (i, j))
        }
    }

    impl<T> Operand for sparse::Mat<T> {
        type Kind = Sparse;
    }

    impl<T> SparseOperand for sparse::Mat<T> {
        type Scalar = T;

        #[inline]
        fn sparse(&self) -> SparseColMatRef<'_, usize, Self::Scalar> {
            self.view()
        }

        #[inline]
        fn rows(&self) -> usize {
            self.nrows()
        }

        #[inline]
        fn cols(&self) -> usize {
            self.ncols()
        }
    }
}

#[cfg(test)]
const _: () = {
    let m1 = crate::dense::Mat::from_array([[0.0; 3]; 5]);
    let m2 = crate::dense::Mat::from_array([[0.0; 5]; 4]);
    let m3 = crate::dense::MatSym::<3>::from_array([0.0; 6]);

    const fn does_multiply<A, B>(a: A, b: B)
    where
        A: MatMul<B>,
    {
        _ = std::mem::ManuallyDrop::new((a, b));
    }

    does_multiply(m1, m2);
    // does_multiply(m2, m1); // 3 != 4
    does_multiply(&m1, m2);
    does_multiply(m1, &m2);
    does_multiply(&m1, &m2);
    does_multiply(m3, m1);
    // does_multiply(m1, m3); // 5 != 3
};
