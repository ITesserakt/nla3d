use faer::Accum;
use faer::traits::ComplexField;
use num_traits::One;

use crate::matmul::implementation_traits::{DenseOperand, Materialize};
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
    accum: Accum,
    dst: P,
    scalar: T,
    lhs: &impl DenseOperand<Scalar = T>,
    rhs: &impl DenseOperand<Scalar = T>,
) where
    T: ComplexField + 'a,
    P: Producer<Item = &'a mut T, Shape: KnownShape>,
{
    log::trace!("Using manual matrix multiplication");
    let op = match accum {
        Accum::Replace => |y: &mut T, acc, scalar| *y = acc * scalar,
        Accum::Add => |y: &mut T, acc, scalar| *y += acc * scalar,
    };
    assert_eq!(lhs.cols().materialize(), inner_size);
    assert_eq!(rhs.rows().materialize(), inner_size);

    Zip::from(dst)
        .indexed()
        .and(Fill(scalar))
        .for_each(|y, (i, j), scalar| {
            let mut acc = T::zero_impl();
            for k in 0..inner_size {
                acc += unsafe { lhs.get_unchecked(i, k) * rhs.get_unchecked(k, j) };
            }
            op(y, acc, scalar);
        });
}

mod implementation_traits {
    use faer::mat::{AsMatMut, AsMatRef};
    use faer::sparse::linalg::matmul::{DenseSparseMatMul, SparseDenseMatMul};
    use faer::sparse::{FaerError, SparseColMatRef, SparseRowMatRef};
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

    pub trait Operand {
        type Kind;
    }

    #[diagnostic::do_not_recommend]
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
        unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Scalar;
    }

    pub trait SparseOperand: Operand<Kind = Sparse> {
        type Scalar;
        type View<'a>
        where
            Self: 'a;

        fn sparse(&self) -> Self::View<'_>;
        fn rows(&self) -> usize;
        fn cols(&self) -> usize;
    }

    #[diagnostic::do_not_recommend]
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
        unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Scalar {
            unsafe { T::get_unchecked(self, i, j) }
        }
    }

    #[diagnostic::do_not_recommend]
    impl<T> SparseOperand for &T
    where
        T: SparseOperand,
    {
        type Scalar = T::Scalar;
        type View<'a>
            = T::View<'a>
        where
            Self: 'a;

        fn sparse(&self) -> T::View<'_> {
            T::sparse(self)
        }

        fn rows(&self) -> usize {
            T::rows(self)
        }

        fn cols(&self) -> usize {
            T::cols(self)
        }
    }

    pub trait Product<L, R> {
        type Scalar;
        type Out;
        type Wrapper<T>;

        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Self::Wrapper<Self::Out>;
    }

    pub trait ProductInto<L, R>: Product<L, R> {
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
        message = "cannot multiply `{Lhs}` by `{Rhs}`: inner dimensions differ",
        label = "left operand has {Self} columns, right operand has {Other} rows"
    )]
    pub trait Conforms<Other, Lhs, Rhs> {}

    pub trait Materialize {
        const VALUE: Option<usize>;

        fn materialize(self) -> usize;
    }

    pub trait HaveOutput<Rhs> {
        type Matrix<T>: AsMatMut<T = T, Rows = usize, Cols = usize>;

        fn zeroed<T: Zero + Clone>(rows: Self, cols: Rhs) -> Self::Matrix<T>;
    }

    pub trait TransposeView {
        type Output;

        fn transpose(self) -> Self::Output;
    }

    trait Is<T = Self> {
        fn cast(self) -> T;
    }

    impl<T> Is for T {
        #[inline]
        fn cast(self) -> Self {
            self
        }
    }

    impl<L, R, S> Product<L, R> for (Dense, Dense)
    where
        L: DenseOperand<Scalar = S>,
        R: DenseOperand<Scalar = S>,
        L::Cols: Conforms<R::Rows, L, R>,
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
    }

    impl<L, R, S> ProductInto<L, R> for (Dense, Dense)
    where
        L: DenseOperand<Scalar = S>,
        R: DenseOperand<Scalar = S>,
        L::Cols: Conforms<R::Rows, L, R>,
        L::Rows: HaveOutput<R::Cols>,
        S: ComplexField + Zero,
    {
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
                    (Some(n), Some(k), Some(m)) => {
                        n.saturating_mul(k).saturating_mul(m) <= SMALL_LIMIT
                    }
                    _ => false,
                }
            }

            if const { manual_override::<L::Rows, L::Cols, R::Cols>() } {
                manual_matmul_inplace(
                    lhs.cols().materialize(),
                    accum,
                    out.as_mat_mut(),
                    scalar,
                    lhs,
                    rhs,
                );
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
        for<'a> L::View<'a>: Is<SparseColMatRef<'a, usize, S>>,
        for<'a> R::View<'a>: Is<SparseColMatRef<'a, usize, S>>,
    {
        type Scalar = S;
        type Out = sparse::Mat<S>;
        type Wrapper<T> = Result<T, FaerError>;

        #[inline]
        fn product(lhs: &L, rhs: &R, scalar: Self::Scalar) -> Result<Self::Out, FaerError> {
            assert_eq!(lhs.cols(), rhs.rows());
            let output = faer::sparse::linalg::matmul::sparse_sparse_matmul(
                lhs.sparse().cast(),
                rhs.sparse().cast(),
                scalar,
                Par::Seq,
            )?;

            let (sparsity, values) = output.into_parts();
            let info = SparsityInfo::from_descriptor(sparsity);
            Ok(sparse::Mat::from_parts(info, values.into_boxed_slice()))
        }
    }

    impl<L, R, S> Product<L, R> for (Dense, Sparse)
    where
        L: DenseOperand<Scalar = S>,
        R: SparseOperand<Scalar = S>,
        S: ComplexField,
        for<'a> R::View<'a>: DenseSparseMatMul<S>,
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
    }

    impl<L, R, S> ProductInto<L, R> for (Dense, Sparse)
    where
        L: DenseOperand<Scalar = S>,
        R: SparseOperand<Scalar = S>,
        S: ComplexField,
        for<'a> R::View<'a>: DenseSparseMatMul<S>,
    {
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
        for<'a> L::View<'a>: SparseDenseMatMul<S>,
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
    }

    impl<L, R, S> ProductInto<L, R> for (Sparse, Dense)
    where
        L: SparseOperand<Scalar = S>,
        R: DenseOperand<Scalar = S>,
        S: ComplexField,
        for<'a> L::View<'a>: SparseDenseMatMul<S>,
    {
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
        K<L, R>: ProductInto<L, R, Scalar = S, Out = O, Wrapper<()> = W>,
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

    impl<const N: usize, L, R> Conforms<Const<N>, L, R> for Const<N> {}
    impl<const N: usize, L, R> Conforms<Const<N>, L, R> for usize {}
    impl<const N: usize, L, R> Conforms<usize, L, R> for Const<N> {}
    impl<L, R> Conforms<usize, L, R> for usize {}

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

    impl<'a, I: faer::Index, T, R: faer::Shape, C: faer::Shape> TransposeView
        for SparseColMatRef<'a, I, T, R, C>
    {
        type Output = SparseRowMatRef<'a, I, T, C, R>;

        #[inline]
        fn transpose(self) -> Self::Output {
            SparseColMatRef::transpose(self)
        }
    }
}

mod implementation {
    use faer::MatRef;
    use faer::mat::AsMatRef;
    use faer::prelude::SparseColMatRef;
    use generic_array::typenum::Const;

    use super::implementation_traits::*;
    use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
    use crate::sparse;
    use crate::zip::ext::TransposeProducer;

    impl<const N: usize, const M: usize, T> Operand for Mat<N, M, T> {
        type Kind = Dense;
    }

    impl<const N: usize, const M: usize, T: Clone> DenseOperand for Mat<N, M, T> {
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
        unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Scalar {
            unsafe { self.get_unchecked(i, j).clone() }
        }
    }

    impl<T> Operand for DMat<T> {
        type Kind = Dense;
    }

    impl<T: Clone> DenseOperand for DMat<T> {
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
        unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Scalar {
            unsafe { self.get_unchecked(i, j).clone() }
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
        unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Scalar {
            unsafe { self.get_unchecked(i, j).clone() }
        }
    }

    impl<T> Operand for sparse::Mat<T> {
        type Kind = Sparse;
    }

    impl<T> SparseOperand for sparse::Mat<T> {
        type Scalar = T;
        type View<'a>
            = SparseColMatRef<'a, usize, T>
        where
            Self: 'a;

        #[inline]
        fn sparse(&self) -> Self::View<'_> {
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

    impl<T: Operand> Operand for TransposeProducer<T> {
        type Kind = T::Kind;
    }

    impl<T: Clone> DenseOperand for TransposeProducer<T>
    where
        T: DenseOperand,
        T: AsMatRef<T = T::Scalar, Rows = usize, Cols = usize>,
    {
        type Scalar = T::Scalar;
        type View<'a>
            = MatRef<'a, T::Scalar>
        where
            Self: 'a;
        type Rows = <T as DenseOperand>::Cols;
        type Cols = <T as DenseOperand>::Rows;

        #[inline]
        fn dense(&self) -> Self::View<'_> {
            self.view()
        }

        #[inline]
        fn rows(&self) -> Self::Rows {
            self.0.cols()
        }

        #[inline]
        fn cols(&self) -> Self::Cols {
            self.0.rows()
        }

        #[inline]
        unsafe fn get_unchecked(&self, i: usize, j: usize) -> Self::Scalar {
            unsafe { self.0.get_unchecked(j, i) }
        }
    }

    impl<T> SparseOperand for TransposeProducer<T>
    where
        T: SparseOperand,
        for<'a> T::View<'a>: TransposeView,
    {
        type Scalar = T::Scalar;
        type View<'a>
            = <T::View<'a> as TransposeView>::Output
        where
            Self: 'a;

        #[inline]
        fn sparse(&self) -> <T::View<'_> as TransposeView>::Output {
            self.0.sparse().transpose()
        }

        #[inline]
        fn rows(&self) -> usize {
            self.0.cols()
        }

        #[inline]
        fn cols(&self) -> usize {
            self.0.rows()
        }
    }
}

#[cfg(false)]
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
