use num_traits::One;

pub trait MatMul<Rhs>: Sized {
    type Scalar;
    type Output;

    #[inline]
    fn matmul(self, rhs: Rhs) -> Self::Output
    where
        Self::Scalar: One,
    {
        self.matmul_scalar(rhs, One::one())
    }

    fn matmul_scalar(self, rhs: Rhs, scalar: Self::Scalar) -> Self::Output;
}

pub trait MatMulAssign<Lhs, Rhs>: Sized
where
    Lhs: MatMul<Rhs, Output = Self>,
{
    fn matmul_assign_scalar(&mut self, lhs: Lhs, rhs: Rhs, scalar: Lhs::Scalar) {
        *self = lhs.matmul_scalar(rhs, scalar);
    }

    #[inline]
    fn matmul_assign(&mut self, lhs: Lhs, rhs: Rhs)
    where
        Lhs::Scalar: One,
    {
        self.matmul_assign_scalar(lhs, rhs, Lhs::Scalar::one());
    }
}

macro_rules! impl_matmul {
    (
        prim <$(const $const_generic:ident: $const_generic_ty:ty$(,)?)*; $($generic:ident$(: $generic_bound:path)?$(,)?)*>
        $lhs:ty, $rhs:ty => $out:ty
        $(where $($where_ty:ty: $where_bound:path$(,)?)*)?
        |$self:ident, $rhs_param:ident, $scalar_param:ident| {
            scalar = $scalar:ty;
            impl = $implementation:expr;
            $(impl assign = |$lhs_param:ident| $implementation_assign:expr;)?
        }
    ) => {
        impl<$(const $const_generic: $const_generic_ty,)* $($generic$(: $generic_bound)?, )*> MatMul<$rhs> for $lhs
        $(
            where $($where_ty: $where_bound, )*
        )?
        {
            type Scalar = $scalar;
            type Output = $out;

            #[inline]
            fn matmul_scalar($self, $rhs_param: $rhs, $scalar_param: Self::Scalar) -> $out {
                $implementation
            }
        }

        impl<$(const $const_generic: $const_generic_ty,)* $($generic $(: $generic_bound)?,)*> MatMulAssign<$lhs, $rhs> for $out
        $(
            where $($where_ty: $where_bound, )*
        )?
        {
            $(#[inline]
            fn matmul_assign_scalar(&mut $self, $lhs_param: $lhs, $rhs_param: $rhs, $scalar_param: $scalar) {
                $implementation_assign
            })?
        }
    };
    (
        prim ref mut <$(const $const_generic:ident: $const_generic_ty:ty$(,)?)*; $($generic:ident$(: $generic_bound:path)?$(,)?)*>
        $lhs:ty, $rhs:ty => $out:ty
        $(where $($where_ty:ty: $where_bound:path$(,)?)*)?
        |$self:ident, $rhs_param:ident, $scalar_param:ident| {
            scalar = $scalar:ty;
            impl = $implementation:expr;
            $(impl assign = |$lhs_param:ident| $implementation_assign:expr;)?
        }
    ) => {
        impl_matmul!(
            prim <$(const $const_generic: $const_generic_ty, )*; $($generic $(: $generic_bound)?, )*>
            $lhs, $rhs => $out
            $(where $($where_ty: $where_bound,)*)?
            |$self, $rhs_param, $scalar_param| {
                scalar = $scalar;
                impl = $implementation;
                $(impl assign = |$lhs_param| $implementation_assign;)?
            }
        );
        impl_matmul!(
            prim <$(const $const_generic: $const_generic_ty, )*; $($generic $(: $generic_bound)?, )*>
            &$lhs, $rhs => $out
            $(where $($where_ty: $where_bound,)*)?
            |$self, $rhs_param, $scalar_param| {
                scalar = $scalar;
                impl = $implementation;
                $(impl assign = |$lhs_param| $implementation_assign;)?
            }
        );
        impl_matmul!(
            prim <$(const $const_generic: $const_generic_ty, )*; $($generic $(: $generic_bound)?, )*>
            $lhs, &$rhs => $out
            $(where $($where_ty: $where_bound,)*)?
            |$self, $rhs_param, $scalar_param| {
                scalar = $scalar;
                impl = $implementation;
                $(impl assign = |$lhs_param| $implementation_assign;)?
            }
        );
        impl_matmul!(
            prim <$(const $const_generic: $const_generic_ty, )*; $($generic $(: $generic_bound)?, )*>
            &$lhs, &$rhs => $out
            $(where $($where_ty: $where_bound,)*)?
            |$self, $rhs_param, $scalar_param| {
                scalar = $scalar;
                impl = $implementation;
                $(impl assign = |$lhs_param| $implementation_assign;)?
            }
        );
    };
    (
        <$(const $const_generic:ident: $const_generic_ty:ty$(,)?)*; $($generic:ident$(: $generic_bound:path)?$(,)?)*>
        $lhs:ty, $rhs:ty => $out:ty
        {
            |$self:ident, $rhs_param:ident, $scalar_param:ident, $lhs_param:ident|
            shape = $shape:expr;
            scalar = $scalar:ty;
            as_mut = $as_mut:expr;
            as_ref_lhs = $as_ref_lhs:expr;
            as_ref_rhs = $as_ref_rhs:expr;
            $(override[$when:expr] = $override:expr;)?
        }
    ) => {
        impl_matmul!(prim ref mut <$(const $const_generic: $const_generic_ty, )*; $($generic $(: $generic_bound)?,)*> $lhs, $rhs => $out
            |$self, $rhs_param, $scalar_param| {
                scalar = $scalar;
                impl = {
                    let mut output: $out = Zip::from(Fill(T::zero_impl())).with_shape($shape).collect_into();
                    output.matmul_assign_scalar($self, $rhs_param, $scalar_param);
                    output
                };
                impl assign = |$lhs_param| {
                    $(if const { $when } { return $override; })?
                    faer::linalg::matmul::matmul(
                        $as_mut($self), Accum::Add,
                        $as_ref_lhs(&$lhs_param), $as_ref_rhs(&$rhs_param),
                        $scalar_param, Par::Seq
                    );
                };
            }
        );
    };
    (
        <$(const $const_generic:ident: $const_generic_ty:ty$(,)?)*; $($generic:ident$(: $generic_bound:path)?$(,)?)*>
        $lhs:ty, $rhs:ty => $out:ty
        {
            |$self:ident, $rhs_param:ident, $scalar_param:ident, $lhs_param:ident|
            symmetric = true;
            scalar = $scalar:ty;
            dense_lhs = $dense_lhs:expr;
            dense_rhs = $dense_rhs:expr;
            $(override[$when:expr] = $override:expr;)?
        }
    ) => {
        impl_matmul!(prim ref mut <$(const $const_generic: $const_generic_ty, )*; $($generic $(: $generic_bound)?,)*> $lhs, $rhs => $out
            where Const<N>: MatSymLength
            |$self, $rhs_param, $scalar_param| {
                scalar = $scalar;
                impl = {
                    let dense_lhs = $dense_lhs(&$self);
                    let dense_rhs = $dense_rhs(&$rhs_param);
                    dense_lhs.matmul_scalar(dense_rhs, $scalar_param)
                };
                impl assign = |$lhs_param| {
                    $(if const { $when } { return $override; })?
                    let dense_lhs = $dense_lhs(&$lhs_param);
                    let dense_rhs = $dense_rhs(&$rhs_param);
                    $self.matmul_assign_scalar(dense_lhs, dense_rhs, $scalar_param);
                };
            }
        );
    };
}

trait RemoveRef {
    type Output;

    fn forward(&self) -> &Self::Output;
}

impl<T: RemoveRef> RemoveRef for &T {
    type Output = T::Output;

    #[inline(always)]
    fn forward(&self) -> &Self::Output {
        T::forward(self)
    }
}

impl<const N: usize, const M: usize, T> RemoveRef for crate::dense::Mat<N, M, T> {
    type Output = Self;

    #[inline(always)]
    fn forward(&self) -> &Self::Output {
        self
    }
}

impl<T, R: faer::Shape, C: faer::Shape> RemoveRef for crate::dense::DMat<T, R, C> {
    type Output = Self;

    #[inline(always)]
    fn forward(&self) -> &Self::Output {
        self
    }
}

impl<const N: usize, T, K> RemoveRef for crate::dense::MatSym<N, T, K>
where
    generic_array::typenum::Const<N>: crate::dense::MatSymLength,
{
    type Output = Self;

    #[inline(always)]
    fn forward(&self) -> &Self::Output {
        self
    }
}

pub use impls::manual_matmul_inplace;

mod impls {
    use core::ops::Index;

    use faer::traits::ComplexField;
    use faer::{Accum, Par};
    use generic_array::typenum::Const;

    use super::{MatMul, MatMulAssign, RemoveRef};
    use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
    use crate::sparse;
    use crate::zip::{Domain, Dynamic, Fill, Free, Join, KnownShape, Meet, Producer, Static, Zip};

    #[inline]
    pub fn manual_matmul_inplace<'a, T, D, P>(
        inner_size: usize,
        dst: P,
        scalar: T,
        lhs: &impl Index<(usize, usize), Output = T>,
        rhs: &impl Index<(usize, usize), Output = T>,
    ) where
        T: ComplexField + 'a,
        P: Producer<Item = &'a mut T, Domain = D>,
        D: Domain + Default + Meet<D>,
        Free: Join<P::Shape, Output: KnownShape>,
    {
        Zip::from_fn(D::default(), |ix| (*ix.i, *ix.j))
            .and(dst)
            .for_each(|(i, j), y| {
                for k in 0..inner_size {
                    *y += scalar.clone() * lhs[(i, k)].clone() * rhs[(k, j)].clone();
                }
            });
    }

    impl_matmul!(<const N: usize, const M: usize, const L: usize; T: ComplexField> Mat<N, M, T>, Mat<M, L, T> => Mat<N, L, T> {
        |self, rhs, scalar, lhs|
        shape = Static;
        scalar = T;
        as_mut = Mat::view_mut;
        as_ref_lhs = Mat::view;
        as_ref_rhs = Mat::view;
        override[N * M * L <= 256] = manual_matmul_inplace(M, self, scalar, &lhs, &rhs);
    });

    impl_matmul!(<; T: ComplexField, N: faer::Shape, M: faer::Shape, K: faer::Shape> DMat<T, N, K>, DMat<T, K, M> => DMat<T, N, M> {
        |self, rhs, scalar, lhs|
        shape = Dynamic { rows: self.nrows().unbound(), cols: rhs.ncols().unbound() };
        scalar = T;
        as_mut = DMat::as_mut;
        as_ref_lhs = DMat::as_ref;
        as_ref_rhs = DMat::as_ref;
    });

    impl_matmul!(<const N: usize, const M: usize; T: ComplexField, R: faer::Shape> DMat<T, R, usize>, Mat<N, M, T> => DMat<T, R, usize> {
        |self, rhs, scalar, lhs|
        shape = Dynamic { rows: self.nrows().unbound(), cols: M };
        scalar = T;
        as_mut = DMat::as_mut;
        as_ref_lhs = DMat::as_ref;
        as_ref_rhs = Mat::view;
    });

    impl_matmul!(<const N: usize, const M: usize; T: ComplexField, R: faer::Shape> Mat<N, M, T>, DMat<T, usize, R> => DMat<T, usize, R> {
        |self, rhs, scalar, lhs|
        shape = Dynamic { rows: N, cols: rhs.ncols().unbound() };
        scalar = T;
        as_mut = DMat::as_mut;
        as_ref_lhs = Mat::view;
        as_ref_rhs = DMat::as_ref;
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind> MatSym<N, T, K>, MatSym<N, T, K> => Mat<N, N, T> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = MatSym::to_mat;
        dense_rhs = MatSym::to_mat;
        override[N <= 5] = manual_matmul_inplace(N, self, scalar, RemoveRef::forward(&lhs), RemoveRef::forward(&rhs));
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind> MatSym<N, T, K>, Mat<N, N, T> => Mat<N, N, T> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = MatSym::to_mat;
        dense_rhs = RemoveRef::forward;
        override[N <= 5] = manual_matmul_inplace(N, self, scalar, RemoveRef::forward(&lhs), RemoveRef::forward(&rhs));
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind> Mat<N, N, T>, MatSym<N, T, K> => Mat<N, N, T> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = RemoveRef::forward;
        dense_rhs = MatSym::to_mat;
        override[N <= 5] = manual_matmul_inplace(N, self, scalar, RemoveRef::forward(&lhs), RemoveRef::forward(&rhs));
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind, R: faer::Shape> DMat<T, R, usize>, MatSym<N, T, K> => DMat<T, R, usize> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = RemoveRef::forward;
        dense_rhs = MatSym::to_mat;
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind, R: faer::Shape> MatSym<N, T, K>, DMat<T, usize, R> => DMat<T, usize, R> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = MatSym::to_mat;
        dense_rhs = RemoveRef::forward;
    });

    impl<T: ComplexField, I: faer::Index> MatMul<DMat<T>> for sparse::Mat<T, I> {
        type Scalar = T;
        type Output = DMat<T>;

        fn matmul_scalar(self, rhs: DMat<T>, scalar: Self::Scalar) -> Self::Output {
            let mut output = Zip::from(Fill(T::zero_impl()))
                .with_shape(Dynamic {
                    rows: self.nrows(),
                    cols: rhs.ncols(),
                })
                .collect();
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                output.as_mut(),
                Accum::Replace,
                self.view(),
                rhs.as_ref(),
                scalar,
                Par::Seq,
            );
            output
        }
    }
}
