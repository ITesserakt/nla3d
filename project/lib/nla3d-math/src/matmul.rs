use core::ops::Index;

use faer::traits::ComplexField;
use num_traits::One;

use crate::zip::{Domain, Free, Join, KnownShape, Meet, Producer, Zip};

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
                    let mut output: $out = Zip::from(Fill(T::zero_impl())).with_shape($shape).collect();
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

#[inline]
fn manual_matmul_inplace<'a, T, D, P>(
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
                *y += scalar.clone() * lhs[(i, j)].clone() * rhs[(k, j)].clone();
            }
        });
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

mod impls {
    use faer::traits::ComplexField;
    use faer::{Accum, Par};
    use generic_array::typenum::Const;

    use super::{MatMul, MatMulAssign, RemoveRef, manual_matmul_inplace};
    use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
    use crate::zip::{Dynamic, Fill, Static, Zip};

    impl_matmul!(<const N: usize, const M: usize, const L: usize; T: ComplexField> Mat<N, M, T>, Mat<M, L, T> => Mat<N, L, T> {
        |self, rhs, scalar, lhs|
        shape = Static;
        scalar = T;
        as_mut = Mat::view_mut;
        as_ref_lhs = Mat::view;
        as_ref_rhs = Mat::view;
        override[(N + M + L) as f64 / 3.0 <= 3.0] = manual_matmul_inplace(M, self, scalar, &lhs, &rhs);
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
        override[N <= 4] = manual_matmul_inplace(N, self, scalar, RemoveRef::forward(&lhs), RemoveRef::forward(&rhs));
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind> MatSym<N, T, K>, Mat<N, N, T> => Mat<N, N, T> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = MatSym::to_mat;
        dense_rhs = RemoveRef::forward;
        override[N <= 4] = manual_matmul_inplace(N, self, scalar, RemoveRef::forward(&lhs), RemoveRef::forward(&rhs));
    });

    impl_matmul!(<const N: usize; T: ComplexField, K: SymKind> Mat<N, N, T>, MatSym<N, T, K> => Mat<N, N, T> {
        |self, rhs, scalar, lhs|
        symmetric = true;
        scalar = T;
        dense_lhs = RemoveRef::forward;
        dense_rhs = MatSym::to_mat;
        override[N <= 4] = manual_matmul_inplace(N, self, scalar, RemoveRef::forward(&lhs), RemoveRef::forward(&rhs));
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
}

#[cfg(test)]
mod tests {
    use matrixcompare::prop_assert_matrix_eq;
    use proptest::{prop_assert_eq, proptest};

    use crate::matmul::MatMul;
    use crate::{FaerCompare, arbitrary::mat};
    use crate::dense::DMat;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            #[cfg(miri)]
            failure_persistence: None,
            #[cfg(miri)]
            cases: 32,
            ..Default::default()
        })]

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn test_matmul_static(a in mat::<2, 4, _>(-1000.0..1000.0), b in mat::<4, 3, _>(-1000.0..1000.0), s in -1000.0..1000.0) {
            let da = a.into_dyn();
            let db = b.into_dyn();

            let out_manual = a.matmul_scalar(b, s);
            let out_faer: DMat = s * da * db;
            prop_assert_eq!(out_manual.shape(), out_faer.shape());
            prop_assert_matrix_eq!(out_manual, FaerCompare(out_faer), comp=abs, tol=1e-7);
        }
    }
}
