mod accessor;
mod mat;
mod mat_sym;

pub use accessor::*;
use generic_array::typenum::Const;
pub use mat_sym::*;

use crate::dense::mat::ArrayStorage;

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

#[macro_export]
macro_rules! or_else {
    (, $else:tt) => {
        $else
    };
    ($or:tt, $else:tt) => {
        $or
    };
}

#[macro_export]
macro_rules! impl_operator {
    (
        <$($lt:lifetime$(,)?)*; $(const $const_generic:ident: $const_bound:ident$(,)?)*; $($generic:ident$(,)?)*>
        impl $trait_name:ident$(<$rhs:ty>)? for $name:ty
        $(where $($bind:ty: $bound:path$(,)?)*)?
        {
            fn = $trait_fn:ident;
            scalar = $scalar:ty$(, $scalar_rhs:ty)?;
            output = $output:ty;
            output_scalar = $output_scalar:ident;
            handle self = $handle_self:expr;
            handle rhs = $handle_rhs:expr;
        }
    ) => {
        impl<$($lt, )* $(const $const_generic: $const_bound, )* $($generic, )* $output_scalar> $trait_name$(<$rhs>)? for $name
        where
            $scalar: $trait_name<$($scalar_rhs, )? Output = $output_scalar>,
            $($($bind: $bound,)*)?
        {
            type Output = $output;

            #[inline]
            fn $trait_fn(self, rhs: $crate::or_else!($($rhs)?, Self)) -> Self::Output {
                Zip::from($handle_self(self))
                    .and($handle_rhs(rhs))
                    .map(|x, y| x.$trait_fn(y))
            }
        }
    };
    (
        <$($lt:lifetime$(,)?)*; $(const $const_generic:ident: $const_bound:ident$(,)?)*; $($generic:ident$(,)?)*>
        impl Ops$(<$rhs:ty>)? for $name:ty
        $(where $($bind:ty: $bound:path$(,)?)*)? {
            scalar = $scalar:ty$(, $scalar_rhs:ty)?;
            output = $output:ty;
            output scalar = $output_scalar:ident;
            handle self = $handle_self:expr;
            handle rhs = $handle_rhs:expr;
        }
    ) => {
        $crate::impl_operator!(
            <$($lt),*; $(const $const_generic: $const_bound),*; $($generic),*>
            impl Add$(<$rhs>)? for $name
            $(where $($bind: $bound,)*)?
            {
                fn = add;
                scalar = $scalar$(, $scalar_rhs)?;
                output = $output;
                output_scalar = $output_scalar;
                handle self = $handle_self;
                handle rhs = $handle_rhs;
            }
        );
        $crate::impl_operator!(
            <$($lt),*; $(const $const_generic: $const_bound),*; $($generic),*>
            impl Sub$(<$rhs>)? for $name
            $(where $($bind: $bound,)*)?
            {
                fn = sub;
                scalar = $scalar$(, $scalar_rhs)?;
                output = $output;
                output_scalar = $output_scalar;
                handle self = $handle_self;
                handle rhs = $handle_rhs;
            }
        );
        $crate::impl_operator!(
            <$($lt),*; $(const $const_generic: $const_bound),*; $($generic),*>
            impl Mul$(<$rhs>)? for $name
            $(where $($bind: $bound,)*)?
            {
                fn = mul;
                scalar = $scalar$(, $scalar_rhs)?;
                output = $output;
                output_scalar = $output_scalar;
                handle self = $handle_self;
                handle rhs = $handle_rhs;
            }
        );
        $crate::impl_operator!(
            <$($lt),*; $(const $const_generic: $const_bound),*; $($generic),*>
            impl Div$(<$rhs>)? for $name
            $(where $($bind: $bound,)*)?
            {
                fn = div;
                scalar = $scalar$(, $scalar_rhs)?;
                output = $output;
                output_scalar = $output_scalar;
                handle self = $handle_self;
                handle rhs = $handle_rhs;
            }
        );
    };
}

#[macro_export]
macro_rules! impl_assign_operator {
    (
        <$($lt:lifetime$(,)?)*; $(const $const_generic:ident: $const_bound:ident$(,)?)*; $($generic:ident$(,)?)*>
        impl $trait_name:ident$(<$rhs:ty>)? for $name:ty
        $(where $($bind:ty: $bound:path$(,)?)*)?
        {
            fn = $trait_fn:ident;
            scalar = $scalar:ty$(, $scalar_rhs:ty)?;
            handle rhs = $handle_rhs:expr;
        }
    ) => {
        impl<$($lt, )* $(const $const_generic: $const_bound, )* $($generic, )*> $trait_name$(<$rhs>)? for $name
        where
            $scalar: $trait_name<$($scalar_rhs, )?>,
            $($($bind: $bound,)*)?
        {
            #[inline]
            fn $trait_fn(&mut self, rhs: $crate::or_else!($($rhs)?, Self)) {
                Zip::from(self)
                    .and($handle_rhs(rhs))
                    .for_each(|x, y| x.$trait_fn(y));
            }
        }
    };
    (
        <$($lt:lifetime$(,)?)*; $(const $const_generic:ident: $const_bound:ident$(,)?)*; $($generic:ident$(,)?)*>
        impl Ops$(<$rhs:ty>)? for $name:ty
        $(where $($bind:ty: $bound:path$(,)?)*)? {
            scalar = $scalar:ty$(, $scalar_rhs:ty)?;
            handle rhs = $handle_rhs:expr;
        }
    ) => {
        $crate::impl_assign_operator!(
            <$($lt),*; $(const $const_generic: $const_bound),*; $($generic),*>
            impl AddAssign$(<$rhs>)? for $name
            $(where $($bind: $bound,)*)?
            {
                fn = add_assign;
                scalar = $scalar$(, $scalar_rhs)?;
                handle rhs = $handle_rhs;
            }
        );
    };
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::dense::{Lower, Mat, MatSym, SymKind, Upper};

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
