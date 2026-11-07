mod construct;
mod drive;
mod ext;
mod producer;
mod shape;
mod tuple;

pub use construct::{ElementwiseConstructible, Nothing};
pub use drive::{Domain, DomainIndex, Full, Pattern, Triangular};
pub use ext::{CollectIntoExt, OwnedExt, TriExt};
pub use producer::{Consume, Fill, FillStatic, FromFn, Producer, RawMut, ToOwnedProducer, Tri};
pub use shape::{Dynamic, Fits, Free, Join, KnownShape, Shape, Static};
pub use tuple::Push;

#[cfg(feature = "rand")]
pub use ext::RandomConstructible;

use crate::{
    dense::{MatSymLength, SymKind},
    zip::drive::Meet,
};
use faer::Index;

pub struct Space<S, D>(S, D);

#[diagnostic::on_unimplemented(
    message = "`map` has no default result type for `{Self}`",
    label = "no default output for this shape and domain",
    note = "use `map_into::<M>` / `collect::<M>` with an explicit target, or add `.with_shape(..)`"
)]
pub trait DefaultOutput {
    type Shape: Shape;
    type Domain: Domain;
    type Matrix<V>: ElementwiseConstructible<Self::Shape, Self::Domain, Item = V>;
}

type MatrixFor<S, D, T> = <Space<S, D> as DefaultOutput>::Matrix<T>;

impl<const N: usize, const M: usize> DefaultOutput for Space<Static<N, M>, Full> {
    type Shape = Static<N, M>;
    type Domain = Full;
    type Matrix<V> = crate::dense::Mat<N, M, V>;
}

impl DefaultOutput for Space<Dynamic, Full> {
    type Shape = Dynamic;
    type Domain = Full;
    type Matrix<V> = crate::dense::DMat<V>;
}

impl<const N: usize, K: SymKind> DefaultOutput for Space<Static<N, N>, Triangular<K>>
where
    generic_array::typenum::Const<N>: MatSymLength,
{
    type Shape = Static<N, N>;
    type Domain = Triangular<K>;
    type Matrix<V> = crate::dense::MatSym<N, V, K>;
}

impl<'p, I: Index, S: Shape> DefaultOutput for Space<S, Pattern<'p, I>> {
    type Shape = S;
    type Domain = Pattern<'p, I>;
    type Matrix<V> = crate::sparse::Mat<V, I>;
}

pub struct Zip<T, S, D>(T, D, S);

impl Zip<(), Free, Full> {
    #[inline]
    pub fn from<T>(value: T) -> Zip<(T,), T::Shape, T::Domain>
    where
        T: Producer,
    {
        let domain = value.domain();
        let shape = value.shape();
        Zip((value,), domain, shape)
    }

    pub fn from_fn<T, F, D>(domain: D, f: F) -> Zip<(FromFn<D, F>,), Free, D>
    where
        F: FnMut(DomainIndex) -> T,
        D: Domain,
    {
        Zip::from(FromFn::new(domain, f))
    }
}

impl<T, S, D> Zip<T, S, D> {
    #[inline]
    pub fn and<U>(self, value: U) -> Zip<T::Output, S::Output, D::Output>
    where
        T: Push<U>,
        U: Producer,
        D: Meet<U::Domain>,
        S: Join<U::Shape>,
    {
        let domain = value.domain();
        let shape = value.shape();
        Zip(self.0.push(value), self.1.meet(domain), self.2.join(shape))
    }
}

impl<T, D> Zip<T, Free, D> {
    #[inline]
    pub fn with_shape<S>(self, shape: S) -> Zip<T, S, D> {
        Zip(self.0, self.1, shape)
    }
}

#[cfg(test)]
mod tests {
    use core::mem::ManuallyDrop;

    use matrixcompare::assert_matrix_eq;

    use crate::dense::Mat;
    use crate::zip::Zip;
    use crate::zip::producer::Consume;
    use crate::{dense::MatSym, m};

    const A: Mat<3, 3, usize> = m![[1, 2, 3], [2, 3, 4], [5, 6, 7]];

    fn b() -> MatSym<3, f64> {
        MatSym::from_mat(m![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    }

    #[test]
    fn test_zip_fold() {
        let sum = Zip::from(&A)
            .and(&b())
            .fold(0.0, |acc, x, y| acc + (*x as f64) * *y);

        assert_eq!(sum, 4.0);
    }

    #[test]
    fn test_zip_for_each() {
        let mut total_non_zero = 0;
        Zip::from(&A).and(&b()).for_each(|x, y| {
            if (*x as f64) * *y != 0.0 {
                total_non_zero += 1;
            }
        });

        assert_eq!(total_non_zero, 2);
    }

    #[test]
    fn test_zip_map_different_types() {
        let out = Zip::from(&A).and(&b()).map(|x, y| (*x as f64) * *y);

        assert_matrix_eq!(out, m![[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_zip_map_same_type() {
        let out: Mat<_, _, u8> = Zip::from(&b()).and(&b()).map_into(|x, y| (*x * *y) as u8);

        assert_matrix_eq!(out, m![[0, 1, 0], [1, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn for_each_mut_items_can_be_retained() {
        let mut a = m![[1, 2], [3, 4]].into_dyn();
        let mut v = Vec::new();
        Zip::from(&mut a).for_each(|x| v.push(x));
        for x in v {
            *x += 1;
        }
    }

    #[test]
    fn change_output_type() {
        let a = m![[1, 2], [3, 4]];
        let b = Zip::from(Consume(a)).map(|x| x);

        assert_matrix_eq!(a, b);
    }

    #[test]
    fn test_panicing() {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        struct IncOnDrop(u32);

        impl Drop for IncOnDrop {
            fn drop(&mut self) {
                COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let a = m![[IncOnDrop(1), IncOnDrop(2)], [IncOnDrop(3), IncOnDrop(4)]];

        let result = std::panic::catch_unwind(|| {
            Zip::from(Consume(a)).for_each(|x| {
                let x = ManuallyDrop::new(x);
                if x.0 == 3 {
                    panic!("Test");
                }
            });
        });

        assert_eq!(COUNTER.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(matches!(result, Err(_)));
    }
}
