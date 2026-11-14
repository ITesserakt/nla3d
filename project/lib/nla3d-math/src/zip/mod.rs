mod construct;
mod drive;
pub mod ext;
mod producer;
mod shape;
mod tuple;

pub use construct::{ElementwiseConstructible, Nothing};
pub use drive::{Domain, DomainIndex, Full, Meet, Pattern, Triangular};
pub use producer::{Producer, RawMut};
pub use shape::{Dynamic, Fits, Free, Join, KnownShape, Shape, Static};
pub use tuple::Push;

use crate::dense::{MatSymLength, SymKind};
use crate::zip::ext::FromFn;

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

impl<'p, I: faer::Index, S: Shape> DefaultOutput for Space<S, Pattern<I>> {
    type Shape = S;
    type Domain = Pattern<I>;
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

    #[inline]
    pub fn indexed(self) -> Zip<T::Output, S, D>
    where
        T: Push<FromFn<D, fn(DomainIndex) -> (usize, usize)>>,
        D: Domain,
    {
        Zip(
            self.0
                .push(FromFn::new(self.1.clone(), |ix| (*ix.i, *ix.j))),
            self.1,
            self.2,
        )
    }
}

impl<T, D> Zip<T, Free, D> {
    #[inline]
    pub fn with_shape<S>(self, shape: S) -> Zip<T, S, D> {
        Zip(self.0, self.1, shape)
    }
}
