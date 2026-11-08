use core::marker::PhantomData;

use generic_array::typenum::Const;

use crate::dense::{DMat, Lower, Mat, MatSym, MatSymLength, SymKind, Upper};
use crate::zip::producer::ExpandProducer;
use crate::zip::{ConsumeProducer, ElementwiseConstructible, KnownShape};
use crate::zip::{
    Producer, Zip,
    drive::Full,
    producer::{ToOwnedProducer, Tri},
};

pub trait TriExt: Sized {
    fn lower(self) -> Tri<Lower, Self>;
    fn upper(self) -> Tri<Upper, Self>;
    fn tri<K: SymKind>(self) -> Tri<K, Self>;
}

impl<T> TriExt for T
where
    T: Producer<Domain = Full>,
{
    #[inline(always)]
    fn lower(self) -> Tri<Lower, Self> {
        Tri(self, PhantomData)
    }

    #[inline(always)]
    fn upper(self) -> Tri<Upper, Self> {
        Tri(self, PhantomData)
    }

    #[inline(always)]
    fn tri<K: SymKind>(self) -> Tri<K, Self> {
        Tri(self, PhantomData)
    }
}

pub trait CollectIntoExt: Producer
where
    Self::Shape: KnownShape,
{
    fn into_dyn(self) -> DMat<Self::Item>
    where
        DMat<Self::Item>: ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>;

    fn into_static<const N: usize, const M: usize>(self) -> Mat<N, M, Self::Item>
    where
        Mat<N, M, Self::Item>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>;

    fn into_lower<const N: usize>(self) -> MatSym<N, Self::Item, Lower>
    where
        MatSym<N, Self::Item, Lower>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength;

    fn into_upper<const N: usize>(self) -> MatSym<N, Self::Item, Upper>
    where
        MatSym<N, Self::Item, Upper>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength;
}

impl<T: Producer<Shape: KnownShape>> CollectIntoExt for T {
    #[inline]
    fn into_dyn(self) -> DMat<Self::Item>
    where
        DMat<Self::Item>: ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
    {
        Zip::from(self).collect()
    }

    #[inline]
    fn into_static<const N: usize, const M: usize>(self) -> Mat<N, M, Self::Item>
    where
        Mat<N, M, Self::Item>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
    {
        Zip::from(self).collect()
    }

    #[inline]
    fn into_lower<const N: usize>(self) -> MatSym<N, Self::Item, Lower>
    where
        MatSym<N, Self::Item, Lower>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength,
    {
        Zip::from(self).collect()
    }

    #[inline]
    fn into_upper<const N: usize>(self) -> MatSym<N, Self::Item, Upper>
    where
        MatSym<N, Self::Item, Upper>:
            ElementwiseConstructible<Self::Shape, Self::Domain, Item = Self::Item>,
        Const<N>: MatSymLength,
    {
        Zip::from(self).collect()
    }
}

pub trait OwnedExt: Sized {
    fn as_owned(self) -> ToOwnedProducer<Self>;
}

impl<T: Producer> OwnedExt for T {
    #[inline]
    fn as_owned(self) -> ToOwnedProducer<Self> {
        ToOwnedProducer(self)
    }
}

pub trait ExpandExt: Sized {
    fn expand(self) -> ExpandProducer<Self>;
}

impl<T> ExpandExt for T
where
    ExpandProducer<T>: Producer,
{
    fn expand(self) -> ExpandProducer<Self> {
        ExpandProducer(self)
    }
}

pub trait ConsumeExt: Sized {
    fn consume(self) -> ConsumeProducer<Self>;
}

impl<T> ConsumeExt for T
where 
    ConsumeProducer<T>: Producer
{
    fn consume(self) -> ConsumeProducer<Self> {
        ConsumeProducer(self)
    }
}

#[cfg(feature = "rand")]
pub trait RandomConstructible<S: KnownShape, D> {
    type Item;

    fn random(
        shape: S,
        rng: &mut impl rand::Rng,
        distribution: impl rand::prelude::Distribution<Self::Item>,
    ) -> Self;

    fn random_uniform(shape: S, rng: &mut impl rand::Rng) -> Self
    where
        Self: Sized,
        rand::distr::StandardUniform: rand::prelude::Distribution<Self::Item>,
    {
        Self::random(shape, rng, rand::distr::StandardUniform)
    }
}

#[cfg(feature = "rand")]
const _: () = {
    use crate::zip::{ElementwiseConstructible, drive::Domain};

    impl<S, E, D> RandomConstructible<S, D> for E
    where
        E: ElementwiseConstructible<S, D>,
        D: Domain + Default,
        S: KnownShape,
    {
        type Item = E::Item;

        #[inline]
        fn random(
            shape: S,
            rng: &mut impl rand::Rng,
            distribution: impl rand::prelude::Distribution<Self::Item>,
        ) -> Self {
            use crate::zip::Zip;

            Zip::from_fn(D::default(), |_| distribution.sample(rng))
                .with_shape(shape)
                .collect()
        }
    }
};
