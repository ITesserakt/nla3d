use core::marker::PhantomData;

use generic_array::typenum::Const;

use crate::dense::{DMat, Lower, Mat, MatSym, MatSymLength, Upper};
use crate::zip::{ElementwiseConstructible, KnownShape};
use crate::zip::{
    Producer, Zip,
    drive::Full,
    producer::{ToOwnedProducer, Tri},
};

pub trait TriExt: Sized {
    fn lower(self) -> Tri<Lower, Self>;
    fn upper(self) -> Tri<Upper, Self>;
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
    use rand::{Rng, distr::Distribution};

    use crate::zip::{
        ElementwiseConstructible, Free,
        drive::{Domain, DomainIndex},
    };

    struct RandomProducer<'a, R, D, T, Dom> {
        distribution: D,
        rng: &'a mut R,
        _phantom: PhantomData<(T, Dom)>,
    }

    unsafe impl<R, D, T, Dom> Producer for RandomProducer<'_, R, D, T, Dom>
    where
        D: Distribution<T>,
        R: Rng,
        Dom: Domain + Default,
    {
        type Domain = Dom;
        type Shape = Free;
        type Item = T;
        type Cursor = Self;

        fn domain(&self) -> Self::Domain {
            Dom::default()
        }

        #[inline(always)]
        fn shape(&self) -> Self::Shape {
            Free
        }

        #[inline(always)]
        fn into_cursor(self) -> Self::Cursor {
            self
        }

        #[inline(always)]
        unsafe fn get_unchecked(cursor: &mut Self::Cursor, _index: DomainIndex) -> Self::Item {
            cursor.distribution.sample(cursor.rng)
        }
    }

    impl<S, E, D> RandomConstructible<S, D> for E
    where
        E: ElementwiseConstructible<S, D>,
        D: Domain + Default,
        S: KnownShape
    {
        type Item = E::Item;

        #[inline]
        fn random(
            shape: S,
            rng: &mut impl rand::Rng,
            distribution: impl rand::prelude::Distribution<Self::Item>,
        ) -> Self {
            use crate::zip::Zip;

            let producer = RandomProducer {
                distribution,
                rng,
                _phantom: PhantomData,
            };

            Zip::from(producer).with_shape(shape).map_into(|x| x)
        }
    }
};
