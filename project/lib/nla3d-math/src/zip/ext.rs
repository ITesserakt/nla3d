use core::marker::PhantomData;

use generic_array::typenum::Const;

use crate::dense::{DMat, Lower, Mat, MatSym, MatSymLength, Upper};
use crate::zip::{
    Dynamic, Fits, Producer, Static, Symmetric, Zip,
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

pub trait CollectIntoExt: Producer {
    fn into_dyn(self) -> DMat<Self::Item>
    where
        Self::Layout: Fits<Dynamic>;

    fn into_static<const N: usize, const M: usize>(self) -> Mat<N, M, Self::Item>
    where
        Self::Layout: Fits<Static<N, M>>;

    fn into_lower<const N: usize>(self) -> MatSym<N, Self::Item, Lower>
    where
        Self::Layout: Fits<Symmetric<N, Lower>>,
        Const<N>: MatSymLength;

    fn into_upper<const N: usize>(self) -> MatSym<N, Self::Item, Upper>
    where
        Self::Layout: Fits<Symmetric<N, Upper>>,
        Const<N>: MatSymLength;
}

impl<T: Producer> CollectIntoExt for T {
    #[inline]
    fn into_dyn(self) -> DMat<Self::Item>
    where
        Self::Layout: Fits<Dynamic>,
    {
        Zip::from(self).collect()
    }

    #[inline]
    fn into_static<const N: usize, const M: usize>(self) -> Mat<N, M, Self::Item>
    where
        Self::Layout: Fits<Static<N, M>>,
    {
        Zip::from(self).collect()
    }

    #[inline]
    fn into_lower<const N: usize>(self) -> MatSym<N, Self::Item, Lower>
    where
        Self::Layout: Fits<Symmetric<N, Lower>>,
        Const<N>: MatSymLength,
    {
        Zip::from(self).collect()
    }

    #[inline]
    fn into_upper<const N: usize>(self) -> MatSym<N, Self::Item, Upper>
    where
        Self::Layout: Fits<Symmetric<N, Upper>>,
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
pub trait RandomConstructible {
    type Item;

    fn random(
        shape: (usize, usize),
        rng: &mut impl rand::Rng,
        distribution: impl rand::prelude::Distribution<Self::Item>,
    ) -> Self;

    fn random_uniform(shape: (usize, usize), rng: &mut impl rand::Rng) -> Self
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

    use crate::zip::{Any, ElementwiseConstructible};

    struct RandomProducer<'a, R, D, T> {
        shape: (usize, usize),
        distribution: D,
        rng: &'a mut R,
        _phantom: PhantomData<T>,
    }

    unsafe impl<R, D, T> Producer for RandomProducer<'_, R, D, T>
    where
        D: Distribution<T>,
        R: Rng,
    {
        type Domain = Full;
        type Layout = Any;
        type Item = T;
        type Cursor = Self;

        #[inline(always)]
        fn shape(&self) -> (usize, usize) {
            self.shape
        }

        #[inline(always)]
        fn into_cursor(self) -> Self::Cursor {
            self
        }

        #[inline(always)]
        unsafe fn get_unchecked(
            cursor: &mut Self::Cursor,
            _: faer::utils::bound::Idx,
            _: faer::utils::bound::Idx,
        ) -> Self::Item {
            cursor.distribution.sample(cursor.rng)
        }
    }

    impl<E> RandomConstructible for E
    where
        E: ElementwiseConstructible,
    {
        type Item = E::Item;

        #[inline]
        fn random(
            shape: (usize, usize),
            rng: &mut impl rand::Rng,
            distribution: impl rand::prelude::Distribution<Self::Item>,
        ) -> Self {
            use crate::zip::Zip;

            let producer = RandomProducer {
                shape,
                distribution,
                rng,
                _phantom: PhantomData,
            };

            Zip::from(producer).collect()
        }
    }
};
