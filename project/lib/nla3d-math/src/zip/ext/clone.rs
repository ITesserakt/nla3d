use std::marker::PhantomData;

use platform::compat::borrow::ToOwned;

use crate::zip::{DomainIndex, Producer};

pub trait OwnedExt<'a, T: 'a>: Producer<Item = &'a T>
where
    T: ToOwned,
{
    type Producer: Producer<Item = T::Owned>;

    fn as_owned(self) -> Self::Producer;
}

impl<'a, P, T> OwnedExt<'a, T> for P
where
    P: Producer<Item = &'a T>,
    T: 'a + ToOwned,
{
    type Producer = ToOwnedProducer<'a, P, T>;

    #[inline]
    fn as_owned(self) -> Self::Producer {
        ToOwnedProducer(self, PhantomData)
    }
}

pub struct ToOwnedProducer<'a, P, T>(pub P, PhantomData<&'a T>);

unsafe impl<'a, P, T> Producer for ToOwnedProducer<'a, P, T>
where
    P: Producer<Item = &'a T>,
    T: ToOwned,
{
    type Domain = P::Domain;
    type Shape = P::Shape;
    type Item = T::Owned;
    type Cursor = P::Cursor;

    fn domain(&self) -> Self::Domain {
        P::domain(&self.0)
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        self.0.shape()
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self.0.into_cursor()
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let item = unsafe { P::get_unchecked(cursor, index) };
        item.to_owned()
    }
}
