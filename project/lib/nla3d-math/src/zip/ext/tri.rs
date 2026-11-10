use std::marker::PhantomData;

use crate::dense::{Lower, Upper};
use crate::zip::{DomainIndex, Full, Producer, SymKind, Triangular};

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

pub struct Tri<K: SymKind, T>(pub T, pub PhantomData<K>);

unsafe impl<K: SymKind, T> Producer for Tri<K, T>
where
    T: Producer<Domain = Full>,
{
    type Domain = Triangular<K>;
    type Shape = T::Shape;
    type Item = T::Item;
    type Cursor = T::Cursor;

    fn domain(&self) -> Self::Domain {
        Triangular(PhantomData)
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
        unsafe { T::get_unchecked(cursor, index) }
    }
}