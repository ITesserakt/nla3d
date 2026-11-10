use generic_array::typenum::Const;

use crate::dense::{MatSym, MatSymLength, SymKind};
use crate::sparse;
use crate::zip::{DomainIndex, Dynamic, Full, Producer, Static};

pub trait ExpandExt: Sized {
    type Producer: Producer;

    fn expand(self) -> Self::Producer;
}

impl<T> ExpandExt for T
where
    ExpandProducer<T>: Producer,
{
    type Producer = ExpandProducer<T>;

    fn expand(self) -> ExpandProducer<Self> {
        ExpandProducer(self)
    }
}

pub struct ExpandProducer<T>(pub T);

unsafe impl<'a, const N: usize, T, K> Producer for ExpandProducer<&'a MatSym<N, T, K>>
where
    Const<N>: MatSymLength,
    K: SymKind,
{
    type Domain = Full;
    type Shape = Static<N, N>;
    type Item = &'a T;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        unsafe { <&MatSym<N, T, K>>::get_unchecked(&mut cursor.0, index) }
    }
}

unsafe impl<'a, T, I, R, C> Producer for ExpandProducer<&'a sparse::Mat<T, I, R, C>>
where
    I: faer::Index,
    R: faer::Shape,
    C: faer::Shape,
{
    type Domain = Full;
    type Shape = Dynamic;
    type Item = Option<&'a T>;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        self.0.shape()
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let i = unsafe { cursor.0.nrows().unchecked_idx(*index.i) };
        let j = unsafe { cursor.0.ncols().unchecked_idx(*index.j) };
        cursor
            .0
            .sparsity()
            .get(i, j)
            .map(|it| unsafe { cursor.0.get_unchecked(it) })
    }
}

unsafe impl<'a, T, I, R, C> Producer for ExpandProducer<&'a mut sparse::Mat<T, I, R, C>>
where
    I: faer::Index,
    R: faer::Shape,
    C: faer::Shape,
{
    type Domain = Full;
    type Shape = Dynamic;
    type Item = Option<&'a mut T>;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        self.0.shape()
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let i = unsafe { cursor.0.nrows().unchecked_idx(*index.i) };
        let j = unsafe { cursor.0.ncols().unchecked_idx(*index.j) };
        cursor
            .0
            .sparsity()
            .get(i, j)
            .map(|it| unsafe { cursor.0.get_unchecked_ptr_mut(it) })
            .map(|mut it| unsafe { it.as_mut() })
    }
}
