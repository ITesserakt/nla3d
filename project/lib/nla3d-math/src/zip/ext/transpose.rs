use std::ops::Index;

use faer::mat::{AsMatMut, AsMatRef};
use faer::{MatMut, MatRef};

use crate::dense::SymKind;
use crate::sparse::SparsityInfo;
use crate::zip::shape::ShapeTranspose;
use crate::zip::{
    DefaultOutput, Domain, DomainIndex, Full, Join, KnownShape, MatrixFor, Meet, Pattern, Producer,
    Space, Triangular,
};

pub trait TransposeExt: Sized {
    type Producer: Producer;

    fn transpose(self) -> Self::Producer;

    fn transposed<S, D, T>(self) -> MatrixFor<S, D, T>
    where
        Self::Producer: Producer<Shape = S, Domain = D, Item = T>,
        S: KnownShape,
        D: Domain,
        Space<S, D>: DefaultOutput<Shape = S, Domain = D>,
    {
        crate::zip::Zip::from(self.transpose()).collect()
    }

    fn is_symmetric<T, D, S>(self) -> bool
    where
        Self: Copy + Producer<Item = T, Domain = D, Shape = S>,
        Self::Producer: Producer<Item = T, Domain = D, Shape = <S as ShapeTranspose>::Output>,
        T: PartialEq,
        D: Domain + Meet<D>,
        S: ShapeTranspose + Join<<S as ShapeTranspose>::Output, Output: KnownShape>,
    {
        crate::zip::Zip::from(self)
            .and(self.transpose())
            .fold(true, |acc, a, b| acc && a == b)
    }
}

impl<T> TransposeExt for T
where
    TransposeProducer<T>: Producer,
{
    type Producer = TransposeProducer<T>;

    fn transpose(self) -> Self::Producer {
        TransposeProducer(self)
    }
}

pub struct TransposeProducer<P>(pub(crate) P);

pub trait Helper<P>
where
    P: Producer<Domain = Self>,
{
    type Cursor;
    type Item;

    fn into_cursor(producer: P) -> Self::Cursor;

    unsafe fn get_transposed_unchecked(cursor: &mut Self::Cursor, index: DomainIndex)
    -> Self::Item;
}

unsafe impl<P> Producer for TransposeProducer<P>
where
    P: Producer,
    P::Domain: Helper<P>,
    P::Shape: ShapeTranspose,
{
    type Domain = P::Domain;
    type Shape = <P::Shape as ShapeTranspose>::Output;
    type Item = <P::Domain as Helper<P>>::Item;
    type Cursor = <P::Domain as Helper<P>>::Cursor;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        self.0.domain()
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        self.0.shape().transpose()
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        P::Domain::into_cursor(self.0)
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        unsafe { P::Domain::get_transposed_unchecked(cursor, index) }
    }
}

impl<P> Helper<P> for Full
where
    P: Producer<Domain = Self>,
    P::Shape: KnownShape,
{
    type Cursor = (P::Cursor, P::Shape);
    type Item = P::Item;

    #[inline]
    fn into_cursor(producer: P) -> Self::Cursor {
        let shape = producer.shape();
        (producer.into_cursor(), shape)
    }

    unsafe fn get_transposed_unchecked(
        cursor: &mut Self::Cursor,
        index: DomainIndex,
    ) -> Self::Item {
        let (_, cols) = cursor.1.materialize();
        let index = DomainIndex {
            i: index.j,
            j: index.i,
            pos: *index.i * cols + *index.j,
        };
        unsafe { P::get_unchecked(&mut cursor.0, index) }
    }
}

impl<P, K> Helper<P> for Triangular<K>
where
    P: Producer<Domain = Self>,
    K: SymKind,
{
    type Cursor = P::Cursor;
    type Item = P::Item;

    fn into_cursor(producer: P) -> Self::Cursor {
        producer.into_cursor()
    }

    unsafe fn get_transposed_unchecked(
        cursor: &mut Self::Cursor,
        index: DomainIndex,
    ) -> Self::Item {
        unsafe { P::get_unchecked(cursor, index) }
    }
}

impl<P, I: faer::Index> Helper<P> for Pattern<I>
where
    P: Producer<Domain = Self>,
    P::Shape: KnownShape,
{
    type Cursor = (P::Cursor, SparsityInfo<I, usize, usize>);
    type Item = Option<P::Item>;

    fn into_cursor(producer: P) -> Self::Cursor {
        let domain = producer.domain();
        let shape = producer.shape().materialize();
        (
            producer.into_cursor(),
            SparsityInfo::from_pattern(shape, domain),
        )
    }

    unsafe fn get_transposed_unchecked(
        (cursor, info): &mut Self::Cursor,
        index: DomainIndex,
    ) -> Self::Item {
        let location = info.get(*index.j, *index.i)?;
        let pos = unsafe { location.into_inner() };
        let index = DomainIndex {
            i: index.j,
            j: index.i,
            pos,
        };

        Some(unsafe { P::get_unchecked(cursor, index) })
    }
}

impl<T: AsMatRef> TransposeProducer<T> {
    pub fn view(&self) -> MatRef<'_, T::T, T::Cols, T::Rows> {
        self.0.as_mat_ref().transpose()
    }
}

impl<T: AsMatMut> TransposeProducer<T> {
    pub fn view_mut(&mut self) -> MatMut<'_, T::T, T::Cols, T::Rows> {
        self.0.as_mat_mut().transpose_mut()
    }
}

impl<T: Index<(usize, usize)>> Index<(usize, usize)> for TransposeProducer<T> {
    type Output = T::Output;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.0.index((j, i))
    }
}

impl<T: Index<(usize, usize)>> Index<(usize, usize)> for &TransposeProducer<T> {
    type Output = T::Output;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.0.index((j, i))
    }
}
