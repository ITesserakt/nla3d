use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr::NonNull;

use faer::{MatMut, MatRef, Stride};
use generic_array::typenum::Const;
use platform::compat::borrow::ToOwned;

use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
use crate::zip::drive::{Domain, DomainIndex, Full, Triangular};
use crate::zip::{Dynamic, Free, Shape, Static};

/// # Safety
/// - `shape()` is the true extent: `get_unchecked` is sound for every
///   `i < rows`, `j < cols` on a cursor from `into_cursor`, each `(i, j)` at most once.
/// - Items for distinct `(i, j)` must not alias unless they are shared references.
/// - `get_unchecked` must not create a reference covering memory other than
///   the returned item (it would invalidate items returned earlier).
pub unsafe trait Producer: Sized {
    type Domain: Domain;
    type Shape: Shape;
    type Item;
    type Cursor;

    fn domain(&self) -> Self::Domain;
    fn shape(&self) -> Self::Shape;
    fn into_cursor(self) -> Self::Cursor;

    /// # Safety
    /// `(i, j)` within `shape()`, requested at most once per cursor.
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item;
}

pub struct ConsumeProducer<T>(pub T);

pub struct ConsumeCursor<const N: usize, const M: usize, T> {
    mat: ManuallyDrop<Mat<N, M, T>>,
    taken: [[bool; N]; M],
}

pub struct Fill<T>(pub T);

pub struct FillStatic<const N: usize, const M: usize, T>(pub T);

/// Raw pointer to a mutably borrowed container; made once, never re-borrowed.
pub struct RawMut<'a, X: ?Sized> {
    ptr: NonNull<X>,
    _marker: PhantomData<&'a mut X>,
}

unsafe impl<'a, const N: usize, const M: usize, T> Producer for &'a Mat<N, M, T> {
    type Domain = Full;
    type Shape = Static<N, M>;
    type Item = &'a T;
    type Cursor = Self;

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
    unsafe fn get_unchecked(c: &mut Self::Cursor, index: DomainIndex) -> &'a T {
        unsafe { c.get_unchecked(*index.i, *index.j) }
    }
}

unsafe impl<'a, const N: usize, const M: usize, T> Producer for &'a mut Mat<N, M, T> {
    type Domain = Full;
    type Shape = Static<N, M>;
    type Item = &'a mut T;
    type Cursor = RawMut<'a, T>;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        RawMut {
            ptr: self.as_mut_ptr().cast(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, index: DomainIndex) -> &'a mut T {
        unsafe { c.ptr.add(*index.j * N + *index.i).as_mut() }
    }
}

unsafe impl<'a, const N: usize, T, Kind> Producer for &'a MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
{
    type Domain = Triangular<Kind>;
    type Shape = Static<N, N>;
    type Item = &'a T;
    type Cursor = Self;

    fn domain(&self) -> Self::Domain {
        Triangular::default()
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
    unsafe fn get_unchecked(c: &mut Self::Cursor, index: DomainIndex) -> &'a T {
        unsafe { c.get_unchecked(*index.i, *index.j) }
    }
}

unsafe impl<'a, T, Rows, Cols, RStride, CStride> Producer
    for MatRef<'a, T, Rows, Cols, RStride, CStride>
where
    Rows: faer::Shape + 'a,
    Cols: faer::Shape + 'a,
    RStride: Stride,
    CStride: Stride,
    T: 'a,
{
    type Domain = Full;
    type Shape = Dynamic;
    type Item = &'a T;
    type Cursor = Self;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Dynamic {
            rows: self.nrows().unbound(),
            cols: self.ncols().unbound(),
        }
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, index: DomainIndex) -> &'a T {
        let i = unsafe { c.nrows().unchecked_idx(*index.i) };
        let j = unsafe { c.ncols().unchecked_idx(*index.j) };
        unsafe { c.ptr_inbounds_at(i, j).as_ref_unchecked() }
    }
}

unsafe impl<'a, T, Rows, Cols, RStride, CStride> Producer
    for MatMut<'a, T, Rows, Cols, RStride, CStride>
where
    Rows: faer::Shape,
    Cols: faer::Shape,
    RStride: Stride,
    CStride: Stride,
{
    type Domain = Full;
    type Shape = Dynamic;
    type Item = &'a mut T;
    type Cursor = Self;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Dynamic {
            rows: self.nrows().unbound(),
            cols: self.ncols().unbound(),
        }
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, index: DomainIndex) -> &'a mut T {
        let i = unsafe { c.nrows().unchecked_idx(*index.i) };
        let j = unsafe { c.ncols().unchecked_idx(*index.j) };
        unsafe { c.ptr_inbounds_at_mut(i, j).as_mut_unchecked() }
    }
}

unsafe impl<'a, T, Rows: faer::Shape, Cols: faer::Shape> Producer for &'a DMat<T, Rows, Cols> {
    type Domain = Full;
    type Shape = Dynamic;
    type Item = &'a T;
    type Cursor = Self;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Dynamic {
            rows: self.nrows().unbound(),
            cols: self.ncols().unbound(),
        }
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let i = unsafe { cursor.nrows().unchecked_idx(*index.i) };
        let j = unsafe { cursor.ncols().unchecked_idx(*index.j) };
        unsafe { cursor.ptr_inbounds_at(i, j).as_ref_unchecked() }
    }
}

unsafe impl<'a, T, Rows: faer::Shape, Cols: faer::Shape> Producer for &'a mut DMat<T, Rows, Cols> {
    type Domain = Full;
    type Shape = Dynamic;
    type Item = &'a mut T;
    type Cursor = Self;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Dynamic {
            rows: self.nrows().unbound(),
            cols: self.ncols().unbound(),
        }
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let i = unsafe { cursor.nrows().unchecked_idx(*index.i) };
        let j = unsafe { cursor.ncols().unchecked_idx(*index.j) };
        unsafe { cursor.ptr_inbounds_at_mut(i, j).as_mut_unchecked() }
    }
}

unsafe impl<const N: usize, const M: usize, T> Producer for ConsumeProducer<Mat<N, M, T>> {
    type Domain = Full;
    type Shape = Static<N, M>;
    type Item = T;
    type Cursor = ConsumeCursor<N, M, T>;

    fn domain(&self) -> Self::Domain {
        Full
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        ConsumeCursor {
            mat: ManuallyDrop::new(self.0),
            taken: [[false; N]; M],
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let value = unsafe { std::ptr::read(cursor.mat.get_unchecked(*index.i, *index.j)) };
        cursor.taken[*index.j][*index.i] = true;
        value
    }
}

impl<const N: usize, const M: usize, T> Drop for ConsumeCursor<N, M, T> {
    fn drop(&mut self) {
        if const { !std::mem::needs_drop::<T>() } {
            return;
        }

        let base = self.mat.as_mut_ptr().cast::<T>();
        self.taken
            .as_flattened()
            .iter()
            .enumerate()
            .filter(|&(_, taken)| !taken)
            .map(|(offset, _)| unsafe { base.add(offset) })
            .for_each(|it| unsafe { it.drop_in_place() });
    }
}

unsafe impl<T: Clone> Producer for Fill<T> {
    type Domain = Full;
    type Shape = Free;
    type Item = T;
    type Cursor = Self;

    fn domain(&self) -> Self::Domain {
        Full
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
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, _: DomainIndex) -> Self::Item {
        cursor.0.clone()
    }
}

unsafe impl<const N: usize, const M: usize, T: Clone> Producer for FillStatic<N, M, T> {
    type Domain = Full;
    type Shape = Static<N, M>;
    type Item = T;
    type Cursor = Self;

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
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, _: DomainIndex) -> Self::Item {
        cursor.0.clone()
    }
}

unsafe impl<'a, const N: usize, T, K: SymKind> Producer for &'a mut MatSym<N, T, K>
where
    Const<N>: MatSymLength,
{
    type Domain = Triangular<K>;
    type Shape = Static<N, N>;
    type Item = &'a mut T;
    type Cursor = RawMut<'a, [T]>;

    fn domain(&self) -> Self::Domain {
        Triangular(PhantomData)
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Static
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        RawMut {
            ptr: self.as_mut_ptr(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let base = cursor.ptr.cast::<T>();
        unsafe { base.add(index.pos).as_mut() }
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

pub struct ToOwnedProducer<T>(pub T);

unsafe impl<T, O> Producer for ToOwnedProducer<T>
where
    T: Producer,
    T::Item: ToOwned<Owned = O>,
{
    type Domain = T::Domain;
    type Shape = T::Shape;
    type Item = O;
    type Cursor = T::Cursor;

    fn domain(&self) -> Self::Domain {
        T::domain(&self.0)
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
        let item = unsafe { T::get_unchecked(cursor, index) };
        item.to_owned()
    }
}

pub struct FromFn<D, F>(D, F);

impl<D, F> FromFn<D, F> {
    pub fn new<T>(domain: D, f: F) -> Self
    where
        F: FnMut(DomainIndex) -> T,
    {
        Self(domain, f)
    }
}

unsafe impl<D, T, F> Producer for FromFn<D, F>
where
    D: Domain,
    F: FnMut(DomainIndex) -> T,
{
    type Domain = D;
    type Shape = Free;
    type Item = T;
    type Cursor = F;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        self.0.clone()
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        Free
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self.1
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        cursor(index)
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
