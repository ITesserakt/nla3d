use core::marker::PhantomData;
use core::ptr::NonNull;

use faer::{MatMut, MatRef, Stride};
use generic_array::typenum::Const;

use crate::dense::{DMat, Mat, MatSym, MatSymLength, SymKind};
use crate::sparse;
use crate::sparse::SparseLocation;
use crate::zip::drive::{Domain, DomainIndex, Full, Triangular};
use crate::zip::{Dynamic, Pattern, Shape, Static};

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

unsafe impl<'a, T, I, R, C> Producer for &'a sparse::Mat<T, I, R, C>
where
    I: faer::Index,
    R: faer::Shape,
    C: faer::Shape,
{
    type Domain = Pattern<I>;
    type Shape = Dynamic;
    type Item = &'a T;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        self.sparsity().pattern.clone()
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        let info = self.sparsity();
        Dynamic {
            rows: info.rows.unbound(),
            cols: info.cols.unbound(),
        }
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let pos = unsafe { SparseLocation::new(index.pos) };
        unsafe { cursor.get_unchecked(pos) }
    }
}

unsafe impl<'a, T, I, R, C> Producer for &'a mut sparse::Mat<T, I, R, C>
where
    I: faer::Index,
    R: faer::Shape,
    C: faer::Shape,
{
    type Domain = Pattern<I>;
    type Shape = Dynamic;
    type Item = &'a mut T;
    type Cursor = Self;

    #[inline(always)]
    fn domain(&self) -> Self::Domain {
        self.sparsity().pattern.clone()
    }

    #[inline(always)]
    fn shape(&self) -> Self::Shape {
        let info = self.sparsity();
        Dynamic {
            rows: info.rows.unbound(),
            cols: info.cols.unbound(),
        }
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, index: DomainIndex) -> Self::Item {
        let pos = unsafe { SparseLocation::new(index.pos) };
        let mut slot = unsafe { cursor.get_unchecked_ptr_mut(pos) };
        unsafe { slot.as_mut() }
    }
}
