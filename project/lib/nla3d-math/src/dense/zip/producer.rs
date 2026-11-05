use core::mem::ManuallyDrop;
use core::ptr::NonNull;
use core::{marker::PhantomData, mem::MaybeUninit};

use faer::{MatMut, MatRef, Shape, Stride, utils::bound::Idx};
use generic_array::typenum::Const;

use crate::dense::mat::ArrayStorage;
use crate::dense::{Mat, MatSym, MatSymLength, SymKind};

/// # Safety
/// - `shape()` is the true extent: `get_unchecked` is sound for every
///   `i < rows`, `j < cols` on a cursor from `into_cursor`, each `(i, j)` at most once.
/// - Items for distinct `(i, j)` must not alias unless they are shared references.
/// - `get_unchecked` must not create a reference covering memory other than
///   the returned item (it would invalidate items returned earlier).
pub unsafe trait Producer: Sized {
    type Item;
    type Cursor;

    fn shape(&self) -> (usize, usize);
    fn into_cursor(self) -> Self::Cursor;

    /// # Safety
    /// `(i, j)` within `shape()`, requested at most once per cursor.
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, i: Idx, j: Idx) -> Self::Item;
}

pub struct Consume<T>(pub T);

pub struct ConsumeCursor<const N: usize, const M: usize, T> {
    mat: ManuallyDrop<Mat<N, M, T>>,
    taken: usize,
}

pub struct Fill<T> {
    pub element: T,
    pub shape: (usize, usize),
}

/// Raw pointer to a mutably borrowed container; made once, never re-borrowed.
pub struct RawMut<'a, X> {
    ptr: NonNull<X>,
    _marker: PhantomData<&'a mut X>,
}

impl<'a, X> RawMut<'a, X> {
    #[inline(always)]
    fn new(x: &'a mut X) -> Self {
        Self {
            ptr: NonNull::from(x),
            _marker: PhantomData,
        }
    }
}

unsafe impl<'a, const N: usize, const M: usize, T> Producer for &'a Mat<N, M, T> {
    type Item = &'a T;
    type Cursor = Self;

    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        (N, M)
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, i: Idx, j: Idx) -> &'a T {
        let this: &'a Mat<N, M, T> = *c;
        let col = unsafe { this.0.0.get_unchecked(*j) };
        unsafe { col.get_unchecked(*i) }
    }
}

unsafe impl<'a, const N: usize, const M: usize, T> Producer for &'a mut Mat<N, M, T> {
    type Item = &'a mut T;
    type Cursor = RawMut<'a, Mat<N, M, T>>;

    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        (N, M)
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        RawMut::new(self)
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, i: Idx, j: Idx) -> &'a mut T {
        // Assumes `.0.0` is contiguous column-major storage (N elements per column).
        const { assert!(size_of::<Mat<N, M, T>>() == N * M * size_of::<T>()) };
        // Pointer arithmetic only: no `&mut` to the matrix or a column is created.
        let data = unsafe { &raw mut (*c.ptr.as_ptr()).0.0 }.cast::<T>();
        unsafe { &mut *data.add(*j * N + *i) }
    }
}

unsafe impl<'a, const N: usize, T, Kind> Producer for &'a MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
{
    type Item = &'a T;
    type Cursor = Self;

    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        (N, N)
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, i: Idx, j: Idx) -> &'a T {
        let this: &'a MatSym<N, T, Kind> = *c;
        unsafe { this.get_unchecked(*i, *j) }
    }
}

unsafe impl<'a, T, Rows, Cols, RStride, CStride> Producer
    for MatRef<'a, T, Rows, Cols, RStride, CStride>
where
    Rows: Shape + 'a,
    Cols: Shape + 'a,
    RStride: Stride,
    CStride: Stride,
    T: 'a,
{
    type Item = &'a T;
    type Cursor = Self;

    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        (self.nrows().unbound(), self.ncols().unbound())
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, i: Idx, j: Idx) -> &'a T {
        let i = unsafe { c.nrows().unchecked_idx(*i) };
        let j = unsafe { c.ncols().unchecked_idx(*j) };
        unsafe { c.ptr_inbounds_at(i, j).as_ref_unchecked() }
    }
}

unsafe impl<'a, T, Rows, Cols, RStride, CStride> Producer
    for MatMut<'a, T, Rows, Cols, RStride, CStride>
where
    Rows: Shape,
    Cols: Shape,
    RStride: Stride,
    CStride: Stride,
{
    type Item = &'a mut T;
    type Cursor = Self;

    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        (self.nrows().unbound(), self.ncols().unbound())
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        self
    }

    #[inline(always)]
    unsafe fn get_unchecked(c: &mut Self::Cursor, i: Idx, j: Idx) -> &'a mut T {
        let i = unsafe { c.nrows().unchecked_idx(*i) };
        let j = unsafe { c.ncols().unchecked_idx(*j) };
        unsafe { c.ptr_inbounds_at_mut(i, j).as_mut_unchecked() }
    }
}

unsafe impl<const N: usize, const M: usize, T> Producer for Consume<Mat<N, M, T>> {
    type Item = T;
    type Cursor = ConsumeCursor<N, M, T>;

    #[inline(always)]
    fn shape(&self) -> (usize, usize) {
        (N, M)
    }

    #[inline(always)]
    fn into_cursor(self) -> Self::Cursor {
        ConsumeCursor {
            mat: ManuallyDrop::new(self.0),
            taken: 0,
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(cursor: &mut Self::Cursor, i: Idx, j: Idx) -> Self::Item {
        let value = unsafe { std::ptr::read(cursor.mat.get_unchecked(*i, *j)) };
        cursor.taken = *j * N + *i + 1;
        value
    }
}

impl<const N: usize, const M: usize, T> Drop for ConsumeCursor<N, M, T> {
    fn drop(&mut self) {
        let base = self.mat.0.0.as_flattened_mut().as_mut_ptr();
        for k in self.taken..N * M {
            unsafe {
                std::ptr::drop_in_place(base.add(k));
            }
        }
    }
}

unsafe impl<T: Clone> Producer for Fill<T> {
    type Item = T;
    type Cursor = Self;

    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn into_cursor(self) -> Self::Cursor {
        self
    }

    unsafe fn get_unchecked(cursor: &mut Self::Cursor, _: Idx, _: Idx) -> Self::Item {
        cursor.element.clone()
    }
}
