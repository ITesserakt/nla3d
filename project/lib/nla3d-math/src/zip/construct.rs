use core::mem::MaybeUninit;
use core::{marker::PhantomData, mem::ManuallyDrop};

use faer::{Shape, utils::bound::Idx};
use generic_array::typenum::Const;

use crate::dense::{Mat, MatSym, MatSymLength, SymKind};
use crate::zip::{Any, Dynamic, Layout, Static, Symmetric};

/// # Safety
/// `uninit(shape)` must panic unless `shape` is exactly the shape of `Self`;
/// `set` is sound for every `(i, j)` within that shape.
pub unsafe trait ElementwiseConstructible {
    type Layout: Layout;
    type Item;
    type Uninit;

    fn uninit(shape: (usize, usize)) -> Self::Uninit;

    /// # Safety
    /// `(i, j)` within the shape passed to `uninit`.
    unsafe fn set(matrix: &mut Self::Uninit, i: Idx, j: Idx, value: Self::Item);

    /// # Safety
    /// Every `(i, j)` within the shape was `set` at least once.
    unsafe fn assume_init(value: Self::Uninit) -> Self;
}

unsafe impl<const N: usize, const M: usize, T> ElementwiseConstructible for Mat<N, M, T> {
    type Layout = Static<N, M>;
    type Item = T;
    type Uninit = Mat<N, M, MaybeUninit<T>>;

    #[inline(always)]
    fn uninit(shape: (usize, usize)) -> Self::Uninit {
        assert_eq!(shape, (N, M));
        Mat::uninit()
    }

    #[inline(always)]
    unsafe fn set(matrix: &mut Self::Uninit, i: Idx, j: Idx, value: T) {
        unsafe { matrix.get_unchecked_mut(*i, *j) }.write(value);
    }

    #[inline(always)]
    unsafe fn assume_init(value: Self::Uninit) -> Self {
        unsafe { Mat::assume_init(value) }
    }
}

unsafe impl<const N: usize, T, Kind> ElementwiseConstructible for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
{
    type Layout = Symmetric<N, Kind>;
    type Item = T;
    type Uninit = MatSym<N, MaybeUninit<T>, Kind>;

    #[inline(always)]
    fn uninit(shape: (usize, usize)) -> Self::Uninit {
        assert_eq!(shape, (N, N));
        MatSym::uninit()
    }

    #[inline(always)]
    unsafe fn set(matrix: &mut Self::Uninit, i: Idx, j: Idx, value: T) {
        unsafe { matrix.get_unchecked_mut(*i, *j) }.write(value);
    }

    #[inline(always)]
    unsafe fn assume_init(value: Self::Uninit) -> Self {
        unsafe { MatSym::assume_init(value) }
    }
}

pub enum Nothing<T> {
    Fake(!, PhantomData<T>),
}

unsafe impl<T> ElementwiseConstructible for Nothing<T> {
    type Layout = Any;
    type Item = T;
    type Uninit = !;

    fn uninit(_: (usize, usize)) -> Self::Uninit {
        unreachable!()
    }

    unsafe fn set(matrix: &mut Self::Uninit, _: Idx, _: Idx, _: T) {
        match *matrix {}
    }

    unsafe fn assume_init(value: Self::Uninit) -> Self {
        match value {}
    }
}

unsafe impl<T, Rows, Cols> ElementwiseConstructible for faer::Mat<T, Rows, Cols>
where
    Rows: Shape,
    Cols: Shape,
{
    type Layout = Dynamic;
    type Item = T;
    type Uninit = faer::Mat<MaybeUninit<T>, Rows, Cols>;

    #[inline(always)]
    fn uninit(shape: (usize, usize)) -> Self::Uninit {
        let rows = unsafe { Rows::new_unbound(shape.0) };
        let cols = unsafe { Cols::new_unbound(shape.1) };
        faer::Mat::from_fn(rows, cols, |_, _| const { MaybeUninit::uninit() })
    }

    #[inline(always)]
    unsafe fn set(matrix: &mut Self::Uninit, i: Idx, j: Idx, value: Self::Item) {
        let i = unsafe { matrix.nrows().unchecked_idx(*i) };
        let j = unsafe { matrix.ncols().unchecked_idx(*j) };
        matrix[(i, j)].write(value);
    }

    #[inline(always)]
    unsafe fn assume_init(value: Self::Uninit) -> Self {
        union Cast<T, Rows: Shape, Cols: Shape> {
            uninit: ManuallyDrop<faer::Mat<MaybeUninit<T>, Rows, Cols>>,
            init: ManuallyDrop<faer::Mat<T, Rows, Cols>>,
        }

        let cast = Cast {
            uninit: ManuallyDrop::new(value),
        };
        ManuallyDrop::into_inner(unsafe { cast.init })
    }
}
