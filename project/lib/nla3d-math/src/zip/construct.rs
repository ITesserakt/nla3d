use core::mem::MaybeUninit;
use core::{marker::PhantomData, mem::ManuallyDrop};

use faer::Index;
use generic_array::typenum::Const;
use platform::compat::boxed::Box;

use crate::dense::{Mat, MatSym, MatSymLength, SymKind};
use crate::sparse::{self, SparsityInfo};
use crate::zip::drive::{Domain, DomainIndex, Full, Pattern, Triangular};
use crate::zip::shape::{Fits, Free, Shape};
use crate::zip::{Dynamic, Static};

/// # Safety
/// `uninit(shape)` must panic unless `shape` is exactly the shape of `Self`;
/// `set` is sound for every `(i, j)` within that shape.
pub unsafe trait ElementwiseConstructible<S: Shape, D: Domain> {
    type Item;
    type Uninit;

    fn uninit(shape: (usize, usize), domain: D) -> Self::Uninit;

    /// # Safety
    /// `(i, j)` within the shape passed to `uninit`.
    unsafe fn set(matrix: &mut Self::Uninit, index: DomainIndex, value: Self::Item);

    /// # Safety
    /// Every `(i, j)` within the shape was `set` at least once.
    unsafe fn assume_init(value: Self::Uninit) -> Self;
}

unsafe impl<const N: usize, const M: usize, T, S> ElementwiseConstructible<S, Full> for Mat<N, M, T>
where
    S: Fits<Static<N, M>>,
{
    type Item = T;
    type Uninit = Mat<N, M, MaybeUninit<T>>;

    #[inline(always)]
    fn uninit(shape: (usize, usize), _: Full) -> Self::Uninit {
        assert_eq!(shape, (N, M));
        Mat::uninit()
    }

    #[inline(always)]
    unsafe fn set(matrix: &mut Self::Uninit, index: DomainIndex, value: T) {
        let base = matrix.as_mut_ptr().cast::<MaybeUninit<T>>();
        let mut ith = unsafe { base.add(index.pos) };
        unsafe { ith.as_mut() }.write(value);
    }

    #[inline(always)]
    unsafe fn assume_init(value: Self::Uninit) -> Self {
        unsafe { Mat::assume_init(value) }
    }
}

unsafe impl<const N: usize, T, Kind, S> ElementwiseConstructible<S, Triangular<Kind>>
    for MatSym<N, T, Kind>
where
    Const<N>: MatSymLength,
    Kind: SymKind,
    S: Fits<Static<N, N>>,
{
    type Item = T;
    type Uninit = MatSym<N, MaybeUninit<T>, Kind>;

    #[inline(always)]
    fn uninit(shape: (usize, usize), _: Triangular<Kind>) -> Self::Uninit {
        assert_eq!(shape, (N, N));
        MatSym::uninit()
    }

    #[inline(always)]
    unsafe fn set(matrix: &mut Self::Uninit, index: DomainIndex, value: T) {
        let base = matrix.as_mut_ptr().cast::<MaybeUninit<T>>();
        let mut ith = unsafe { base.add(index.pos) };
        unsafe { ith.as_mut() }.write(value);
    }

    #[inline(always)]
    unsafe fn assume_init(value: Self::Uninit) -> Self {
        unsafe { MatSym::assume_init(value) }
    }
}

pub enum Nothing<T> {
    Fake(!, PhantomData<T>),
}

unsafe impl<T, D: Domain> ElementwiseConstructible<Free, D> for Nothing<T> {
    type Item = T;
    type Uninit = !;

    fn uninit(_: (usize, usize), _: D) -> Self::Uninit {
        unreachable!()
    }

    unsafe fn set(matrix: &mut Self::Uninit, _: DomainIndex, _: T) {
        match *matrix {}
    }

    unsafe fn assume_init(value: Self::Uninit) -> Self {
        match value {}
    }
}

unsafe impl<T, Rows, Cols, S> ElementwiseConstructible<S, Full> for faer::Mat<T, Rows, Cols>
where
    Rows: faer::Shape,
    Cols: faer::Shape,
    S: Fits<Dynamic>,
{
    type Item = T;
    type Uninit = faer::Mat<MaybeUninit<T>, Rows, Cols>;

    #[inline(always)]
    fn uninit(shape: (usize, usize), _: Full) -> Self::Uninit {
        let rows = unsafe { Rows::new_unbound(shape.0) };
        let cols = unsafe { Cols::new_unbound(shape.1) };
        faer::Mat::from_fn(rows, cols, |_, _| const { MaybeUninit::uninit() })
    }

    #[inline(always)]
    unsafe fn set(matrix: &mut Self::Uninit, index: DomainIndex, value: Self::Item) {
        let i = unsafe { matrix.nrows().unchecked_idx(*index.i) };
        let j = unsafe { matrix.ncols().unchecked_idx(*index.j) };
        matrix[(i, j)].write(value);
    }

    #[inline(always)]
    unsafe fn assume_init(value: Self::Uninit) -> Self {
        union Cast<T, Rows: faer::Shape, Cols: faer::Shape> {
            uninit: ManuallyDrop<faer::Mat<MaybeUninit<T>, Rows, Cols>>,
            init: ManuallyDrop<faer::Mat<T, Rows, Cols>>,
        }

        let cast = Cast {
            uninit: ManuallyDrop::new(value),
        };
        ManuallyDrop::into_inner(unsafe { cast.init })
    }
}

unsafe impl<'p, I, T, S> ElementwiseConstructible<S, Pattern<'p, I>>
    for sparse::Mat<T, I, usize, usize>
where
    I: Index,
    S: Fits<Dynamic>,
{
    type Item = T;
    type Uninit = (SparsityInfo<I>, Box<[MaybeUninit<T>]>);

    fn uninit(shape: (usize, usize), domain: Pattern<'p, I>) -> Self::Uninit {
        let info = SparsityInfo::from_pattern(shape, domain);
        let nnz = info.compute_nnz();
        (info, Box::new_uninit_slice(nnz))
    }

    unsafe fn set((_, values): &mut Self::Uninit, index: DomainIndex, value: Self::Item) {
        unsafe { values.get_unchecked_mut(index.pos) }.write(value);
    }

    unsafe fn assume_init(value: Self::Uninit) -> Self {
        Self::from_parts(value.0, unsafe { value.1.assume_init() })
    }
}
