use core::alloc::Layout;
use core::marker::PhantomData;
use core::ptr::NonNull;
use std::ops::Deref;

use faer::sparse::{CreationError, FaerError};
use faer::traits::SignedIndex;
use faer::utils::bound::{Dim, Idx};
use faer::{Index, make_guard};
use platform::collections::vec::Vec;
use platform::compat::boxed::Box;

use super::{DefaultOutput, ElementwiseConstructible, KnownShape, MatrixFor, Producer, Space, Zip};
use crate::dense::SymKind;

#[derive(Debug, Clone, Copy)]
pub struct DomainIndex<'r, 'c> {
    pub i: Idx<'r>,
    pub j: Idx<'c>,
    pub pos: usize,
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an iteration domain",
    label = "expected `Full`, `Triangular<_>` or `Pattern<_>`"
)]
pub unsafe trait Domain: Clone {
    fn drive<R>(
        &self,
        rows: Dim<'_>,
        cols: Dim<'_>,
        initial: R,
        f: impl FnMut(R, DomainIndex) -> R,
    ) -> R;
}

#[diagnostic::on_unimplemented(
    message = "cannot zip a `{Self}` producer with a `{Rhs}` producer",
    label = "these producers visit different sets of positions",
    note = "`Full` combines with every domain and a domain combines with itself; `Triangular<_>` and `Pattern<_>` never combine with each other",
    note = "to read a dense matrix over a triangle use `.lower()` / `.upper()`"
)]
pub unsafe trait Meet<Rhs: Domain>: Domain {
    type Output: Domain;

    fn meet(self, rhs: Rhs) -> Self::Output;
}

#[derive(Clone, Copy, Default)]
pub struct Full;

pub struct Triangular<Kind>(pub PhantomData<Kind>);

#[derive(Debug)]
pub struct PatternInner<I> {
    pub(crate) col_ptr: Box<[I]>,
    pub(crate) col_nnz: Option<Box<[I]>>,
    pub(crate) row_idx: Box<[I]>,
}

#[derive(Debug)]
pub struct Pattern<I = usize>(platform::compat::sync::Arc<PatternInner<I>>);

impl<Kind> Clone for Triangular<Kind> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Kind> Copy for Triangular<Kind> {}

impl<Kind> Default for Triangular<Kind> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<I> Clone for Pattern<I> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<I: PartialEq> PartialEq for Pattern<I> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        platform::compat::sync::Arc::ptr_eq(&self.0, &other.0)
            || self.col_ptr == other.col_ptr
                && self.col_nnz == other.col_nnz
                && self.row_idx == other.row_idx
    }
}

unsafe impl Domain for Full {
    #[inline]
    fn drive<R>(
        &self,
        rows: Dim<'_>,
        cols: Dim<'_>,
        mut initial: R,
        mut f: impl FnMut(R, DomainIndex) -> R,
    ) -> R {
        for j in cols.indices() {
            for i in rows.indices() {
                initial = f(
                    initial,
                    DomainIndex {
                        i,
                        j,
                        pos: *j * *rows + *i,
                    },
                );
            }
        }

        initial
    }
}

unsafe impl<K: SymKind> Domain for Triangular<K> {
    #[inline]
    fn drive<R>(
        &self,
        rows: Dim<'_>,
        cols: Dim<'_>,
        mut initial: R,
        mut f: impl FnMut(R, DomainIndex) -> R,
    ) -> R {
        assert_eq!(*rows, *cols);

        make_guard!(guard);
        let n = *rows;
        let items = Dim::new(n * (n + 1) / 2, guard);
        for k in items.indices() {
            let (i, j) = K::wide_index(n, *k);
            initial = f(
                initial,
                DomainIndex {
                    i: unsafe { Idx::new_unchecked(i, rows) },
                    j: unsafe { Idx::new_unchecked(j, cols) },
                    pos: *k,
                },
            );
        }

        initial
    }
}

unsafe impl<I: Index> Domain for Pattern<I> {
    fn drive<R>(
        &self,
        rows: Dim<'_>,
        cols: Dim<'_>,
        mut initial: R,
        mut f: impl FnMut(R, DomainIndex) -> R,
    ) -> R {
        let col_ptr = &self.col_ptr;
        let row_idx = &self.row_idx;
        for j in cols.indices() {
            let start = col_ptr[*j].zx();
            let end = col_ptr[*j + 1].zx();
            for p in start..end {
                let i = unsafe { Idx::new_unchecked(row_idx[p].zx(), rows) };
                initial = f(initial, DomainIndex { i, j, pos: p });
            }
        }
        initial
    }
}

unsafe impl<D: Domain> Meet<Full> for D {
    type Output = D;

    fn meet(self, _: Full) -> Self::Output {
        self
    }
}

unsafe impl<K: SymKind> Meet<Triangular<K>> for Full {
    type Output = Triangular<K>;

    fn meet(self, rhs: Triangular<K>) -> Self::Output {
        rhs
    }
}

unsafe impl<'p, I: Index> Meet<Pattern<I>> for Full {
    type Output = Pattern<I>;

    fn meet(self, rhs: Pattern<I>) -> Self::Output {
        rhs
    }
}

unsafe impl<K: SymKind> Meet<Triangular<K>> for Triangular<K> {
    type Output = Triangular<K>;

    fn meet(self, rhs: Triangular<K>) -> Self::Output {
        rhs
    }
}

unsafe impl<'p, I: Index> Meet<Pattern<I>> for Pattern<I> {
    type Output = Pattern<I>;

    fn meet(self, rhs: Pattern<I>) -> Self::Output {
        assert_eq!(
            self, rhs,
            "Different sparsity patterns are not allowed: mix them beforehand"
        );
        self
    }
}

impl<I> Deref for Pattern<I> {
    type Target = PatternInner<I>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<I: Clone> Pattern<I> {
    #[inline]
    pub fn from_parts(col_ptr: Box<[I]>, col_nnz: Option<Box<[I]>>, row_idx: Box<[I]>) -> Self {
        Self(platform::compat::sync::Arc::new(PatternInner {
            col_ptr,
            col_nnz,
            row_idx,
        }))
    }

    #[inline]
    fn try_zeroed(len: usize) -> Result<Box<[I]>, FaerError>
    where
        I: Index,
    {
        use platform::allocation::Allocator;

        let ptr = if size_of::<I>() == 0 || len == 0 {
            NonNull::dangling()
        } else {
            let layout = match Layout::array::<I>(len) {
                Ok(l) => l,
                Err(_) => return Err(FaerError::OutOfMemory),
            };
            platform::allocation::Global
                .allocate_zeroed(layout)
                .map_err(|_| FaerError::OutOfMemory)?
                .cast()
        };
        Ok(unsafe { Vec::from_parts(ptr, len, len) }.into_boxed_slice())
    }

    #[inline]
    pub fn try_from_entries(
        rows: usize,
        cols: usize,
        mut entries: Box<[(I, I)]>,
    ) -> Result<Self, CreationError>
    where
        I: Index,
    {
        let max_len = I::Signed::MAX.zx();
        if rows > max_len || cols > max_len || entries.len() > max_len {
            return Err(FaerError::IndexOverflow)?;
        }
        if entries.is_empty() {
            let pattern = PatternInner {
                col_ptr: Self::try_zeroed(cols)?,
                col_nnz: None,
                row_idx: Box::new([]),
            };
            return Ok(Self(platform::compat::sync::Arc::new(pattern)));
        }

        entries.sort_unstable_by_key(|&(row, col)| (col, row));
        let mut col_counts = Self::try_zeroed(cols + 1)?;
        for &(row, col) in &entries {
            let slot = col_counts
                .get_mut(col.zx() + 1)
                .ok_or(CreationError::OutOfBounds {
                    row: row.zx(),
                    col: col.zx(),
                })?;
            *slot += I::truncate(1);
        }
        let col_ptr: Box<_> = col_counts
            .iter()
            .scan(I::truncate(0), |acc, count| {
                *acc += *count;
                Some(*acc)
            })
            .collect();

        let pattern = PatternInner {
            col_ptr,
            col_nnz: None,
            row_idx: entries.into_iter().map(|it| it.0).collect(),
        };

        Ok(Self(platform::compat::sync::Arc::new(pattern)))
    }
}

macro_rules! impl_zip {
    ($($n:ident: $t:ident$(,)?)+) => {
        impl<S, D: Domain, $($t: Producer,)+> Zip<($($t,)+), S, D> {
            /// Visits every (i, j) exactly once, column-major.
            #[inline(always)]
            fn drive<R>(self, initial: R, mut f: impl FnMut(R, DomainIndex, $($t::Item,)+) -> R) -> R
            where
                S: KnownShape
            {
                let (rows, cols) = self.2.materialize();
                let ($($n,)+) = self.0;
                let ($(mut $n,)+) = ($($n.into_cursor(),)+);

                make_guard!(guard);
                let rows = Dim::new(rows, guard);
                make_guard!(guard);
                let cols = Dim::new(cols, guard);

                self.1.drive(rows, cols, initial, |acc, index| {
                    f(acc, index, $(unsafe { $t::get_unchecked(&mut $n, index) }, )+)
                })
            }

            #[inline]
            pub fn for_each(self, mut f: impl FnMut($($t::Item,)+))
            where
                S: KnownShape
            {
                self.drive((), |_, _, $($n,)+| f($($n,)+));
            }

            #[inline]
            pub fn fold<R>(self, initial: R, mut f: impl FnMut(R, $($t::Item,)+) -> R) -> R
            where
                S: KnownShape
            {
                self.drive(initial, |acc, _, $($n,)+| {
                    f(acc, $($n,)+)
                })
            }

            #[inline(always)]
            fn map_into_inner<M>(self, mut f: impl FnMut($($t::Item,)+) -> M::Item) -> M
            where
                M: ElementwiseConstructible<S, D>,
                S: KnownShape
            {
                let shape = self.2.materialize();
                let mut matrix = M::uninit(shape, self.1.clone());
                self.drive((), |_, index, $($n,)+| {
                    let value = f($($n,)+);
                    unsafe { M::set(&mut matrix, index, value) };
                });
                unsafe { M::assume_init(matrix) }
            }

            #[inline]
            pub fn map<R>(self, f: impl FnMut($($t::Item, )+) -> R) -> MatrixFor<S, D, R>
            where
                Space<S, D>: DefaultOutput<Shape = S, Domain = D>,
                S: KnownShape
            {
                self.map_into_inner(f)
            }

            #[inline]
            pub fn map_into<M>(self, f: impl FnMut($($t::Item, )+) -> M::Item) -> M
            where
                M: ElementwiseConstructible<S, D>,
                S: KnownShape
            {
                self.map_into_inner(f)
            }
        }
    };
}

impl<T, S, D: Domain> Zip<(T,), S, D> {
    #[inline]
    pub fn collect<M>(self) -> M
    where
        T: Producer,
        M: ElementwiseConstructible<S, D, Item = T::Item>,
        S: KnownShape,
    {
        self.map_into_inner(|x| x)
    }
}

impl_zip!(a: T1);
impl_zip!(a: T1, b: T2);
impl_zip!(a: T1, b: T2, c: T3);
impl_zip!(a: T1, b: T2, c: T3, d: T4);
impl_zip!(a: T1, b: T2, c: T3, d: T4, e: T5);
impl_zip!(a: T1, b: T2, c: T3, d: T4, e: T5, f: T6);
