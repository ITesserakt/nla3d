use core::marker::PhantomData;

use faer::{
    Index, make_guard,
    utils::bound::{Dim, Idx},
};

use crate::dense::SymKind;

use super::{DefaultOutput, ElementwiseConstructible, KnownShape, MatrixFor, Producer, Space, Zip};

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
pub struct Pattern<'p, I = usize> {
    pub(crate) col_ptr: &'p [I],
    pub(crate) col_nnz: Option<&'p [I]>,
    pub(crate) row_idx: &'p [I],
}

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

impl<I> Clone for Pattern<'_, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for Pattern<'_, I> {}

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

unsafe impl<I: Index> Domain for Pattern<'_, I> {
    fn drive<R>(
        &self,
        rows: Dim<'_>,
        cols: Dim<'_>,
        mut initial: R,
        mut f: impl FnMut(R, DomainIndex) -> R,
    ) -> R {
        let col_ptr = self.col_ptr;
        let row_idx = self.row_idx;
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

unsafe impl<'p, I: Index> Meet<Pattern<'p, I>> for Full {
    type Output = Pattern<'p, I>;

    fn meet(self, rhs: Pattern<'p, I>) -> Self::Output {
        rhs
    }
}

unsafe impl<K: SymKind> Meet<Triangular<K>> for Triangular<K> {
    type Output = Triangular<K>;

    fn meet(self, rhs: Triangular<K>) -> Self::Output {
        rhs
    }
}

unsafe impl<'p, I: Index> Meet<Pattern<'p, I>> for Pattern<'p, I> {
    type Output = Pattern<'p, I>;

    fn meet(self, rhs: Pattern<'p, I>) -> Self::Output {
        assert_eq!(
            self, rhs,
            "Different sparsity patterns are not allowed: mix them beforehand"
        );
        self
    }
}

impl<'p, I: PartialEq> PartialEq for Pattern<'p, I> {
    fn eq(&self, other: &Self) -> bool {
        self.col_ptr == other.col_ptr
            && self.col_nnz == other.col_nnz
            && self.row_idx == other.row_idx
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
        S: KnownShape
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
