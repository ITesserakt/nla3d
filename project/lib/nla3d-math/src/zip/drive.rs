use core::marker::PhantomData;

use faer::{
    make_guard,
    utils::bound::{Dim, Idx},
};

use crate::dense::SymKind;

use super::{ElementwiseConstructible, Fits, Layout, Producer, Zip};

pub trait Domain {
    fn drive<R>(rows: Dim<'_>, cols: Dim<'_>, initial: R, f: impl FnMut(R, Idx, Idx) -> R) -> R;
}

pub struct Full;

pub struct Triangular<Kind>(PhantomData<Kind>);

impl Domain for Full {
    #[inline]
    fn drive<R>(
        rows: Dim<'_>,
        cols: Dim<'_>,
        mut initial: R,
        mut f: impl FnMut(R, Idx, Idx) -> R,
    ) -> R {
        for j in cols.indices() {
            for i in rows.indices() {
                initial = f(initial, i, j);
            }
        }

        initial
    }
}

impl<K: SymKind> Domain for Triangular<K> {
    #[inline]
    fn drive<R>(
        rows: Dim<'_>,
        cols: Dim<'_>,
        mut initial: R,
        mut f: impl FnMut(R, Idx, Idx) -> R,
    ) -> R {
        assert_eq!(*rows, *cols);

        make_guard!(guard);
        let n = *rows;
        let items = Dim::new(n * (n + 1) / 2, guard);
        for k in items.indices() {
            let (i, j) = K::wide_index(n, *k);
            let i = unsafe { Idx::new_unchecked(i, rows) };
            let j = unsafe { Idx::new_unchecked(j, cols) };
            initial = f(initial, i, j);
        }

        initial
    }
}

macro_rules! impl_zip {
    ($($n:ident: $t:ident$(,)?)+) => {
        impl<D: Domain, $($t: Producer<Domain = D>,)+ L: Layout> Zip<($($t,)+), L> {
            /// Visits every (i, j) exactly once, column-major.
            #[inline(always)]
            fn drive<R>(self, initial: R, mut f: impl FnMut(R, Idx, Idx, $($t::Item,)+) -> R) -> R {
                let (rows, cols) = self.0.0.shape();
                let ($($n,)+) = self.0;
                // Load-bearing for soundness: keep these as real asserts.
                $( assert_eq!((rows, cols), $n.shape()); )+
                let ($(mut $n,)+) = ($($n.into_cursor(),)+);

                make_guard!(guard);
                let rows = Dim::new(rows, guard);
                make_guard!(guard);
                let cols = Dim::new(cols, guard);

                D::drive(rows, cols, initial, |acc, i, j| {
                    f(acc, i, j, $(unsafe { $t::get_unchecked(&mut $n, i, j) }, )+)
                })
            }

            #[inline]
            pub fn for_each(self, mut f: impl FnMut($($t::Item,)+)) {
                self.drive((), |_, _, _, $($n,)+| f($($n,)+));
            }

            #[inline]
            pub fn fold<R>(self, initial: R, mut f: impl FnMut(R, $($t::Item,)+) -> R) -> R {
                self.drive(initial, |acc, _, _, $($n,)+| {
                    f(acc, $($n,)+)
                })
            }

            #[inline(always)]
            fn map_into_inner<M>(self, mut f: impl FnMut($($t::Item,)+) -> M::Item) -> M
            where
                M: ElementwiseConstructible,
            {
                let mut matrix = M::uninit(self.0.0.shape());
                self.drive((), |_, i, j, $($n,)+| {
                    let value = f($($n,)+);
                    unsafe { M::set(&mut matrix, i, j, value) };
                });
                unsafe { M::assume_init(matrix) }
            }

            #[inline]
            pub fn map<R>(self, f: impl FnMut($($t::Item, )+) -> R) -> L::Matrix<R> {
                self.map_into_inner(f)
            }

            #[inline]
            pub fn map_into<M>(self, f: impl FnMut($($t::Item, )+) -> M::Item) -> M
            where
                M: ElementwiseConstructible,
                L: Fits<M::Layout>
            {
                self.map_into_inner(f)
            }
        }
    };
}

impl<T, L> Zip<(T, ), L> {
    #[inline]
    pub fn collect<M>(self) -> M
    where 
        T: Producer,
        M: ElementwiseConstructible<Item = T::Item>,
        L: Fits<M::Layout>
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
