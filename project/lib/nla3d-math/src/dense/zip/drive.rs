use faer::{
    make_guard,
    utils::bound::{Dim, Idx},
};

use super::{ElementwiseConstructible, Fits, Layout, Producer, Zip};

macro_rules! impl_zip {
    ($($n:ident: $t:ident$(,)?)+) => {
        impl<$($t: Producer,)+ L: Layout> Zip<($($t,)+), L> {
            /// Visits every (i, j) exactly once, column-major.
            #[inline(always)]
            fn drive(self, mut f: impl FnMut(Idx, Idx, ($($t::Item,)+))) {
                let (rows, cols) = self.0.0.shape();
                let ($($n,)+) = self.0;
                // Load-bearing for soundness: keep these as real asserts.
                $( assert_eq!((rows, cols), $n.shape()); )+
                let ($(mut $n,)+) = ($($n.into_cursor(),)+);

                make_guard!(guard);
                let rows = Dim::new(rows, guard);
                make_guard!(guard);
                let cols = Dim::new(cols, guard);

                for j in cols.indices() {
                    for i in rows.indices() {
                        f(i, j, ($(unsafe { $t::get_unchecked(&mut $n, i, j) },)+));
                    }
                }
            }

            #[inline]
            pub fn for_each(self, mut f: impl FnMut($($t::Item,)+)) {
                self.drive(|_, _, ($($n,)+)| f($($n,)+));
            }

            #[inline]
            pub fn fold<R>(self, initial: R, mut f: impl FnMut(R, $($t::Item,)+) -> R) -> R {
                let mut acc = Some(initial);
                self.drive(|_, _, ($($n,)+)| {
                    acc = Some(f(acc.take().unwrap(), $($n,)+));
                });
                acc.unwrap()
            }

            #[inline]
            fn map_into_inner<M>(self, mut f: impl FnMut($($t::Item,)+) -> M::Item) -> M
            where
                M: ElementwiseConstructible,
            {
                let mut matrix = M::uninit(self.0.0.shape());
                self.drive(|i, j, ($($n,)+)| {
                    let value = f($($n,)+);
                    unsafe { M::set(&mut matrix, i, j, value) };
                });
                unsafe { M::assume_init(matrix) }
            }

            #[inline(always)]
            pub fn map<R>(self, f: impl FnMut($($t::Item, )+) -> R) -> L::Matrix<R> {
                self.map_into_inner(f)
            }

            #[inline(always)]
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

impl_zip!(a: T1);
impl_zip!(a: T1, b: T2);
impl_zip!(a: T1, b: T2, c: T3);
impl_zip!(a: T1, b: T2, c: T3, d: T4);
impl_zip!(a: T1, b: T2, c: T3, d: T4, e: T5);
impl_zip!(a: T1, b: T2, c: T3, d: T4, e: T5, f: T6);
