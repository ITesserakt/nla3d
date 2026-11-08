use core::{fmt::Debug, marker::PhantomData};
use std::collections::btree_set::BTreeSet;

use generic_array::typenum::Const;
use proptest::{
    collection::btree_set,
    prelude::{
        Arbitrary, Strategy,
        prop::{strategy::NewTree, test_runner::TestRunner},
    },
    strategy::ValueTree,
};

use crate::{
    dense::{DMat, Mat, MatSym, MatSymLength, SymKind},
    sparse,
    zip::{
        Domain, Dynamic, ElementwiseConstructible, Full, KnownShape, Pattern, Static, Triangular,
        Zip,
    },
};

pub struct MatrixStrategy<M, S, D, E> {
    shape: S,
    domain: D,
    element: E,
    _phantom: PhantomData<fn() -> M>,
}

pub struct MatrixTree<M, S, D, V> {
    shape: S,
    domain: D,
    trees: Vec<V>,
    next: usize,
    last: Option<usize>,
    _phantom: PhantomData<fn() -> M>,
}

impl<M, S, D, E> MatrixStrategy<M, S, D, E> {
    pub const fn new(shape: S, domain: D, element: E) -> Self {
        Self {
            shape,
            domain,
            element,
            _phantom: PhantomData,
        }
    }
}

impl<M, S, D, E: Debug> Debug for MatrixStrategy<M, S, D, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MatrixStrategy")
            .field("element", &self.element)
            .field("matrix", &std::any::type_name::<M>())
            .finish_non_exhaustive()
    }
}

fn count_points<S: KnownShape, D: Domain>(shape: S, domain: D) -> usize {
    Zip::from_fn(domain, |_| ())
        .with_shape(shape)
        .fold(0, |acc, ()| acc + 1)
}

impl<M, S, D, E> Strategy for MatrixStrategy<M, S, D, E>
where
    S: KnownShape,
    D: Domain,
    E: Strategy,
    M: ElementwiseConstructible<S, D, Item = E::Value> + Debug,
{
    type Tree = MatrixTree<M, S, D, E::Tree>;
    type Value = M;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let len = count_points(self.shape, self.domain.clone());
        let trees = (0..len)
            .map(|_| self.element.new_tree(runner))
            .collect::<Result<_, _>>()?;
        Ok(MatrixTree {
            shape: self.shape,
            domain: self.domain.clone(),
            trees,
            next: 0,
            last: None,
            _phantom: PhantomData,
        })
    }
}

impl<M, S, D, V> ValueTree for MatrixTree<M, S, D, V>
where
    S: KnownShape,
    D: Domain,
    V: ValueTree,
    M: ElementwiseConstructible<S, D, Item = V::Value> + Debug,
{
    type Value = M;

    fn current(&self) -> Self::Value {
        let mut trees = self.trees.iter();
        Zip::from_fn(self.domain.clone(), move |_| {
            trees.next().unwrap().current()
        })
        .with_shape(self.shape)
        .collect()
    }

    fn simplify(&mut self) -> bool {
        while self.next < self.trees.len() {
            if self.trees[self.next].simplify() {
                self.last = Some(self.next);
                return true;
            }
            self.next += 1;
        }
        false
    }

    fn complicate(&mut self) -> bool {
        let Some(k) = self.last else { return false };
        self.next = k;
        if self.trees[k].complicate() {
            true
        } else {
            self.last = None;
            false
        }
    }
}

pub const fn mat<const N: usize, const M: usize, E: Strategy>(
    element: E,
) -> MatrixStrategy<Mat<N, M, E::Value>, Static<N, M>, Full, E> {
    MatrixStrategy::new(Static, Full, element)
}

pub const fn mat_sym<const N: usize, K: SymKind, E: Strategy>(
    element: E,
) -> MatrixStrategy<MatSym<N, E::Value, K>, Static<N, N>, Triangular<K>, E>
where
    Const<N>: MatSymLength,
{
    MatrixStrategy::new(Static, Triangular(PhantomData), element)
}

pub const fn dmat<E: Strategy>(
    rows: usize,
    cols: usize,
    element: E,
) -> MatrixStrategy<DMat<E::Value>, Dynamic, Full, E> {
    MatrixStrategy::new(Dynamic { rows, cols }, Full, element)
}

impl<const N: usize, const M: usize, T: Arbitrary> Arbitrary for Mat<N, M, T> {
    type Parameters = T::Parameters;
    type Strategy = MatrixStrategy<Self, Static<N, M>, Full, T::Strategy>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        mat(T::arbitrary_with(args))
    }
}

impl<const N: usize, T: Arbitrary, K: SymKind> Arbitrary for MatSym<N, T, K>
where
    Const<N>: MatSymLength,
{
    type Parameters = T::Parameters;
    type Strategy = MatrixStrategy<Self, Static<N, N>, Triangular<K>, T::Strategy>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        mat_sym(T::arbitrary_with(args))
    }
}

pub fn sparse_mat<E>(
    rows: usize,
    cols: usize,
    max_per_col: usize,
    element: E,
) -> impl Strategy<Value = sparse::Mat<E::Value>>
where
    E: Strategy + Clone,
{
    use proptest::collection::vec;

    let per_col = max_per_col.min(rows);
    vec(btree_set(0..rows, 0..=per_col), cols).prop_flat_map(move |pattern| {
        let nnz = pattern.iter().map(|it| it.len()).sum::<usize>();
        vec(element.clone(), nnz).prop_map(move |values| build_sparse(rows, cols, &pattern, values))
    })
}

fn build_sparse<T>(
    rows: usize,
    cols: usize,
    pattern: &[BTreeSet<usize>],
    values: Vec<T>,
) -> sparse::Mat<T> {
    let mut col_ptr = Vec::with_capacity(cols + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for column in pattern {
        row_idx.extend(column.iter().copied());
        col_ptr.push(row_idx.len());
    }
    let pattern = Pattern {
        col_ptr: &col_ptr,
        col_nnz: None,
        row_idx: &row_idx,
    };
    let mut values = values.into_iter();

    Zip::from_fn(pattern, |_| values.next().unwrap())
        .with_shape(Dynamic { rows, cols })
        .collect()
}

#[cfg(test)]
mod tests {
    use core::{cell::RefCell, panic::AssertUnwindSafe};
    use std::panic::catch_unwind;

    use matrixcompare::{prop_assert_matrix_eq, prop_assert_scalar_eq};
    use num_traits::Zero;
    use platform::collections::hash_set::HashSet;
    use proptest::{
        prelude::{Just, Strategy, any},
        prop_assert, prop_assert_eq, prop_assume, proptest,
    };

    use crate::{
        arbitrary::{dmat, mat, mat_sym, sparse_mat},
        dense::{Lower, Mat, MatSym, SymKind, Upper},
        zip::{ConsumeProducer, Domain, Dynamic, Full, TriExt, Triangular, Zip},
    };

    fn visit<D: Domain>(shape: Dynamic, d: D) -> Vec<(usize, usize, usize)> {
        let mut v = Vec::new();

        Zip::from_fn(d, |ix| (*ix.i, *ix.j, ix.pos))
            .with_shape(shape)
            .for_each(|it| v.push(it));
        v
    }

    fn sym_case() -> impl Strategy<Value = (usize, usize, usize)> {
        (1usize..5000).prop_flat_map(|n| (Just(n), 0..n, 0..n))
    }

    thread_local!(static DROPS: RefCell<Vec<usize>> = RefCell::new(vec![]));
    struct Tracked(usize);
    impl Drop for Tracked {
        fn drop(&mut self) {
            DROPS.with(|d| d.borrow_mut().push(self.0))
        }
    }

    fn normal_f64() -> impl Strategy<Value = f64> {
        -1000.0..1000.0
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            #[cfg(miri)]
            failure_persistence: None,
            #[cfg(miri)]
            cases: 32,
            ..Default::default()
        })]

        #[test]
        fn mirror_is_consistent_lower(s in mat_sym::<4, Lower, _>(any::<f64>())) {
            prop_assert!(s == s.to_mat());
        }

        #[test]
        fn mirror_is_consistent_upper(s in mat_sym::<4, Upper, _>(any::<f64>())) {
            prop_assert!(s == s.to_mat());
        }

        #[test]
        fn dynamic_shape(m in dmat(5, 7, any::<f64>())) {
            prop_assert_eq!(m.nrows(), 5);
            prop_assert_eq!(m.ncols(), 7);
        }

        #[test]
        fn sparse_stays_in_pattern(m in sparse_mat(5, 4, 3, any::<f64>())) {
            let stored = (0..5)
                .flat_map(|i| (0..4).map(move |j| (i, j)))
                .filter(|&(i, j)| m.view().get(i, j).is_some())
                .count();
            prop_assert_eq!(stored, m.compute_nnz());
            prop_assert!(stored <= 4 * 3);
        }

        #[test]
        fn full_domain(rows in 0usize..12, cols in 0usize..12) {
            let v = visit(Dynamic { rows, cols }, Full);
            prop_assert_eq!(v.len(), rows * cols);

            for (k, p) in v.into_iter().enumerate() {
                prop_assert_eq!(p, (k % rows, k / rows, k));
            }
        }

        #[test]
        fn lower_domain(n in 0usize..20) {
            let v = visit(Dynamic { rows: n, cols: n }, Triangular::<Lower>::default());
            prop_assert_eq!(v.len(), n * (n + 1) / 2);
            let unique: HashSet<_> = v.iter().map(|p| (p.0, p.1)).collect();
            prop_assert_eq!(unique.len(), v.len());
            for (k, (i, j, pos)) in v.into_iter().enumerate() {
                prop_assert!(j <= i && i < n);
                prop_assert_eq!(pos, k);
                prop_assert_eq!(Lower::flat_index(n, i, j), pos);
            }
        }

        #[test]
        fn upper_domain(n in 0usize..20) {
            let v = visit(Dynamic { rows: n, cols: n }, Triangular::<Upper>::default());
            prop_assert_eq!(v.len(), n * (n + 1) / 2);
            let unique: HashSet<_> = v.iter().map(|p| (p.0, p.1)).collect();
            prop_assert_eq!(unique.len(), v.len());
            for (k, (i, j, pos)) in v.into_iter().enumerate() {
                prop_assert!(i <= j && j < n);
                prop_assert_eq!(pos, k);
                prop_assert_eq!(Upper::flat_index(n, i, j), pos);
            }
        }

        #[test]
        fn upper_index((n, i, j) in sym_case()) {
            let k = Upper::flat_index(n, i, j);
            prop_assert!(k < n * (n + 1) / 2);
            prop_assert_eq!(k, Upper::flat_index(n, j, i));
            let (a, b) = Upper::wide_index(n, k);
            prop_assert!(a <= b && (a, b) == (i.min(j), i.max(j)));
            prop_assert_eq!(Upper::flat_index(n, a, b), k);
        }

        #[test]
        fn sym_updated_from_mat(m in mat::<4, 4, _>(any::<i32>())) {
            let mut s = MatSym::<4, i32, Lower>::zero();
            Zip::from(&mut s).and(&m).for_each(|x, y| *x = *y);
            prop_assert_eq!(s, MatSym::from_mat(m));
        }

        #[test]
        fn mat_touched_only_on_triangle(a in mat::<4, 4, _>(any::<i32>()), b in mat::<4, 4, _>(any::<i32>())) {
            let mut out = a;
            Zip::from(&mut out).and(b.lower()).for_each(|x, y| *x = *y);
            for i in 0..4 { for j in 0..4 {
                prop_assert_eq!(out[(i, j)], if i >= j { b[(i, j)] } else { a[(i, j)] });
            }}
        }

        #[test]
        fn consume_drops_each_once(panic_at in 0usize..=9, tri in any::<bool>()) {
            DROPS.with(|d| d.borrow_mut().clear());
            let m = Mat::<3, 3, _>::from_array(std::array::from_fn(|j| std::array::from_fn(|i| Tracked(j * 3 + i))));
            let _ = catch_unwind(AssertUnwindSafe(|| {
                let mut seen = 0;
                let f = |x: Tracked| { if seen == panic_at { panic!() } seen += 1; drop(x) };
                if tri { Zip::from(ConsumeProducer(m).lower()).for_each(f) }
                else   { Zip::from(ConsumeProducer(m)).for_each(f) }
            }));
            let mut d = DROPS.with(|d| d.borrow().clone());
            d.sort();
            prop_assert_eq!(d, (0..9).collect::<Vec<_>>());
        }

        #[test]
        fn add_variants_agree(a in mat::<3, 3, _>(any::<f64>()), b in mat::<3, 3, _>(any::<f64>())) {
            let r = a + b;
            prop_assert_eq!(&a + b, r);
            prop_assert_eq!(a + &b, r);
            prop_assert_eq!(&a + &b, r);
            let (mut c, mut d) = (a, a);
            c += b; d += &b;
            prop_assert_eq!(c, r);
            prop_assert_eq!(d, r);
            prop_assert_eq!(r, b + a);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn det3_matches_faer(m in mat::<3, 3, _>(normal_f64())) {
            let d = m.determinant();
            prop_assume!(d.abs() < 1e9);
            let f =  m.view().determinant();
            prop_assert_scalar_eq!(d, f, comp=abs, tol=1e-5);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn det2_matches_faer(m in mat::<2, 2, _>(normal_f64())) {
            let d = m.determinant();
            prop_assume!(d.abs() < 1e9);
            let f =  m.view().determinant();
            prop_assert_scalar_eq!(d, f, comp=abs, tol=1e-5);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn det1_matches_faer(m in mat::<1, 1, _>(normal_f64())) {
            let d = m.determinant();
            prop_assume!(d.abs() < 1e9);
            let f =  m.view().determinant();
            prop_assert_scalar_eq!(d, f, comp=abs, tol=1e-5);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn inverse_involution3(m in mat::<3, 3, _>(normal_f64())) {
            prop_assume!(m.determinant().abs() > 1.0);
            let back = m.inv().inv();
            prop_assert_matrix_eq!(m, back, comp=abs, tol=1e-6);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn inverse_involution2(m in mat::<2, 2, _>(normal_f64())) {
            prop_assume!(m.determinant().abs() > 1.0);
            let back = m.inv().inv();
            prop_assert_matrix_eq!(m, back, comp=abs, tol=1e-6);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn inverse_involution1(m in mat::<1, 1, _>(normal_f64())) {
            prop_assume!(m.determinant().abs() > 1.0);
            let back = m.inv().inv();
            prop_assert_matrix_eq!(m, back, comp=abs, tol=1e-6);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn det3_sym_matches_faer(m in mat_sym::<3, Lower, _>(normal_f64())) {
            let d = m.determinant();
            prop_assume!(d.abs() < 1e9);
            let f =  m.to_mat().determinant();
            prop_assert_scalar_eq!(d, f, comp=abs, tol=1e-5);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn det2_sym_matches_faer(m in mat_sym::<2, Lower, _>(normal_f64())) {
            let d = m.determinant();
            prop_assume!(d.abs() < 1e9);
            let f =  m.to_mat().determinant();
            prop_assert_scalar_eq!(d, f, comp=abs, tol=1e-5);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn det1_sym_matches_faer(m in mat_sym::<1, Lower, _>(normal_f64())) {
            let d = m.determinant();
            prop_assume!(d.abs() < 1e9);
            let f =  m.to_mat().determinant();
            prop_assert_scalar_eq!(d, f, comp=abs, tol=1e-5);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn inverse_involution3_sym(m in mat_sym::<3, Lower, _>(normal_f64())) {
            prop_assume!(m.determinant().abs() > 1.0);
            let back = m.inv().inv();
            prop_assert_matrix_eq!(m, back, comp=abs, tol=1e-6);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn inverse_involution2_sym(m in mat_sym::<2, Lower, _>(normal_f64())) {
            prop_assume!(m.determinant().abs() > 1.0);
            let back = m.inv().inv();
            prop_assert_matrix_eq!(m, back, comp=abs, tol=1e-6);
        }

        #[test]
        #[allow(semicolon_in_expressions_from_non_local_macros)]
        fn inverse_involution1_sym(m in mat_sym::<1, Lower, _>(normal_f64())) {
            prop_assume!(m.determinant().abs() > 1.0);
            let back = m.inv().inv();
            prop_assert_matrix_eq!(m, back, comp=abs, tol=1e-6);
        }
    }
}
